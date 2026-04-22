from __future__ import annotations

import json
import shutil
import warnings
import os
import inspect 
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union, Any, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import networkx as nx
import dill

# osmnx (optional at import time, required when used)
try:
    import osmnx as ox
except ImportError:
    ox = None  # type: ignore

from . import graph_processing
from . import geometry_utils
from . import osm
from . import plot_helpers
from . import scoring
from . import poi_utils

# --- Types and constants ---
GeoFmt = Literal["geoparquet", "parquet", "fgb", "gpkg", "geojson", "shp"]
_FORMAT_VERSION = "1.0"

_FMT_EXT: Dict[str, str] = {
    "geoparquet": ".geoparquet",
    "parquet": ".geoparquet",
    "fgb": ".fgb",
    "gpkg": ".gpkg",
    "geojson": ".geojson",
    "shp": ".shp",
}

# --- Low-level I/O Helpers ---

def _save_gdf(gdf: gpd.GeoDataFrame, directory: Path, stem: str, fmt: str) -> str:
    """Save a GeoDataFrame in the requested format."""
    fmt = fmt.lower()
    if fmt not in _FMT_EXT:
        raise ValueError(f"Unsupported format: '{fmt}'. Options: {list(_FMT_EXT)}")

    dest = directory / f"{stem}{_FMT_EXT[fmt]}"
    if gdf.index.name and gdf.index.name not in gdf.columns:
        gdf = gdf.reset_index()

    if fmt in ("geoparquet", "parquet"):
        gdf.to_parquet(dest, compression="zstd", geometry_encoding="geoarrow")
    elif fmt == "fgb":
        gdf.to_file(dest, driver="FlatGeobuf")
    elif fmt == "gpkg":
        gdf.to_file(dest, driver="GPKG")
    elif fmt == "geojson":
        gdf.to_file(dest, driver="GeoJSON")
    elif fmt == "shp":
        gdf.to_file(dest, driver="ESRI Shapefile")
    return dest.name

def _load_gdf(directory: Path, filename: str) -> gpd.GeoDataFrame:
    """Load a GeoDataFrame, auto-detecting format from extension."""
    p = directory / filename
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() in (".geoparquet", ".parquet"):
        return gpd.read_parquet(p)
    return gpd.read_file(p)

def _sanitize_col(series: pd.Series) -> pd.Series:
    """Serialize list/dict values to JSON strings for columnar formats."""
    if series.dtype != object:
        return series
    return series.apply(
        lambda x: json.dumps(x, default=str) if isinstance(x, (list, dict)) else x
    )

def _deserialize_col(series: pd.Series) -> pd.Series:
    """Restore list/dict values from JSON strings."""
    if series.dtype != object:
        return series

    def _try(v):
        if not isinstance(v, str):
            return v
        s = v.strip()
        if s and s[0] in ("[", "{"):
            try:
                return json.loads(s)
            except (json.JSONDecodeError, ValueError):
                pass
        return v
    return series.apply(_try)

def _save_graph(G: nx.MultiDiGraph, directory: Path, fmt: str) -> Dict[str, str]:
    """Persist OSMnx graph, dropping redundant node geometry in Parquet."""
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    for col in nodes_gdf.select_dtypes("object").columns:
        nodes_gdf[col] = _sanitize_col(nodes_gdf[col])
    for col in edges_gdf.select_dtypes("object").columns:
        edges_gdf[col] = _sanitize_col(edges_gdf[col])

    fmt_norm = fmt.lower()
    nodes_flat = nodes_gdf.reset_index()

    if fmt_norm in ("geoparquet", "parquet"):
        nodes_df = pd.DataFrame(nodes_flat.drop(columns=["geometry"]))
        nodes_file = "nodes.parquet"
        nodes_df.to_parquet(directory / nodes_file, compression="zstd")
    else:
        nodes_file = _save_gdf(nodes_flat, directory, "nodes", fmt)

    edges_flat = edges_gdf.reset_index()
    edges_file = _save_gdf(edges_flat, directory, "edges", fmt)

    graph_attrs = {k: str(v) for k, v in G.graph.items()}
    (directory / "graph_attrs.json").write_text(
        json.dumps(graph_attrs, indent=2), encoding="utf-8"
    )

    return {
        "nodes_file": nodes_file,
        "edges_file": edges_file,
        "graph_attrs_file": "graph_attrs.json",
    }

def _load_graph(directory: Path, nodes_file: str, edges_file: str) -> nx.MultiDiGraph:
    """Reconstruct an OSMnx graph from stored files."""
    graph_attrs = json.loads((directory / "graph_attrs.json").read_text(encoding="utf-8"))

    nodes_path = directory / nodes_file
    is_flat_parquet = nodes_path.suffix.lower() == ".parquet" and not nodes_file.endswith(".geoparquet")

    if is_flat_parquet:
        nodes_df = pd.read_parquet(nodes_path)
        for col in nodes_df.select_dtypes("object").columns:
            nodes_df[col] = _deserialize_col(nodes_df[col])
        crs = graph_attrs.get("crs", "EPSG:4326")
        nodes_gdf = gpd.GeoDataFrame(
            nodes_df,
            geometry=gpd.points_from_xy(nodes_df["x"], nodes_df["y"]),
            crs=crs,
        )
    else:
        nodes_gdf = _load_gdf(directory, nodes_file)
        for col in nodes_gdf.select_dtypes("object").columns:
            nodes_gdf[col] = _deserialize_col(nodes_gdf[col])

    if "osmid" in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.set_index("osmid")

    edges_gdf = _load_gdf(directory, edges_file)
    for col in edges_gdf.select_dtypes("object").columns:
        edges_gdf[col] = _deserialize_col(edges_gdf[col])

    for c in ("u", "v", "key"):
        if c in edges_gdf.columns:
            edges_gdf[c] = edges_gdf[c].astype(int)
    edges_gdf = edges_gdf.set_index(["u", "v", "key"])

    if "length" not in edges_gdf.columns:
        warnings.warn("Column 'length' missing; recalculating.", UserWarning)
        edges_gdf["length"] = edges_gdf.geometry.length

    return ox.graph_from_gdfs(nodes_gdf, edges_gdf, graph_attrs=graph_attrs)

# --- Manifest and directory helpers ---

def _to_serializable(obj: Any) -> Any:
    """Recursively convert NumPy types and tuples to JSON-safe formats."""
    if obj is None:
        return None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (tuple, list)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj

def _write_manifest(directory: Path, class_name: str, meta: dict) -> None:
    manifest = {"class": class_name, "version": _FORMAT_VERSION, **meta}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

def _read_manifest(directory: Path) -> dict:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

def _check_class(manifest: dict, expected: str) -> None:
    if manifest.get("class") != expected:
        raise TypeError(f"Manifest mismatch: expected {expected}, got {manifest.get('class')}")

def _ensure_dir(path: Union[str, Path], overwrite: bool) -> Path:
    p = Path(path)
    if p.exists():
        if not overwrite:
            raise FileExistsError(f"Directory '{p}' already exists. Pass overwrite=True.")
        shutil.rmtree(p)
    p.mkdir(parents=True)
    return p

def _ensure_gdf(data: Any) -> gpd.GeoDataFrame:
    if isinstance(data, gpd.GeoDataFrame):
        return data
    if isinstance(data, gpd.GeoSeries):
        return gpd.GeoDataFrame(geometry=data, crs=data.crs)
    if isinstance(data, (str, Path)):
        return geometry_utils.read_geofile(str(data))
    raise TypeError(f"Unsupported input type: {type(data)}")

def _bounds_from(aoi: Any) -> Union[gpd.GeoDataFrame, gpd.GeoSeries]:
    if hasattr(aoi, 'data'):
        return aoi.data
    return aoi


# --- Main Classes ---

class MultiresPolygonData:
    """Handles hierarchical spinal layers and named side-branch polygon datasets.

    The spinal hierarchy (e.g., State > County > Block Group) is managed via 
    absolute resolutions (0, 1, 2...). Standard Python indexing is supported, 
    so [-1] returns the highest resolution spinal layer.
    
    Special layers (e.g., 'City') are side-branches linked to specific spinal 
    resolutions but exist outside the linear integer sequence.
    """

    def __init__(
        self,
        gdfs: Union[List[gpd.GeoDataFrame], Dict[str, gpd.GeoDataFrame]],
        agg_method: Dict[str, str]
    ) -> None:
        """Initializes MultiresPolygonData.

        Args:
            gdfs: List of GDFs (ordered 0 to N) or dict of {name: GDF}.
            agg_method: Mapping of column names to aggregation methods.
        """
        self.agg_method = agg_method
        self.gdfs: Dict[Union[int, str], gpd.GeoDataFrame] = {}
        self.links: List[Tuple[Union[int, str], Union[int, str]]] = []
        self._name_to_res: Dict[str, int] = {}

        # Initial processing
        if isinstance(gdfs, list):
            for i, gdf in enumerate(gdfs):
                self.add_layer(resolution=i, data=gdf)
        else:
            for i, (name, gdf) in enumerate(gdfs.items()):
                self.add_layer(name=name, resolution=i, data=gdf)

    def __getitem__(self, key: Union[int, str]) -> gpd.GeoDataFrame:
        """Access a layer by hierarchy index, name, or resolution alias.

        Args:
            key: Int (0, 1, -1) for hierarchy or Str ('city', 'blockgroup') for names.

        Returns:
            gpd.GeoDataFrame: The requested layer.
        """
        if isinstance(key, int):
            h_keys = sorted([k for k in self.gdfs.keys() if isinstance(k, int)])
            if not h_keys:
                raise IndexError("No spinal hierarchy layers available.")
            # Map index (including negative) to actual resolution key
            actual_key = h_keys[key]
            return self.gdfs[actual_key]
        
        if key in self.gdfs:
            return self.gdfs[key]
        
        if key in self._name_to_res:
            return self.gdfs[self._name_to_res[key]]
        
        raise KeyError(f"Layer '{key}' not found.")

    def _get_ref_key(self, key: Union[int, str]) -> Union[int, str]:
        """Resolve a user-provided key to the internal dictionary key."""
        if isinstance(key, str):
            if key in self._name_to_res:
                return self._name_to_res[key]
            return key
        if isinstance(key, int):
            h_keys = sorted([k for k in self.gdfs.keys() if isinstance(k, int)])
            return h_keys[key]
        return key

    def add_layer(
        self,
        data: gpd.GeoDataFrame,
        name: Optional[str] = None,
        resolution: Optional[int] = None,
        agg_from: Optional[Union[int, str]] = None
    ) -> None:
        """Adds a layer to the multi-resolution system.

        Args:
            data: Polygon GeoDataFrame to add.
            name: Name of the layer. Required if resolution is None.
            resolution: Integer level in the spinal hierarchy.
            agg_from: For special layers (no resolution), the layer to sample from.

        Raises:
            ValueError: If parameters violate hierarchy or naming rules.
        """
        if name is None and resolution is None:
            raise ValueError("Must provide either a 'name' or a 'resolution' to add a layer.")
        
        if resolution is not None and agg_from is not None:
            raise ValueError("Hierarchy layers (with 'resolution') are auto-linked; 'agg_from' is not allowed.")
        
        if resolution is None and agg_from is None:
            raise ValueError("Special layers (string name only) require 'agg_from' to define data source.")

        # Standardize Geometry
        data = data.copy()
        if data.geometry.name != "geometry":
            data = data.rename(columns={data.geometry.name: "geometry"}).set_geometry("geometry")
        if "area" not in data.columns:
            data["area"] = geometry_utils.area(data)

        # 1. Determine Internal Key
        # Hierarchy layers use int keys; Special layers use str keys.
        key = resolution if resolution is not None else name
        self.gdfs[key] = data

        # 2. Update Linkage and Metadata
        if resolution is not None:
            if name is not None:
                self._name_to_res[name] = resolution
            if resolution > 0:
                # Hierarchy auto-links to the level above it
                self.links.append((resolution, resolution - 1))
        else:
            # Special layer links to the user-specified agg_from layer
            ref_key = self._get_ref_key(agg_from)
            # Linkage direction: Source (child) -> Name (parent/target)
            self.links.append((ref_key, name))

        self._fill_columns_internal()

    def _resolution_mapping(self, parent_key: Union[int, str], child_key: Union[int, str]) -> None:
        """Centroid-based spatial join to link child units to exactly one parent."""
        parent_gdf = self.gdfs[parent_key]
        child_gdf = self.gdfs[child_key]

        centroids = child_gdf.copy()
        centroids.geometry = child_gdf.geometry.centroid
        
        parent_geom_only = parent_gdf[[parent_gdf.geometry.name]]
        joined = centroids.sjoin(parent_geom_only, how="left", predicate="intersects")
        
        # Take first match to ensure strictly 1-to-1 mapping
        joined = joined.groupby(level=0).first()
        
        idx_cols = [c for c in joined.columns if c.startswith("index_")]
        if not idx_cols:
            raise RuntimeError(f"Join failed between child {child_key} and parent {parent_key}")
        
        right_idx_col = idx_cols[0]
        self.gdfs[child_key][f"parent_{parent_key}_idx"] = joined[right_idx_col]
        
        counts = joined[right_idx_col].value_counts()
        self.gdfs[parent_key][f"child_{child_key}_count"] = (
            self.gdfs[parent_key].index.map(counts).fillna(0).astype(int)
        )

    def _propagate(self, src: Union[int, str], tgt: Union[int, str], direction: Literal["up", "down"]) -> bool:
        """Attribute propagation between linked layers if column is missing in target."""
        parent_key = tgt if direction == "up" else src
        child_key = src if direction == "up" else tgt
        
        source_gdf = self.gdfs[src]
        target_gdf = self.gdfs[tgt]
        
        cols = [c for c in self.agg_method if c in source_gdf.columns and c not in target_gdf.columns]
        if not cols: 
            return False

        mapping_col = f"parent_{parent_key}_idx"
        if mapping_col not in self.gdfs[child_key].columns:
            self._resolution_mapping(parent_key, child_key)

        if direction == "up":
            temp_child = self.gdfs[child_key].copy()
            ops = {}
            for c in cols:
                method = self.agg_method[c]
                if method.startswith("density_"):
                    denom = method[len("density_"):]
                    temp_child[c] = temp_child[c] * temp_child[denom]
                    ops[c] = "sum"
                else:
                    ops[c] = method
            
            res = temp_child.groupby(mapping_col).agg(ops)
            self.gdfs[parent_key] = target_gdf.merge(res, left_index=True, right_index=True, how="left")
            for c in cols:
                if self.agg_method[c].startswith("density_"):
                    denom = self.agg_method[c][len("density_"):]
                    self.gdfs[parent_key][c] /= self.gdfs[parent_key][denom].replace(0, np.nan)
        else:
            temp_parent = self.gdfs[parent_key][cols].copy()
            for c in cols:
                if self.agg_method[c] == "sum":
                    count_col = f"child_{child_key}_count"
                    temp_parent[c] /= self.gdfs[parent_key][count_col].replace(0, np.nan)
            
            self.gdfs[child_key] = target_gdf.merge(temp_parent, left_on=mapping_col, right_index=True, how="left")
        
        return True

    def _fill_columns_internal(self) -> None:
        """Consistency pass across all links."""
        for _ in range(len(self.gdfs) + 2):
            changed = False
            for child, parent in self.links:
                if self._propagate(child, parent, "up"): changed = True
            for child, parent in self.links:
                if self._propagate(parent, child, "down"): changed = True
            if not changed: break

    def add_feature(
        self, 
        data: gpd.GeoDataFrame, 
        agg_method: Union[str, Dict[str, str]], 
        to_resolution: Union[str, int] = 0,
        fill = None
    ) -> None:
        """Resamples external data into a layer and triggers propagation."""
        res_key = self._get_ref_key(to_resolution)
        
        if data.geometry.name != "geometry":
            data = data.rename(columns={data.geometry.name: "geometry"}).set_geometry("geometry")
        
        columns = []
        for c in data.columns:
            if c != "geometry":
                self.agg_method[c] = agg_method if isinstance(agg_method, str) else agg_method.get(c, "sum")

            if c not in self.gdfs[res_key].columns:
                columns.append(c)
 
        if len(columns) == 0:
            return self.gdfs[res_key] 
        
        self.gdfs[res_key] = geometry_utils.resample_gdf(
            data, self.gdfs[res_key], method="sum" if isinstance(agg_method, str) else "max"
        )
        if fill is not None:
            if isinstance(fill,dict):
                for col, value in fill.items():
                    if col in columns:
                        self.gdfs[res_key][col] = self.gdfs[res_key][col].fillna(value)
            else:
                self.gdfs[res_key][columns] = self.gdfs[res_key][columns].fillna(fill)

        self._fill_columns_internal()
        
        return self.gdfs[res_key]

    def save(self, path: Union[str, Path], overwrite: bool = True) -> Path:
        directory = _ensure_dir(path, overwrite)
        meta = {
            "class": "MultiresPolygonData",
            "version": _FORMAT_VERSION,
            "agg_method": self.agg_method,
            "links": self.links,
            "name_to_res": self._name_to_res,
            "files": {str(k): f"layer_{k}.geoparquet" for k in self.gdfs}
        }
        for k, gdf in self.gdfs.items():
            gdf.to_parquet(directory / meta["files"][str(k)])
        _write_manifest(directory, "MultiresPolygonData", meta)
        return directory

    @classmethod
    def load(cls, path: Union[str, Path]) -> MultiresPolygonData:
        directory = Path(path)
        manifest = _read_manifest(directory)
        obj = cls.__new__(cls)
        obj.agg_method = manifest["agg_method"]
        obj.links = [tuple(link) for link in manifest["links"]]
        obj._name_to_res = manifest.get("name_to_res", {})
        obj.gdfs = {}
        for k_str, file in manifest["files"].items():
            key = int(k_str) if k_str.isdigit() else k_str
            obj.gdfs[key] = gpd.read_parquet(directory / file)
        return obj

    def __repr__(self) -> str:
        layers = sorted([str(k) for k in self.gdfs.keys() if isinstance(k, int)])
        specials = [str(k) for k in self.gdfs.keys() if isinstance(k, str) and k not in self._name_to_res]
        return f"MultiresPolygonData(Spinal={layers}, Special={specials})"
    
# ──────────────────────────────────────────────────────────────────────────────
# 1. AreaOfInterest
# ──────────────────────────────────────────────────────────────────────────────

class AreaOfInterest:
    """
    Define and manage the area of interest (AOI) for an analysis.

    Parameters
    ----------
    data : str | GeoDataFrame | GeoSeries
        City name (geocoded automatically), path to a geospatial file, or
        a GeoDataFrame / GeoSeries with the AOI geometry.
    crs : int, optional
        Output CRS. Defaults to 4326 (WGS 84).
    buffer : float, optional
        Buffer distance in metres applied after loading the geometry.

    Persistence
    -----------
    >>> aoi.save("output/my_aoi")                  # GeoParquet (default)
    >>> aoi.save("output/my_aoi", fmt="geojson")
    >>> aoi.save("output/my_aoi", fmt="gpkg")
    >>> aoi = AreaOfInterest.load("output/my_aoi")
    """

    def __init__(
        self,
        data: Union[str, gpd.GeoDataFrame, gpd.GeoSeries],
        crs: int|None = None,
        buffer: float = 0.0,
    ):
        if isinstance(data, str) and not Path(data).exists():
            gdf = utils.get_city_geometry(data)
        else:
            gdf = _ensure_gdf(data)

        if crs is None:
            crs = 4326

        gdf = gdf.to_crs(crs)
        crs = gdf.crs.copy()

        if buffer > 0:
            if not gdf.crs.is_projected:
                gdf = gdf.to_crs(gdf.estimate_utm_crs())

            gdf = gdf.assign(geometry=gdf.geometry.buffer(buffer))
            gdf = gdf.to_crs(crs)

        self.data: gpd.GeoDataFrame = gdf
        self.crs = crs
        self._buffer = buffer

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def geometry(self) -> gpd.GeoSeries:
        return self.data.geometry

    @property
    def union(self):
        """Single union geometry of all AOI features."""
        return self.data.union_all()

    @property
    def bounds(self) -> tuple:
        return tuple(self.data.total_bounds)

    # ── Methods ───────────────────────────────────────────────────────────────

    def clip(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Clip a GeoDataFrame to the AOI extent."""
        return gdf.to_crs(self.crs).loc[
            gdf.intersects(self.union)
        ]

    def plot(self, **kwargs):
        """Interactive Folium map of the AOI."""
        return plot_helpers.general_map(aoi=self.data, **kwargs)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        path: Union[str, Path],
        fmt: GeoFmt = "geoparquet",
        overwrite: bool = True,
    ) -> Path:
        """
        Save the AOI to disk.

        Parameters
        ----------
        path : str | Path
            Output directory (created if it does not exist).
        fmt : GeoFmt
            Output format: "geoparquet" | "fgb" | "gpkg" | "geojson" | "shp".
        overwrite : bool
            If True (default), an existing directory is replaced.

        Returns
        -------
        Path
            Path of the created directory.

        Examples
        --------
        >>> aoi.save("data/berlin_aoi")
        >>> aoi.save("data/berlin_aoi", fmt="geojson")
        """
        directory = _ensure_dir(path, overwrite)
        filename = _save_gdf(self.data, directory, "aoi", fmt)
        _write_manifest(directory, "AreaOfInterest", {
            "fmt": fmt, "crs": self.crs, "buffer": self._buffer, "aoi_file": filename,
        })
        return directory

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AreaOfInterest":
        """
        Load an AreaOfInterest from a directory created by ``save()``.

        Parameters
        ----------
        path : str | Path
            Persistence directory.

        Examples
        --------
        >>> aoi = AreaOfInterest.load("data/berlin_aoi")
        """
        directory = Path(path)
        manifest = _read_manifest(directory)
        _check_class(manifest, "AreaOfInterest")

        obj = cls.__new__(cls)
        obj.data = _load_gdf(directory, manifest["aoi_file"])
        obj.crs = manifest.get("crs", 4326)
        obj._buffer = manifest.get("buffer", 0.0)
        return obj

    def __repr__(self) -> str:
        return f"AreaOfInterest(geometries={len(self.data)}, crs={self.data.crs.to_epsg()})"


class StreetNetwork:
    """OSMnx-based street network graph handler.

    Args:
        G (nx.MultiDiGraph): The underlying graph object.
        min_edge_length (Optional[float]): Minimum edge length for simplification.
        undirected (Optional[bool]): Whether the graph should be undirected.
    """

    def __init__(
        self,
        G: nx.MultiDiGraph,
        min_edge_length: Optional[float] = None,
        undirected: Optional[bool] = None,
    ):
        self._G = G
        self._nodes: Optional[gpd.GeoDataFrame] = None
        self._edges: Optional[gpd.GeoDataFrame] = None

        # Simplification logic (Triggered only on explicit user input)
        if min_edge_length is not None or undirected is not None:
            self.min_edge_length = min_edge_length if min_edge_length is not None else 0.0
            self.undirected = undirected if undirected is not None else False
            self._G = graph_processing.simplify_graph(
                self._G,
                min_edge_length=self.min_edge_length,
                min_edge_separation=self.min_edge_length * 2,
                undirected=self.undirected,
            )
        else:
            self.min_edge_length = 0.0
            self.undirected = False

    # --- Factory Methods ---

    @classmethod
    def from_xml(cls, xml_file: Union[str, Path], **kwargs) -> StreetNetwork:
        """Create a network from a .osm or .xml file."""
        return cls(G=ox.graph_from_xml(str(xml_file)), **kwargs)

    @classmethod
    def from_graphml(cls, graphml_file: Union[str, Path], **kwargs) -> StreetNetwork:
        """Create a network from a .graphml file."""
        return cls(G=ox.load_graphml(str(graphml_file)), **kwargs)

    @classmethod
    def from_gdfs(cls, nodes: Any, edges: Any, **kwargs) -> StreetNetwork:
        """Create a network from nodes and edges GeoDataFrames or files."""
        gdf_nodes = _ensure_gdf(nodes)
        gdf_edges = _ensure_gdf(edges)
        if "osmid" in gdf_nodes.columns: gdf_nodes = gdf_nodes.set_index("osmid")
        if all(c in gdf_edges.columns for c in ["u", "v", "key"]):
            gdf_edges = gdf_edges.set_index(["u", "v", "key"])
        return cls(G=ox.graph_from_gdfs(gdf_nodes, gdf_edges), **kwargs)

    @classmethod
    def from_pbf(
        cls, 
        pbf_path: Union[str, Path], 
        aoi: Any, 
        xml_file: Optional[Union[str, Path]] = None,
        network_type: Optional[str] = None,
        **kwargs
    ) -> StreetNetwork:
        """Create a network from a PBF file using Osmium for filtering and cropping."""
        _pbf = Path(pbf_path)
        if not _pbf.is_file():
            warnings.warn(f"PBF file not found. Downloading best matching OSM PBF to {_pbf}")

        xml_out = Path(xml_file) if xml_file else _pbf.with_suffix(".osm")
        if network_type is not None:
            network_type = osm.osmium_network_filter(network_type)

        osm.geofabrik_to_osm(
            str(xml_out),
            input_file=str(_pbf),
            aoi=_bounds_from(aoi),
            osmium_filter_args=network_type,
            overwrite=False
        )
        return cls(G=ox.graph_from_xml(str(xml_out)), **kwargs)

    @classmethod
    def from_overpass(
        cls, 
        aoi: Any, 
        network_type: str = "walk", 
        custom_filter: Optional[str] = None, 
        **kwargs
    ) -> StreetNetwork:
        """Download and create a network from OpenStreetMap via Overpass API."""
        G = osm.download_street_graph(
            _bounds_from(aoi), 
            network_type=network_type, 
            custom_filter=custom_filter
        )
        return cls(G=G, **kwargs)

    # --- Properties and Methods ---

    @property
    def graph(self) -> nx.MultiDiGraph:
        """Underlying NetworkX MultiDiGraph."""
        return self._G

    @property
    def nodes(self) -> gpd.GeoDataFrame:
        """Node GeoDataFrame (cached)."""
        if self._nodes is None: self._nodes, self._edges = ox.graph_to_gdfs(self._G)
        return self._nodes

    @property
    def edges(self) -> gpd.GeoDataFrame:
        """Edge GeoDataFrame (cached)."""
        if self._edges is None: self._nodes, self._edges = ox.graph_to_gdfs(self._G)
        return self._edges

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Alias for edges."""
        return self.edges

    def simplify(self, min_edge_length: float = 0.0, undirected: bool = False, **kwargs) -> StreetNetwork:
        """Return a new simplified StreetNetwork."""
        new_G = graph_processing.simplify_graph(
            self._G, min_edge_length=min_edge_length, 
            min_edge_separation=min_edge_length * 2 if min_edge_length > 0 else 0,
            undirected=undirected, **kwargs
        )
        return StreetNetwork(G=new_G, min_edge_length=min_edge_length, undirected=undirected)

    def crop(self, aoi: Any) -> StreetNetwork:
        """Return a new StreetNetwork clipped to an AOI."""
        aoi_gdf = _bounds_from(aoi)
        return StreetNetwork(
            G=graph_processing.crop_by_aoi(aoi_gdf, G=self._G),
            min_edge_length=self.min_edge_length,
            undirected=self.undirected
        )

    def add_points(self, points: Any, max_dist: Optional[float] = None, min_edge_length: float = 0.0) -> Tuple[StreetNetwork, List[int]]:
        """Project points onto the graph."""
        new_G, osmids = graph_processing.add_points_to_graph(points, self._G, max_dist=max_dist, min_edge_length=min_edge_length)
        return StreetNetwork(G=new_G, min_edge_length=self.min_edge_length, undirected=self.undirected), osmids

    def save(self, path: Union[str, Path], fmt: GeoFmt = "geoparquet", overwrite: bool = True) -> Path:
        """Save the network to disk."""
        directory = _ensure_dir(path, overwrite)
        file_refs = _save_graph(self._G, directory, fmt)
        _write_manifest(directory, "StreetNetwork", {
            "fmt": fmt,
            "min_edge_length": self.min_edge_length,
            "undirected": self.undirected,
            **file_refs,
        })
        return directory

    @classmethod
    def load(cls, path: Union[str, Path]) -> StreetNetwork:
        """Load from a saved directory."""
        directory = Path(path)
        manifest = _read_manifest(directory)
        _check_class(manifest, "StreetNetwork")
        G = _load_graph(directory, manifest["nodes_file"], manifest["edges_file"])
        return cls(G=G, min_edge_length=manifest.get("min_edge_length"), undirected=manifest.get("undirected"))

    def __repr__(self) -> str:
        return f"StreetNetwork(nodes={self._G.number_of_nodes()}, edges={self._G.number_of_edges()}, min_edge_length={self.min_edge_length})"
    

class AccessScore:
    """End-to-end scoring system for accessibility evaluation.

    Combines POI attribute scoring, distance-based access functions, 
    and grid-based evaluation into a serializable pipeline.

    Attributes:
        poi_score_func (Callable): Function for scoring POI attributes.
        access_score_func (Callable): Function combining distance and POI scores.
        n_steps (int): Resolution used for grids and normalization.
        poi_param_names (List[str]): Names of parameters in the POI function.
        worst_score (Optional[Sequence]): Calibration anchor for minimum utility.
        best_score (Optional[Sequence]): Calibration anchor for maximum utility.
    """

    def __init__(
        self,
        access_score_func: Callable[..., float],
        poi_score_func: Optional[Callable[...]] = lambda score:score,
        worst_score: Optional[Sequence[Any]] = None,
        best_score: Optional[Sequence[Any]] = None,
        variable_bounds: Optional[Any] = None,
        n_steps: int = 10,
        poi_param_names: Optional[List[str]] = None,
    ) -> None:
        """Initializes the AccessScore model.

        Args:
            poi_score_func: Function scoring POI attributes.
            access_score_func: Function combining distance + POI score.
            worst_score: Lower-bound calibration point (distance, *poi_args).
            best_score: Upper-bound calibration point (distance, *poi_args).
            variable_bounds: Grid definitions for inputs.
            n_steps: Normalization scale. Defaults to 10.
            poi_param_names: Explicit parameter names for the POI function.
        """
        self.n_steps = n_steps
        self.poi_score_func = poi_score_func
        
        if poi_param_names is None:
            sig = inspect.signature(self.poi_score_func)
            self.poi_param_names = list(sig.parameters.keys())
        else:
            self.poi_param_names = poi_param_names

        self.access_score_func = access_score_func

        # Pipeline function: distance, *poi_args
        self.function = lambda distance, *args: self.access_score_func(
            distance,
            self.poi_score_func(*args),
        )

        self.variable_bounds = variable_bounds
        self.distance_grid: Optional[np.ndarray] = None
        self.poi_params_grid: Optional[List[np.ndarray]] = None
        self.poi_grid: Optional[List[float]] = None
        self.scoring_matrix: Optional[pd.DataFrame] = None
        self.worst_score = worst_score
        self.best_score = best_score

        # Auto-calibrate if anchors are provided
        if worst_score is not None and best_score is not None:
            self.calibrate(worst_score, best_score)

        # Build grid if bounds are provided
        if variable_bounds is not None:
            self.create_grid(variable_bounds)

    def calibrate(
        self,
        worst_score: Sequence[Any],
        best_score: Sequence[Any],
    ) -> None:
        """Calibrate functions to range [1/n_steps, 1].

        Args:
            worst_score: Minimum utility reference.
            best_score: Maximum utility reference.
        """
        self.worst_score = worst_score
        self.best_score = best_score

        # Calibrate POI component
        self.poi_score_func = scoring.calibrate_scoring_func(
            self.poi_score_func,
            min_score=1 / self.n_steps,
            max_score=1,
            min_point=worst_score[1:],
            max_point=best_score[1:],
        )

        # Calibrate Access component
        min_access_point = (worst_score[0], self.poi_score_func(*worst_score[1:]))
        max_access_point = (best_score[0], self.poi_score_func(*best_score[1:]))

        self.access_score_func = scoring.calibrate_scoring_func(
            self.access_score_func,
            min_score=1 / self.n_steps,
            max_score=1,
            min_point=min_access_point,
            max_point=max_access_point,
        )

    def create_grid(self, variables: Any) -> None:
        """Build adaptive grids for variables.

        Args:
            variables: Variable bounds or lists.
        """
        self.variable_bounds = variables

        self.distance_grid, self.poi_params_grid = scoring.build_adaptive_grids(
            self.function,
            variables=variables,
            delta=1 / self.n_steps,
        )

        self.poi_grid = [
            self.poi_score_func(*args)
            for args in zip(*self.poi_params_grid)
        ]

        self.__create_scoring_matrix()

    def score_pois(self, pois: Any, snap: bool = True) -> Any:
        """Apply POI scoring to a dataset.

        Args:
            pois: Object with a .gdf attribute or a pandas DataFrame.
            snap: Unused, reserved for future logic.

        Returns:
            The dataset (or a copy) with an added 'score' column.
        """
        pois_copy = copy.copy(pois)
        gdf = pois_copy.gdf if hasattr(pois_copy, "gdf") else pois_copy

        gdf["score"] = gdf.apply(
            lambda row: self.poi_score_func(
                **{k: row[k] for k in self.poi_param_names}
            ),
            axis=1,
        )

        if hasattr(pois_copy, "gdf"):
            pois_copy.gdf = gdf
            return pois_copy
        
        return gdf

    def __create_scoring_matrix(self) -> None:
        """Build internal pre-computed scoring matrix."""
        if self.poi_grid is None or self.distance_grid is None:
            return

        df = pd.DataFrame({"poi_score": self.poi_grid})
        for d in self.distance_grid:
            df[d] = [self.access_score_func(d, poi) for poi in self.poi_grid]

        self.scoring_matrix = df.round(3)

    # --- Persistence ---

    def save(
        self, 
        path: Union[str, Path], 
        overwrite: bool = True
    ) -> Path:
        """Save the model to a directory.

        Args:
            path: Target directory.
            overwrite: If True, replaces existing directory. Defaults to True.

        Returns:
            Path: The directory where the model was saved.
        """
        directory = _ensure_dir(path, overwrite)

        # Save functions using dill (handles lambdas)
        with open(directory / "functions.pkl", "wb") as f:
            dill.dump({
                "poi_score_func": self.poi_score_func,
                "access_score_func": self.access_score_func,
                "function": self.function,
            }, f)

        # Write manifest.json (same style as AOI/StreetNetwork)
        _write_manifest(directory, "AccessScore", {
            "n_steps": self.n_steps,
            "poi_param_names": self.poi_param_names,
            "worst_score": _to_serializable(self.worst_score),
            "best_score": _to_serializable(self.best_score),
            "variable_bounds": _to_serializable(self.variable_bounds),
            "functions_file": "functions.pkl"
        })
        
        return directory

    @classmethod
    def load(cls, path: Union[str, Path]) -> AccessScore:
        """Load an AccessScore model from a directory.

        Args:
            path: Directory containing manifest.json and functions.pkl.

        Returns:
            AccessScore: The reconstructed and calibrated model.
        """
        directory = Path(path)
        manifest = _read_manifest(directory)
        _check_class(manifest, "AccessScore")

        # Load serialized functions
        with open(directory / manifest["functions_file"], "rb") as f:
            funcs = dill.load(f)

        # Reconstruct instance
        return cls(
            poi_score_func=funcs["poi_score_func"],
            access_score_func=funcs["access_score_func"],
            worst_score=manifest.get("worst_score"),
            best_score=manifest.get("best_score"),
            variable_bounds=manifest.get("variable_bounds"),
            n_steps=manifest.get("n_steps", 10),
            poi_param_names=manifest.get("poi_param_names"),
        )

    def __repr__(self) -> str:
        return (
            f"AccessScore(params={self.poi_param_names}, "
            f"n_steps={self.n_steps})"
        )
    


class PointsOfInterest:
    """Handler for loading, scoring, and processing Points of Interest (POIs).

    This class provides a unified interface for managing POIs, allowing for 
    spatial filtering, attribute-based scoring, and persistence. By default, 
    all POIs are initialized with a 'score' column of 0.0.

    Attributes:
        data (gpd.GeoDataFrame): The underlying geospatial data.
        score (pd.Series): The 'score' column from the data.
    """

    def __init__(
        self,
        data: Optional[Union[gpd.GeoDataFrame, str, Path]] = None,
        aoi: Optional[Union[Any, gpd.GeoDataFrame]] = None,
    ):
        """Initializes PointsOfInterest.

        Args:
            data: A GeoDataFrame or a path to a geospatial file.
            aoi: An Area of Interest to filter the POIs upon creation.
        """
        self._data: Optional[gpd.GeoDataFrame] = (
            _ensure_gdf(data) if data is not None else None
        )

        if self._data is not None:
            # 1. Spatial Filtering
            if aoi is not None:
                aoi_gdf = _bounds_from(aoi)
                # Ensure CRS match for intersection
                self._data = self._data.to_crs(aoi_gdf.crs)
                mask = self._data.intersects(aoi_gdf.union_all())
                self._data = self._data.loc[mask].copy()

            # 2. Default Score Initialization
            if "score" not in self._data.columns:
                self._data["score"] = 0.0
            else:
                self._data["score"] = self._data["score"].fillna(0.0).astype(float)

    # --- Properties ---

    @property
    def data(self) -> gpd.GeoDataFrame:
        """The underlying GeoDataFrame."""
        if self._data is None:
            raise ValueError("No data loaded. Initialize with data or use a from_* method.")
        return self._data

    @property
    def crs(self) -> gpd.GeoDataFrame:
        """The underlying GeoDataFrame."""
        if self._data is None:
            raise ValueError("No data loaded. Initialize with data or use a from_* method.")
        return self._data.crs

    @property
    def geometry(self) -> gpd.GeoDataFrame:
        """The underlying GeoDataFrame."""
        if self._data is None:
            raise ValueError("No data loaded. Initialize with data or use a from_* method.")
        return self._data.geometry

    @property
    def score(self) -> pd.Series:
        """The score column of the POI data."""
        return self.data["score"]

    # --- OSM Constructors ---

    @classmethod
    def from_overpass(
        cls, 
        query: str, 
        aoi: Any, 
        **kwargs: Any
    ) -> PointsOfInterest:
        """Creates POIs by querying OpenStreetMap via Overpass.

        Args:
            query: Tag name (e.g., 'schools', 'healthcare') or a raw Overpass query string.
            aoi: Area of Interest to query within.
            **kwargs: Additional arguments passed to specific osm download functions.

        Returns:
            PointsOfInterest: A new instance with the downloaded data.
        """
        bounds = _bounds_from(aoi)
        q = query.lower().replace("_", " ")

        if q == "green areas":
            data = osm.green_areas(bounds, **kwargs)
        elif q == "bus stops":
            data = osm.bus_stops(bounds, **kwargs)
        elif q == "schools":
            data = osm.schools(bounds, **kwargs)
        elif q == "healthcare":
            data = osm.healthcare(bounds, **kwargs)
        elif q == "groceries":
            data = osm.groceries(bounds, **kwargs)
        elif q == "restaurants":
            data = osm.restaurants(bounds, **kwargs)
        else:
            data = osm.overpass_api_query(query, bounds)

        return cls(data=data, aoi=aoi)

    # --- Scoring Methods ---

    def assign_score(
        self,
        function: Union[Callable, Any],
        param_names: Optional[List[str]] = None,
    ) -> PointsOfInterest:
        """Assigns a quality score using a custom function or AccessScore object.

        Args:
            function: A callable that returns a float, or an AccessScore instance.
            param_names: List of column names to pass to the function. If None,
                it is inferred from the function signature.

        Returns:
            PointsOfInterest: Self with updated 'score' column.
        """
        # Handle AccessScore instance input
        if hasattr(function, "poi_score_func"):
            # If function is an AccessScore object
            param_names = getattr(function, "poi_param_names", param_names)
            function = getattr(function, "poi_score_func")
        
        if param_names is None:
            sig = inspect.signature(function)
            param_names = list(sig.parameters.keys())

        # Ensure columns exist before applying
        missing = [p for p in param_names if p not in self.data.columns]
        if missing:
            raise KeyError(f"Columns missing from POI data: {missing}")

        self.data["score"] = self.data.apply(
            lambda row: function(**{k: row[k] for k in param_names}),
            axis=1
        )
        return self

    def assign_score_by_area(
        self,
        area_steps: List[float],
        large_is_better: bool = True,
    ) -> PointsOfInterest:
        """Assigns a quality score [0, 1] based on geometry area.

        Args:
            area_steps: Threshold values for area scoring.
            large_is_better: Whether larger areas receive higher scores.

        Returns:
            PointsOfInterest: Self with updated 'score' column.
        """
        self.data["score"] = poi_utils.score_by_area(
            self.data, area_steps, large_is_better=large_is_better
        )
        return self

    def assign_score_by_column_values(
        self,
        columns: Union[str, List[str]],
        value_priority: List[Any],
    ) -> PointsOfInterest:
        """Assigns a quality score based on categorical priority.

        Args:
            columns: Column(s) to check.
            value_priority: Ordered list of categories for scoring.

        Returns:
            PointsOfInterest: Self with updated 'score' column.
        """
        self.data["score"] = poi_utils.score_by_values(
            self.data[columns], value_priority
        )
        return self

    # --- Spatial Methods ---

    def clip(self, aoi: Any) -> PointsOfInterest:
        """Returns a new PointsOfInterest instance clipped to an AOI.

        Args:
            aoi: Area of Interest (GeoDataFrame or object with .data).

        Returns:
            PointsOfInterest: New clipped instance.
        """
        return PointsOfInterest(data=self.data.copy(), aoi=aoi)

    def add_to_street_network(self, street_network: Any):
        """Stub for future integration logic with street networks."""
        pass

    def plot(
        self,
        aoi: Optional[Any] = None,
        column: Optional[str] = None,
        cmap: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Generates an interactive Folium map.

        Args:
            aoi: Optional AOI to show on the map.
            column: Data column to color by (defaults to 'score').
            cmap: Colormap name.
            **kwargs: Passed to plot_helpers.general_map.
        """
        col = column or "score"
        return plot_helpers.general_map(
            aoi=_bounds_from(aoi) if aoi is not None else None,
            pois=[self.data],
            poi_column=col,
            poi_cmap=cmap or "viridis",
            **kwargs,
        )

    # --- Persistence ---

    def save(
        self,
        path: Union[str, Path],
        fmt: GeoFmt = "geoparquet",
        overwrite: bool = True,
    ) -> Path:
        """Saves the POIs to a directory.

        Args:
            path: Target directory path.
            fmt: Spatial format (e.g., 'geoparquet', 'gpkg').
            overwrite: Whether to replace the existing directory.

        Returns:
            Path: The directory where the data was saved.
        """
        directory = _ensure_dir(path, overwrite)
        filename = _save_gdf(self.data, directory, "pois", fmt)
        _write_manifest(directory, "PointsOfInterest", {
            "fmt": fmt,
            "pois_file": filename,
        })
        return directory

    @classmethod
    def load(cls, path: Union[str, Path]) -> PointsOfInterest:
        """Loads a PointsOfInterest instance from a saved directory.

        Args:
            path: Path to the directory created by save().

        Returns:
            PointsOfInterest: Reconstructed instance.
        """
        directory = Path(path)
        manifest = _read_manifest(directory)
        _check_class(manifest, "PointsOfInterest")

        obj = cls.__new__(cls)
        obj._data = _load_gdf(directory, manifest["pois_file"])
        return obj

    def __len__(self) -> int:
        return len(self._data) if self._data is not None else 0

    def __repr__(self) -> str:
        count = len(self)
        return f"PointsOfInterest(n={count}, columns={list(self.data.columns)})"
    