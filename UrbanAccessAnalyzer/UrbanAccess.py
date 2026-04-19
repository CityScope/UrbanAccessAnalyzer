"""
urban_access.py
===============
High-level classes for urban accessibility analysis.

Classes
-------
- AreaOfInterest        → Area of interest with geocoding and boundary download
- StreetNetwork         → Street graph (download, simplification, isochrones)
- PointsOfInterest      → POIs from OSM or an external GeoDataFrame
- H3Grid                → H3 hexagonal grid (rasterisation and aggregation)
- PopulationLayer       → WorldPop or custom population data
- AccessibilityAnalyzer → Orchestrator: network + POIs → accessibility scores

Common interface
----------------
Every class exposes:
    .data          → GeoDataFrame / DataFrame / graph with the main result
    .plot()        → Interactive Folium map
    .to_h3()       → Converts .data to an H3Grid
    .save(path)    → Persists the object to disk
    .load(path)    → (classmethod) Restores the object from disk

════════════════════════════════════════════════════════════════════════════════
Persistence strategy
════════════════════════════════════════════════════════════════════════════════

Each instance is saved as a **directory** containing:

    <path>/
        manifest.json         → class name, version, format, scalar metadata
        <class-specific data> → see table below

Class                Files written
──────────────────   ────────────────────────────────────────────────────────
AreaOfInterest       aoi.<ext>
StreetNetwork        nodes.<ext>  edges.<ext>  graph_attrs.json
PointsOfInterest     pois.<ext>
H3Grid               grid.parquet              (always Parquet; no geometry)
PopulationLayer      population.<ext>  (GeoDataFrame) | path reference (raster)
AccessibilityAnalyzer
                     network/  pois/  [result_network/]  manifest.json

Supported geospatial formats (``fmt`` parameter)
    "geoparquet"  → GeoParquet + zstd + geoarrow  [DEFAULT — optimal for backends]
    "parquet"     → alias for "geoparquet"
    "fgb"         → FlatGeobuf                     [streaming / map tiles]
    "gpkg"        → GeoPackage                     [QGIS / GDAL compatible]
    "geojson"     → GeoJSON                        [web interoperability]
    "shp"         → Shapefile                      [legacy GIS]

H3Grid always uses flat Parquet (no geometry stored on disk). Hexagon
geometries are reconstructed on demand from the h3_cell index, keeping the
file minimal and directly queryable with DuckDB / Polars / pandas without
any geospatial dependencies.

StreetNetwork stores nodes and edges as separate files:
    - Nodes in GeoParquet/Parquet: no geometry column (x, y are sufficient),
      saving ~30 % of space. Other formats include a Point geometry for GIS.
    - Edges: always stored with LineString geometry.
    - graph_attrs.json: graph-level attributes (crs, name, …).
    - The NetworkX graph is reconstructed via ox.graph_from_gdfs() on load.
"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd

# ── osmnx (optional at import time, required when used) ───────────────────────
try:
    import osmnx as ox
except ImportError:
    ox = None  # type: ignore

# ── Internal package modules (always relative) ────────────────────────────────
from . import (
    geometry_utils,
    graph_processing,
    h3_utils,
    isochrones,
    osm,
    plot_helpers,
    poi_utils,
    population as pop_module,
    quality,
    utils,
)


# ──────────────────────────────────────────────────────────────────────────────
# Types and constants
# ──────────────────────────────────────────────────────────────────────────────

GeoFmt = Literal["geoparquet", "parquet", "fgb", "gpkg", "geojson", "shp"]

_FORMAT_VERSION = "1.0"

_FMT_EXT: Dict[str, str] = {
    "geoparquet": ".geoparquet",
    "parquet":    ".geoparquet",  # alias
    "fgb":        ".fgb",
    "gpkg":       ".gpkg",
    "geojson":    ".geojson",
    "shp":        ".shp",
}


# ──────────────────────────────────────────────────────────────────────────────
# Low-level I/O — GeoDataFrame
# ──────────────────────────────────────────────────────────────────────────────

def _save_gdf(gdf: gpd.GeoDataFrame, directory: Path, stem: str, fmt: str) -> str:
    """
    Save a GeoDataFrame in the requested format.

    A named index is reset to a column so it survives the round-trip intact.
    Returns the filename relative to *directory* (stored in the manifest).
    """
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
    """Load a GeoDataFrame, auto-detecting the format from the file extension."""
    p = directory / filename
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return gpd.read_parquet(p) if p.suffix.lower() in (".geoparquet", ".parquet") else gpd.read_file(p)


# ──────────────────────────────────────────────────────────────────────────────
# Low-level I/O — flat DataFrame (H3Grid)
# ──────────────────────────────────────────────────────────────────────────────

def _save_df(df: pd.DataFrame, directory: Path, stem: str) -> str:
    """Save a geometry-free DataFrame as compressed Parquet."""
    filename = f"{stem}.parquet"
    df.to_parquet(directory / filename, compression="zstd")
    return filename


def _load_df(directory: Path, stem: str) -> pd.DataFrame:
    return pd.read_parquet(directory / f"{stem}.parquet")


# ──────────────────────────────────────────────────────────────────────────────
# Low-level I/O — OSMnx graph
# ──────────────────────────────────────────────────────────────────────────────

def _sanitize_col(series: pd.Series) -> pd.Series:
    """Serialize list/dict values to JSON strings for columnar formats."""
    if series.dtype != object:
        return series
    return series.apply(
        lambda x: json.dumps(x, default=str) if isinstance(x, (list, dict)) else x
    )


def _deserialize_col(series: pd.Series) -> pd.Series:
    """Restore list/dict values from JSON strings where applicable."""
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


def _save_graph(G, directory: Path, fmt: str) -> Dict[str, str]:
    """
    Persist an OSMnx graph as a nodes + edges pair of files.

    Design decisions
    ────────────────
    GeoParquet / Parquet
      • Nodes: scalar columns only + x, y coordinates. The redundant Point
        geometry column is dropped, saving ~30 % of space. It is reconstructed
        from x, y on load.
      • Edges: LineString geometry + columns flattened from the MultiIndex.
        Complex OSM attribute types (lists, dicts) are serialised to JSON strings.

    Other formats (fgb, gpkg, geojson, shp)
      • Nodes: Point geometry included for GIS tool compatibility.
      • Edges: LineString geometry.

    graph_attrs.json stores graph-level attributes (crs, name, …) and is
    always written as JSON regardless of the chosen format.

    Returns a dict of {key: filename} references for the manifest.
    """
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    # Serialise complex columns
    for col in nodes_gdf.select_dtypes("object").columns:
        nodes_gdf[col] = _sanitize_col(nodes_gdf[col])
    for col in edges_gdf.select_dtypes("object").columns:
        edges_gdf[col] = _sanitize_col(edges_gdf[col])

    fmt_norm = fmt.lower()

    # ── Nodes ─────────────────────────────────────────────────────────────────
    nodes_flat = nodes_gdf.reset_index()  # osmid → plain column

    if fmt_norm in ("geoparquet", "parquet"):
        nodes_df = pd.DataFrame(nodes_flat.drop(columns=["geometry"]))
        nodes_file = "nodes.parquet"
        nodes_df.to_parquet(directory / nodes_file, compression="zstd")
    else:
        nodes_file = _save_gdf(nodes_flat, directory, "nodes", fmt)

    # ── Edges ─────────────────────────────────────────────────────────────────
    edges_flat = edges_gdf.reset_index()  # u, v, key → plain columns
    edges_file = _save_gdf(edges_flat, directory, "edges", fmt)

    # ── Graph attributes ──────────────────────────────────────────────────────
    graph_attrs = {k: str(v) for k, v in G.graph.items()}
    (directory / "graph_attrs.json").write_text(
        json.dumps(graph_attrs, indent=2), encoding="utf-8"
    )

    return {
        "nodes_file": nodes_file,
        "edges_file": edges_file,
        "graph_attrs_file": "graph_attrs.json",
    }


def _load_graph(directory: Path, nodes_file: str, edges_file: str):
    """
    Reconstruct an OSMnx graph from the stored nodes and edges files.

    Tolerant: recalculates 'length' if the column is missing from edges.
    """
    graph_attrs = json.loads(
        (directory / "graph_attrs.json").read_text(encoding="utf-8")
    )

    # ── Nodes ─────────────────────────────────────────────────────────────────
    nodes_path = directory / nodes_file
    is_flat_parquet = (
        nodes_path.suffix.lower() == ".parquet"
        and not nodes_file.endswith(".geoparquet")
    )

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

    # ── Edges ─────────────────────────────────────────────────────────────────
    edges_gdf = _load_gdf(directory, edges_file)
    for col in edges_gdf.select_dtypes("object").columns:
        edges_gdf[col] = _deserialize_col(edges_gdf[col])

    for c in ("u", "v", "key"):
        if c in edges_gdf.columns:
            edges_gdf[c] = edges_gdf[c].astype(int)
    edges_gdf = edges_gdf.set_index(["u", "v", "key"])

    if "length" not in edges_gdf.columns:
        warnings.warn("Column 'length' missing from edges; recalculating.", UserWarning)
        edges_gdf["length"] = edges_gdf.geometry.length

    return ox.graph_from_gdfs(nodes_gdf, edges_gdf, graph_attrs=graph_attrs)


# ──────────────────────────────────────────────────────────────────────────────
# Manifest and directory helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_manifest(directory: Path, class_name: str, meta: dict) -> None:
    manifest = {"class": class_name, "version": _FORMAT_VERSION, **meta}
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


def _read_manifest(directory: Path) -> dict:
    p = directory / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"manifest.json not found in '{directory}'")
    return json.loads(p.read_text(encoding="utf-8"))


def _check_class(manifest: dict, expected: str) -> None:
    found = manifest.get("class")
    if found != expected:
        raise TypeError(
            f"Tried to load a '{expected}' but the directory contains "
            f"a '{found}'. Use the correct class for load()."
        )


def _ensure_dir(path: Union[str, Path], overwrite: bool) -> Path:
    p = Path(path)
    if p.exists():
        if not overwrite:
            raise FileExistsError(
                f"Directory '{p}' already exists. Pass overwrite=True to replace it."
            )
        shutil.rmtree(p)
    p.mkdir(parents=True)
    return p


def _ensure_gdf(data, crs=None) -> gpd.GeoDataFrame:
    if isinstance(data, gpd.GeoDataFrame):
        gdf = data
    elif isinstance(data, gpd.GeoSeries):
        gdf = gpd.GeoDataFrame(geometry=data, crs=data.crs)
    elif isinstance(data, str):
        gdf = geometry_utils.read_geofile(data)
    else:
        raise TypeError(f"Unsupported input type: {type(data)}")
    return gdf if crs is None else gdf.to_crs(crs)


def _bounds_from(aoi) -> gpd.GeoDataFrame:
    """Extract the underlying GeoDataFrame from an AOI-like object."""
    return aoi.data if isinstance(aoi, AreaOfInterest) else aoi


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
        crs: int = 4326,
        buffer: float = 0.0,
    ):
        if isinstance(data, str) and not Path(data).exists():
            gdf = utils.get_city_geometry(data)
        else:
            gdf = _ensure_gdf(data)

        gdf = gdf.to_crs(4326)

        if buffer > 0:
            gdf = gdf.to_crs(gdf.estimate_utm_crs())
            gdf = gdf.assign(geometry=gdf.geometry.buffer(buffer))
            gdf = gdf.to_crs(4326)

        if crs != 4326:
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
        return gdf.to_crs(self.data.crs).loc[
            lambda g: g.intersects(self.union)
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


# ──────────────────────────────────────────────────────────────────────────────
# 2. StreetNetwork
# ──────────────────────────────────────────────────────────────────────────────

class StreetNetwork:
    """
    OSMnx-based street network graph.

    Parameters
    ----------
    aoi : AreaOfInterest | GeoDataFrame | GeoSeries
        Area of interest used for the download.
    network_type : str
        OSMnx network type: "walk" | "bike" | "drive" | "all".
    custom_filter : str, optional
        Custom OSM filter string (osmnx syntax).
    min_edge_length : float, optional
        Minimum edge length in metres for simplification. 0 = no simplification.
    G : networkx.MultiDiGraph, optional
        Pre-built graph; skips the download when provided.

    Persistence
    -----------
    >>> net.save("data/berlin_walk")                # GeoParquet (default)
    >>> net.save("data/berlin_walk", fmt="fgb")
    >>> net.save("data/berlin_walk", fmt="gpkg")    # opens in QGIS
    >>> net = StreetNetwork.load("data/berlin_walk")

    Directory layout::

        nodes.parquet       (GeoParquet) or nodes.<ext>
        edges.<ext>
        graph_attrs.json
        manifest.json
    """

    def __init__(
        self,
        aoi: Union[AreaOfInterest, gpd.GeoDataFrame, gpd.GeoSeries],
        network_type: str = "walk",
        custom_filter: Optional[str] = None,
        min_edge_length: float = 0.0,
        G=None,
    ):
        bounds = _bounds_from(aoi) if not isinstance(aoi, gpd.GeoDataFrame) else aoi
        if isinstance(aoi, AreaOfInterest):
            bounds = aoi.data

        if G is not None:
            self._G = G
        else:
            self._G = osm.download_street_graph(
                bounds, network_type=network_type, custom_filter=custom_filter
            )
            if min_edge_length > 0:
                self._G = graph_processing.simplify_graph(
                    self._G,
                    min_edge_length=min_edge_length,
                    min_edge_separation=min_edge_length * 2,
                    undirected=True,
                )

        self.network_type = network_type
        self._custom_filter = custom_filter
        self._nodes: Optional[gpd.GeoDataFrame] = None
        self._edges: Optional[gpd.GeoDataFrame] = None

    # ── Lazy properties ───────────────────────────────────────────────────────

    @property
    def graph(self):
        """Underlying NetworkX MultiDiGraph."""
        return self._G

    @property
    def nodes(self) -> gpd.GeoDataFrame:
        """Node GeoDataFrame (computed and cached on first access)."""
        if self._nodes is None:
            self._nodes, self._edges = ox.graph_to_gdfs(self._G)
        return self._nodes

    @property
    def edges(self) -> gpd.GeoDataFrame:
        """Edge GeoDataFrame (computed and cached on first access)."""
        if self._edges is None:
            self._nodes, self._edges = ox.graph_to_gdfs(self._G)
        return self._edges

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Alias for ``edges`` — uniform interface with other classes."""
        return self.edges

    def _invalidate_cache(self) -> None:
        self._nodes = None
        self._edges = None

    # ── Methods ───────────────────────────────────────────────────────────────

    def simplify(
        self,
        min_edge_length: float = 0,
        min_edge_separation: float = 0,
        loops: bool = True,
        multi: bool = True,
        undirected: bool = False,
    ) -> "StreetNetwork":
        """Return a new simplified StreetNetwork."""
        new_G = graph_processing.simplify_graph(
            self._G,
            min_edge_length=min_edge_length,
            min_edge_separation=min_edge_separation,
            loops=loops,
            multi=multi,
            undirected=undirected,
        )
        return StreetNetwork(aoi=self.nodes, G=new_G, network_type=self.network_type)

    def crop(self, aoi: Union["AreaOfInterest", gpd.GeoDataFrame]) -> "StreetNetwork":
        """Return a new StreetNetwork clipped to *aoi*."""
        aoi_gdf = _bounds_from(aoi)
        return StreetNetwork(
            aoi=aoi_gdf,
            G=graph_processing.crop_by_aoi(aoi_gdf, G=self._G),
            network_type=self.network_type,
        )

    def nearest_nodes(
        self,
        geometries: Union[gpd.GeoDataFrame, gpd.GeoSeries],
        max_dist: Optional[float] = None,
    ) -> list:
        """Return the node IDs nearest to each geometry."""
        return graph_processing.nearest_nodes(geometries, self._G, max_dist=max_dist)

    def add_points(
        self,
        points: Union[gpd.GeoDataFrame, gpd.GeoSeries],
        max_dist: Optional[float] = None,
        min_edge_length: float = 0,
    ) -> tuple["StreetNetwork", list]:
        """
        Project points onto the graph and return ``(StreetNetwork, osmids)``.

        The returned osmids can be used directly as origin nodes for isochrones.
        """
        new_G, osmids = graph_processing.add_points_to_graph(
            points, self._G, max_dist=max_dist, min_edge_length=min_edge_length
        )
        return StreetNetwork(aoi=self.nodes, G=new_G, network_type=self.network_type), osmids

    def plot(
        self,
        aoi: Optional[Union["AreaOfInterest", gpd.GeoDataFrame]] = None,
        column: Optional[str] = None,
        cmap: Optional[str] = None,
        **kwargs,
    ):
        """Interactive Folium map of the network."""
        return plot_helpers.general_map(
            aoi=_bounds_from(aoi) if aoi is not None else None,
            gdfs=[self.edges],
            column=column,
            cmap=cmap,
            **kwargs,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        path: Union[str, Path],
        fmt: GeoFmt = "geoparquet",
        overwrite: bool = True,
    ) -> Path:
        """
        Save the street network to disk.

        In GeoParquet / Parquet format, nodes are stored without a geometry
        column (only x, y), saving ~30 % of space. Other formats include the
        Point geometry for GIS compatibility. Edges always include LineString
        geometry.

        Parameters
        ----------
        path : str | Path
            Output directory.
        fmt : GeoFmt
            "geoparquet" | "fgb" | "gpkg" | "geojson" | "shp".
        overwrite : bool
            Replace an existing directory if True.

        Examples
        --------
        >>> net.save("data/berlin_walk")
        >>> net.save("data/berlin_walk", fmt="gpkg")
        """
        directory = _ensure_dir(path, overwrite)
        file_refs = _save_graph(self._G, directory, fmt)
        _write_manifest(directory, "StreetNetwork", {
            "fmt": fmt,
            "network_type": self.network_type,
            "custom_filter": self._custom_filter,
            **file_refs,
        })
        return directory

    @classmethod
    def load(cls, path: Union[str, Path]) -> "StreetNetwork":
        """
        Load a StreetNetwork from a directory created by ``save()``.

        Parameters
        ----------
        path : str | Path
            Persistence directory.

        Examples
        --------
        >>> net = StreetNetwork.load("data/berlin_walk")
        """
        directory = Path(path)
        manifest = _read_manifest(directory)
        _check_class(manifest, "StreetNetwork")

        obj = cls.__new__(cls)
        obj._G = _load_graph(
            directory,
            nodes_file=manifest["nodes_file"],
            edges_file=manifest["edges_file"],
        )
        obj.network_type = manifest.get("network_type", "walk")
        obj._custom_filter = manifest.get("custom_filter")
        obj._nodes = None
        obj._edges = None
        return obj

    def __repr__(self) -> str:
        return (
            f"StreetNetwork(nodes={self._G.number_of_nodes()}, "
            f"edges={self._G.number_of_edges()}, "
            f"type='{self.network_type}')"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. PointsOfInterest
# ──────────────────────────────────────────────────────────────────────────────

class PointsOfInterest:
    """
    Load and pre-process points of interest (POIs).

    Parameters
    ----------
    data : GeoDataFrame | str | None
        POI GeoDataFrame, path to a geospatial file, or None (use a
        ``from_osm_*`` class method).
    quality_column : str, optional
        Column holding a quality score in [0, 1].
    aoi : AreaOfInterest | GeoDataFrame, optional
        Filters the POIs to the AOI on construction.

    Persistence
    -----------
    >>> pois.save("data/parks")
    >>> pois.save("data/parks", fmt="geojson")
    >>> pois = PointsOfInterest.load("data/parks")
    """

    def __init__(
        self,
        data: Optional[Union[gpd.GeoDataFrame, str]] = None,
        quality_column: Optional[str] = None,
        aoi: Optional[Union[AreaOfInterest, gpd.GeoDataFrame]] = None,
    ):
        self._data: Optional[gpd.GeoDataFrame] = (
            _ensure_gdf(data) if data is not None else None
        )
        self.quality_column = quality_column

        if aoi is not None and self._data is not None:
            aoi_gdf = _bounds_from(aoi)
            self._data = self._data.loc[
                self._data.to_crs(aoi_gdf.crs).intersects(aoi_gdf.union_all())
            ]

    # ── OSM constructors ──────────────────────────────────────────────────────

    @classmethod
    def _from_osm(cls, fn: Callable, aoi, **kwargs) -> "PointsOfInterest":
        return cls(data=fn(_bounds_from(aoi), **kwargs), aoi=aoi)

    @classmethod
    def from_osm_parks(cls, aoi, **kwargs) -> "PointsOfInterest":
        """Download parks and green areas from OSM."""
        return cls._from_osm(osm.green_areas, aoi, **kwargs)

    @classmethod
    def from_osm_bus_stops(cls, aoi) -> "PointsOfInterest":
        """Download bus stops from OSM."""
        return cls._from_osm(osm.bus_stops, aoi)

    @classmethod
    def from_osm_schools(cls, aoi) -> "PointsOfInterest":
        """Download schools from OSM."""
        return cls._from_osm(osm.schools, aoi)

    @classmethod
    def from_osm_healthcare(cls, aoi) -> "PointsOfInterest":
        """Download healthcare facilities from OSM."""
        return cls._from_osm(osm.healthcare, aoi)

    @classmethod
    def from_osm_groceries(cls, aoi) -> "PointsOfInterest":
        """Download supermarkets and grocery stores from OSM."""
        return cls._from_osm(osm.groceries, aoi)

    @classmethod
    def from_osm_restaurants(cls, aoi) -> "PointsOfInterest":
        """Download restaurants and bars from OSM."""
        return cls._from_osm(osm.restaurants, aoi)

    @classmethod
    def from_overpass_query(cls, aoi, query: str) -> "PointsOfInterest":
        """Execute a custom Overpass QL query."""
        return cls(data=osm.overpass_api_query(query, _bounds_from(aoi)), aoi=aoi)

    # ── Main property ─────────────────────────────────────────────────────────

    @property
    def data(self) -> gpd.GeoDataFrame:
        if self._data is None:
            raise ValueError("No POIs loaded. Call a from_osm_* method first.")
        return self._data

    # ── Pre-processing ────────────────────────────────────────────────────────

    def assign_quality(
        self,
        quality_func: Callable,
        column_name: str = "poi_quality",
    ) -> "PointsOfInterest":
        """
        Assign a quality score using a custom function ``f(gdf) → Series``.

        The function receives the POI GeoDataFrame and must return a
        Series / array of float values in [0, 1].
        """
        self._data = self._data.copy()
        self._data[column_name] = quality_func(self._data)
        self.quality_column = column_name
        return self

    def assign_quality_by_area(
        self,
        area_steps: List[float],
        large_is_better: bool = True,
        column_name: str = "poi_quality",
    ) -> "PointsOfInterest":
        """Assign a quality score in [0, 1] based on the geometry area."""
        self._data = self._data.copy()
        self._data[column_name] = poi_utils.quality_by_area(
            self._data, area_steps, large_is_better=large_is_better
        )
        self.quality_column = column_name
        return self

    def assign_quality_by_values(
        self,
        column: str,
        value_priority: List,
        quality_column_name: str = "poi_quality",
    ) -> "PointsOfInterest":
        """Assign a quality score in [0, 1] based on a categorical priority list."""
        self._data = self._data.copy()
        self._data[quality_column_name] = poi_utils.quality_by_values(
            self._data[column], value_priority
        )
        self.quality_column = quality_column_name
        return self

    def to_points(self, street_edges: gpd.GeoDataFrame) -> "PointsOfInterest":
        """Convert polygon POIs to points projected onto the street network."""
        return PointsOfInterest(
            data=poi_utils.polygons_to_points(self._data, street_edges),
            quality_column=self.quality_column,
        )

    def clip(self, aoi: Union[AreaOfInterest, gpd.GeoDataFrame]) -> "PointsOfInterest":
        """Return a new PointsOfInterest clipped to *aoi*."""
        aoi_gdf = _bounds_from(aoi)
        return PointsOfInterest(
            data=self._data.loc[
                self._data.to_crs(aoi_gdf.crs).intersects(aoi_gdf.union_all())
            ],
            quality_column=self.quality_column,
        )

    def to_h3(
        self,
        resolution: int,
        columns: Optional[List[str]] = None,
        method: Union[str, Dict[str, str]] = "max",
        buffer: float = 0.0,
    ) -> "H3Grid":
        """Rasterise the POIs into an H3 grid."""
        cols = columns or ([self.quality_column] if self.quality_column else [])
        df = h3_utils.from_gdf(
            self._data, resolution=resolution, columns=cols,
            method=method, buffer=buffer,
        )
        return H3Grid(data=df, resolution=resolution)

    def plot(
        self,
        aoi: Optional[Union[AreaOfInterest, gpd.GeoDataFrame]] = None,
        column: Optional[str] = None,
        cmap: Optional[str] = None,
        **kwargs,
    ):
        """Interactive Folium map of the POIs."""
        col = column or self.quality_column
        return plot_helpers.general_map(
            aoi=_bounds_from(aoi) if aoi is not None else None,
            pois=[self._data],
            poi_column=col,
            poi_cmap=cmap or ("viridis" if col else None),
            **kwargs,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        path: Union[str, Path],
        fmt: GeoFmt = "geoparquet",
        overwrite: bool = True,
    ) -> Path:
        """
        Save the POIs to disk.

        Parameters
        ----------
        path : str | Path
            Output directory.
        fmt : GeoFmt
            "geoparquet" | "fgb" | "gpkg" | "geojson" | "shp".
        overwrite : bool
            Replace an existing directory if True.

        Examples
        --------
        >>> pois.save("data/berlin_parks")
        >>> pois.save("data/berlin_parks", fmt="geojson")
        """
        directory = _ensure_dir(path, overwrite)
        filename = _save_gdf(self.data, directory, "pois", fmt)
        _write_manifest(directory, "PointsOfInterest", {
            "fmt": fmt,
            "quality_column": self.quality_column,
            "pois_file": filename,
        })
        return directory

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PointsOfInterest":
        """
        Load a PointsOfInterest from a directory created by ``save()``.

        Examples
        --------
        >>> pois = PointsOfInterest.load("data/berlin_parks")
        """
        directory = Path(path)
        manifest = _read_manifest(directory)
        _check_class(manifest, "PointsOfInterest")

        obj = cls.__new__(cls)
        obj._data = _load_gdf(directory, manifest["pois_file"])
        obj.quality_column = manifest.get("quality_column")
        return obj

    def __len__(self) -> int:
        return len(self._data) if self._data is not None else 0

    def __repr__(self) -> str:
        return f"PointsOfInterest(n={len(self)}, quality_column='{self.quality_column}')"


# ──────────────────────────────────────────────────────────────────────────────
# 4. H3Grid
# ──────────────────────────────────────────────────────────────────────────────

class H3Grid:
    """
    H3 hexagonal grid with aggregated data.

    Data is always stored on disk as flat Parquet (no geometry). Hexagon
    geometries are reconstructed on demand from the h3_cell index, keeping
    the file minimal and directly queryable with DuckDB / Polars / pandas
    without any geospatial dependencies.

    Parameters
    ----------
    data : pd.DataFrame | gpd.GeoDataFrame
        DataFrame indexed by ``h3_cell`` with the grid data.
    resolution : int, optional
        H3 resolution of the grid.

    Persistence
    -----------
    H3Grid always uses Parquet. There is no benefit to storing geometry on
    disk because the h3_cell index already encodes each hexagon's shape.

    >>> grid.save("data/berlin_grid")
    >>> grid = H3Grid.load("data/berlin_grid")
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, gpd.GeoDataFrame],
        resolution: Optional[int] = None,
    ):
        self._data = data
        self.resolution = resolution

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_gdf(
        cls,
        gdf: gpd.GeoDataFrame,
        resolution: int,
        columns: Optional[List[str]] = None,
        value_order: Optional[Union[List, Dict[str, List]]] = None,
        buffer: float = 0.0,
        method: Union[str, Dict[str, str]] = "max",
    ) -> "H3Grid":
        """Rasterise a GeoDataFrame into an H3 grid."""
        df = h3_utils.from_gdf(
            gdf, resolution=resolution, columns=columns,
            value_order=value_order, buffer=buffer, method=method,
        )
        return cls(data=df, resolution=resolution)

    @classmethod
    def from_raster(
        cls,
        raster,
        resolution: int,
        aoi=None,
        method: Union[str, Dict[str, str]] = "distribute",
        nodata=None,
    ) -> "H3Grid":
        """Rasterise a raster (path or ndarray) into an H3 grid."""
        df = h3_utils.from_raster(
            raster,
            aoi=aoi.data if isinstance(aoi, AreaOfInterest) else aoi,
            resolution=resolution,
            method=method,
            nodata=nodata,
        )
        return cls(data=df, resolution=resolution)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """GeoDataFrame with hexagon geometries (lazy conversion)."""
        if isinstance(self._data, gpd.GeoDataFrame) and "geometry" in self._data.columns:
            return self._data
        return h3_utils.to_gdf(self._data)

    # ── Methods ───────────────────────────────────────────────────────────────

    def resample(
        self,
        target_resolution: int,
        columns: Optional[List[str]] = None,
        method: Union[str, Dict[str, str]] = "max",
    ) -> "H3Grid":
        """Return a new H3Grid resampled to a coarser resolution."""
        return H3Grid(
            data=h3_utils.resample(self._data, target_resolution, columns=columns, method=method),
            resolution=target_resolution,
        )

    def join(self, other: "H3Grid", how: str = "left") -> "H3Grid":
        """Join two H3Grid objects on their h3_cell index."""
        return H3Grid(
            data=self._data.join(other._data, how=how, rsuffix="_right"),
            resolution=self.resolution,
        )

    def clip(self, aoi: Union[AreaOfInterest, gpd.GeoDataFrame]) -> "H3Grid":
        """Return a new H3Grid clipped to *aoi*."""
        aoi_gdf = _bounds_from(aoi)
        mask = self.gdf.to_crs(aoi_gdf.crs).intersects(aoi_gdf.union_all())
        return H3Grid(data=self._data.loc[mask], resolution=self.resolution)

    def plot(
        self,
        column: Optional[str] = None,
        cmap: str = "viridis",
        aoi: Optional[Union[AreaOfInterest, gpd.GeoDataFrame]] = None,
        **kwargs,
    ):
        """Interactive Folium map of the grid."""
        return plot_helpers.general_map(
            aoi=_bounds_from(aoi) if aoi is not None else None,
            gdfs=[self._data],
            column=column,
            cmap=cmap,
            **kwargs,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path], overwrite: bool = True) -> Path:
        """
        Save the H3 grid as flat Parquet.

        The format is always Parquet (no geometry). This enables direct queries
        with DuckDB / Polars and produces files 3-5× smaller than equivalent
        GeoJSON. Geometry is reconstructed by calling ``.gdf``.

        Parameters
        ----------
        path : str | Path
            Output directory.
        overwrite : bool
            Replace an existing directory if True.

        Examples
        --------
        >>> grid.save("data/berlin_h3_res9")
        >>> grid = H3Grid.load("data/berlin_h3_res9")
        """
        directory = _ensure_dir(path, overwrite)

        df_to_save = self._data
        if isinstance(df_to_save, gpd.GeoDataFrame):
            df_to_save = pd.DataFrame(df_to_save.drop(columns=df_to_save.geometry.name))

        filename = _save_df(df_to_save, directory, "grid")
        _write_manifest(directory, "H3Grid", {
            "resolution": self.resolution,
            "grid_file": filename,
            "columns": list(df_to_save.columns),
        })
        return directory

    @classmethod
    def load(cls, path: Union[str, Path]) -> "H3Grid":
        """
        Load an H3Grid from a directory created by ``save()``.

        Examples
        --------
        >>> grid = H3Grid.load("data/berlin_h3_res9")
        """
        directory = Path(path)
        manifest = _read_manifest(directory)
        _check_class(manifest, "H3Grid")

        obj = cls.__new__(cls)
        obj._data = _load_df(directory, "grid")
        obj.resolution = manifest.get("resolution")
        return obj

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return (
            f"H3Grid(cells={len(self)}, "
            f"resolution={self.resolution}, "
            f"columns={list(self._data.columns)})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 5. PopulationLayer
# ──────────────────────────────────────────────────────────────────────────────

class PopulationLayer:
    """
    Population layer (WorldPop raster or custom data).

    Parameters
    ----------
    data : str | ndarray | GeoDataFrame
        Path to a WorldPop raster, a NumPy array, or a GeoDataFrame with a
        population column.
    population_column : str, optional
        Name of the population column when *data* is a GeoDataFrame.
    aoi : AreaOfInterest | GeoDataFrame, optional
        AOI used for clipping.

    Persistence
    -----------
    - GeoDataFrame → saved in the chosen geodata format.
    - Raster path  → the path is stored in the manifest (set ``copy_raster=True``
      to copy the file into the persistence directory).
    - ndarray      → saved as ``.npy`` (requires ``copy_raster=True``).

    >>> pop.save("data/berlin_pop")
    >>> pop.save("data/berlin_pop", fmt="gpkg", copy_raster=True)
    >>> pop = PopulationLayer.load("data/berlin_pop")
    """

    def __init__(
        self,
        data,
        population_column: Optional[str] = None,
        aoi: Optional[Union[AreaOfInterest, gpd.GeoDataFrame]] = None,
    ):
        self._data = data
        self.population_column = population_column
        self._aoi = _bounds_from(aoi) if aoi is not None else None
        self._transform = None
        self._crs = None

    # ── Alternative constructor ───────────────────────────────────────────────

    @classmethod
    def from_worldpop(
        cls,
        aoi: Union[AreaOfInterest, gpd.GeoDataFrame],
        year: int,
        folder: Optional[str] = None,
        resolution: str = "100m",
    ) -> "PopulationLayer":
        """Download the WorldPop raster for the given AOI and year."""
        from datetime import datetime
        path = pop_module.download_worldpop_population(
            _bounds_from(aoi),
            date=datetime(year=year, month=1, day=1),
            folder=folder,
            resolution=resolution,
        )
        return cls(data=path, aoi=aoi)

    # ── Property ──────────────────────────────────────────────────────────────

    @property
    def data(self):
        return self._data

    # ── Methods ───────────────────────────────────────────────────────────────

    def filter_by_streets(
        self,
        network: StreetNetwork,
        street_buffer: float = 50,
        min_population: float = 0,
        scale: bool = True,
    ) -> "PopulationLayer":
        """Filter population to areas adjacent to the street network."""
        result = pop_module.filter_population_by_streets(
            streets_gdf=network.edges,
            population=self._data,
            street_buffer=street_buffer,
            aoi=self._aoi,
            transform=self._transform,
            crs=self._crs,
            min_population=min_population,
            scale=scale,
            population_column=self.population_column or "population",
        )
        return PopulationLayer(data=result, population_column=self.population_column, aoi=self._aoi)

    def density(
        self,
        buffer: float = 0,
        return_raster: bool = False,
        min_value: float = 0,
    ):
        """Compute population density with optional spatial smoothing."""
        return pop_module.density(
            population_data=self._data,
            aoi=self._aoi,
            buffer=buffer,
            population_column=self.population_column,
            min_value=min_value,
            transform=self._transform,
            crs=self._crs,
            return_raster=return_raster,
        )

    def to_h3(self, resolution: int, method: str = "distribute") -> "H3Grid":
        """Aggregate the population layer into an H3 grid."""
        return H3Grid(
            data=h3_utils.from_raster(
                self._data, aoi=self._aoi, resolution=resolution, method=method
            ),
            resolution=resolution,
        )

    def plot(
        self,
        column: Optional[str] = None,
        cmap: str = "YlOrRd",
        aoi: Optional[Union[AreaOfInterest, gpd.GeoDataFrame]] = None,
        **kwargs,
    ):
        """Interactive Folium map (only available when data is a GeoDataFrame)."""
        if not isinstance(self._data, gpd.GeoDataFrame):
            raise TypeError("plot() is only available when data is a GeoDataFrame.")
        return plot_helpers.general_map(
            aoi=_bounds_from(aoi) if aoi is not None else None,
            gdfs=[self._data],
            column=column or self.population_column,
            cmap=cmap,
            **kwargs,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        path: Union[str, Path],
        fmt: GeoFmt = "geoparquet",
        overwrite: bool = True,
        copy_raster: bool = False,
    ) -> Path:
        """
        Save the population layer to disk.

        Behaviour depends on the type of ``data``:
          - GeoDataFrame → written in the chosen geodata format.
          - str path     → stored as a reference in the manifest
                           (copied if ``copy_raster=True``).
          - ndarray      → saved as ``.npy`` (requires ``copy_raster=True``).

        Parameters
        ----------
        path : str | Path
            Output directory.
        fmt : GeoFmt
            "geoparquet" | "fgb" | "gpkg" | "geojson" | "shp".
        overwrite : bool
            Replace an existing directory if True.
        copy_raster : bool
            Copy the raster file into the persistence directory.

        Examples
        --------
        >>> pop.save("data/berlin_pop")                     # GDF → GeoParquet
        >>> pop.save("data/berlin_pop", copy_raster=True)   # raster → copy
        """
        directory = _ensure_dir(path, overwrite)
        meta: dict = {
            "fmt": fmt,
            "population_column": self.population_column,
            "data_type": type(self._data).__name__,
        }

        if isinstance(self._data, gpd.GeoDataFrame):
            meta["population_file"] = _save_gdf(self._data, directory, "population", fmt)
            meta["storage"] = "geodataframe"

        elif isinstance(self._data, str):
            src = Path(self._data)
            if copy_raster:
                shutil.copy2(src, directory / src.name)
                meta["raster_file"] = src.name
                meta["storage"] = "raster_copy"
            else:
                meta["raster_path"] = str(src.resolve())
                meta["storage"] = "raster_reference"

        elif isinstance(self._data, np.ndarray):
            if not copy_raster:
                raise ValueError("Pass copy_raster=True to persist a NumPy array.")
            np.save(directory / "population_array.npy", self._data)
            meta["storage"] = "ndarray"

        else:
            raise TypeError(f"Cannot persist data of type {type(self._data)}")

        _write_manifest(directory, "PopulationLayer", meta)
        return directory

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PopulationLayer":
        """
        Load a PopulationLayer from a directory created by ``save()``.

        Examples
        --------
        >>> pop = PopulationLayer.load("data/berlin_pop")
        """
        directory = Path(path)
        manifest = _read_manifest(directory)
        _check_class(manifest, "PopulationLayer")

        obj = cls.__new__(cls)
        obj.population_column = manifest.get("population_column")
        obj._aoi = None
        obj._transform = None
        obj._crs = None

        storage = manifest.get("storage", "geodataframe")
        if storage == "geodataframe":
            obj._data = _load_gdf(directory, manifest["population_file"])
        elif storage == "raster_copy":
            obj._data = str(directory / manifest["raster_file"])
        elif storage == "raster_reference":
            obj._data = manifest["raster_path"]
        elif storage == "ndarray":
            obj._data = np.load(directory / "population_array.npy")
        else:
            raise ValueError(f"Unknown storage type: {storage}")

        return obj

    def __repr__(self) -> str:
        return (
            f"PopulationLayer("
            f"data_type={type(self._data).__name__}, "
            f"column='{self.population_column}')"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 6. AccessibilityAnalyzer
# ──────────────────────────────────────────────────────────────────────────────

class AccessibilityAnalyzer:
    """
    Urban accessibility analysis orchestrator.

    Combines a street network and a set of POIs to compute isochrones and
    accessibility scores.

    Parameters
    ----------
    network : StreetNetwork
        Street network to route on.
    pois : PointsOfInterest
        Points of interest with assigned quality scores.
    distance_matrix : pd.DataFrame | list
        Distance-quality-accessibility matrix or a simple list of distances.
    poi_quality_column : str, optional
        Quality column in the POI GeoDataFrame.
    min_edge_length : float, optional
        Minimum edge length (metres) when projecting POIs onto the network.
    max_dist : float, optional
        Maximum distance (metres) from a POI to the nearest network edge.

    Persistence
    -----------
    Each component is stored in its own sub-directory, allowing the backend
    to load only the part it needs (e.g. only the result without loading the
    full input network).

    Directory layout::

        <path>/
            manifest.json       → metadata + serialised distance_matrix
            network/            → input StreetNetwork
            pois/               → PointsOfInterest
            result_network/     → result StreetNetwork (if run() was called)

    >>> analyzer.save("data/berlin_analysis")
    >>> analyzer.save("data/berlin_analysis", fmt="fgb")
    >>> analyzer = AccessibilityAnalyzer.load("data/berlin_analysis")

    # Partial load from a backend (no need to load the full analysis):
    >>> result = StreetNetwork.load("data/berlin_analysis/result_network")
    """

    def __init__(
        self,
        network: StreetNetwork,
        pois: PointsOfInterest,
        distance_matrix,
        poi_quality_column: Optional[str] = None,
        min_edge_length: float = 0.0,
        max_dist: Optional[float] = None,
    ):
        self.network = network
        self.pois = pois
        self.distance_matrix = distance_matrix
        self.poi_quality_column = poi_quality_column or pois.quality_column
        self.min_edge_length = min_edge_length
        self.max_dist = max_dist
        self._result_network: Optional[StreetNetwork] = None
        self._points_with_osmid: Optional[gpd.GeoDataFrame] = None

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> StreetNetwork:
        """
        Compute accessibility isochrones on the street network.

        Returns
        -------
        StreetNetwork
            Network annotated with an ``accessibility`` column on nodes and edges.
        """
        result = isochrones.graph(
            self.network.graph,
            self.pois.data.copy(),
            distance_matrix=self.distance_matrix,
            poi_quality_col=self.poi_quality_column,
            min_edge_length=self.min_edge_length,
            max_dist=self.max_dist,
            verbose=verbose,
        )
        if isinstance(result, tuple):
            new_G, self._points_with_osmid = result
        else:
            new_G = result

        self._result_network = StreetNetwork(
            aoi=self.network.nodes, G=new_G, network_type=self.network.network_type
        )
        return self._result_network

    def run_buffers(
        self,
        accessibility_values: Optional[list] = None,
        verbose: bool = True,
    ) -> gpd.GeoDataFrame:
        """Compute accessibility as Euclidean buffers (no street routing)."""
        return isochrones.buffers(
            service_geoms=self.pois.data,
            distance_matrix=self.distance_matrix,
            accessibility_values=accessibility_values,
            poi_quality_col=self.poi_quality_column,
            verbose=verbose,
        )

    def default_distance_matrix(self) -> pd.DataFrame:
        """Build a default distance matrix from the POI quality scores."""
        if self.poi_quality_column is None:
            raise ValueError("quality_column is required on the POIs.")
        dm, _ = isochrones.default_distance_matrix(
            self.pois.data,
            distance_steps=(
                self.distance_matrix
                if isinstance(self.distance_matrix, list)
                else list(self.distance_matrix.columns[:-1])
            ),
            poi_quality_column=self.poi_quality_column,
        )
        return dm

    # ── Result properties ─────────────────────────────────────────────────────

    @property
    def result_network(self) -> Optional[StreetNetwork]:
        """Result StreetNetwork after calling ``run()``."""
        return self._result_network

    @property
    def result_nodes(self) -> Optional[gpd.GeoDataFrame]:
        """Node GeoDataFrame with ``accessibility`` column."""
        return self._result_network.nodes if self._result_network else None

    @property
    def result_edges(self) -> Optional[gpd.GeoDataFrame]:
        """Edge GeoDataFrame with ``accessibility`` column."""
        return self._result_network.edges if self._result_network else None

    def to_h3(
        self,
        resolution: int,
        column: str = "accessibility",
        method: str = "max",
    ) -> H3Grid:
        """Aggregate the accessibility result into an H3 grid."""
        if self._result_network is None:
            raise RuntimeError("Call run() before to_h3().")
        edges = self._result_network.edges
        if column not in edges.columns:
            raise ValueError(f"Column '{column}' not found in result edges.")
        return H3Grid(
            data=h3_utils.from_gdf(
                edges[[column, "geometry"]],
                resolution=resolution,
                columns=[column],
                method=method,
            ),
            resolution=resolution,
        )

    def plot(
        self,
        column: str = "accessibility",
        cmap: str = "RdYlGn",
        aoi: Optional[Union[AreaOfInterest, gpd.GeoDataFrame]] = None,
        show_pois: bool = True,
        **kwargs,
    ):
        """Interactive Folium map of the accessibility result."""
        if self._result_network is None:
            raise RuntimeError("Call run() before plot().")
        return plot_helpers.general_map(
            aoi=_bounds_from(aoi) if aoi is not None else None,
            gdfs=[self._result_network.edges],
            column=column,
            cmap=cmap,
            pois=[self.pois.data] if show_pois else [],
            **kwargs,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        path: Union[str, Path],
        fmt: GeoFmt = "geoparquet",
        overwrite: bool = True,
    ) -> Path:
        """
        Save the full analysis to disk.

        Each component is written to its own sub-directory, enabling partial
        loads from the backend without reading the entire analysis into memory.

        Parameters
        ----------
        path : str | Path
            Root output directory.
        fmt : GeoFmt
            Format applied to all geodata components.
        overwrite : bool
            Replace an existing directory if True.

        Examples
        --------
        >>> analyzer.save("data/berlin_analysis")
        >>> analyzer.save("data/berlin_analysis", fmt="fgb")

        # Partial load from a backend:
        >>> result = StreetNetwork.load("data/berlin_analysis/result_network")
        """
        directory = _ensure_dir(path, overwrite)

        self.network.save(directory / "network", fmt=fmt, overwrite=True)
        self.pois.save(directory / "pois", fmt=fmt, overwrite=True)

        has_result = self._result_network is not None
        if has_result:
            self._result_network.save(directory / "result_network", fmt=fmt, overwrite=True)

        if isinstance(self.distance_matrix, list):
            dm_serial: dict = {"type": "list", "values": self.distance_matrix}
        elif isinstance(self.distance_matrix, pd.DataFrame):
            dm_serial = {
                "type": "dataframe",
                "data": self.distance_matrix.to_dict(orient="list"),
            }
        else:
            dm_serial = {"type": "unknown", "values": str(self.distance_matrix)}

        _write_manifest(directory, "AccessibilityAnalyzer", {
            "fmt": fmt,
            "poi_quality_column": self.poi_quality_column,
            "min_edge_length": self.min_edge_length,
            "max_dist": self.max_dist,
            "has_result": has_result,
            "distance_matrix": dm_serial,
        })
        return directory

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AccessibilityAnalyzer":
        """
        Load an AccessibilityAnalyzer from a directory created by ``save()``.

        Parameters
        ----------
        path : str | Path
            Directory generated by ``save()``.

        Examples
        --------
        >>> analyzer = AccessibilityAnalyzer.load("data/berlin_analysis")
        """
        directory = Path(path)
        manifest = _read_manifest(directory)
        _check_class(manifest, "AccessibilityAnalyzer")

        dm_serial = manifest.get("distance_matrix", {})
        if dm_serial.get("type") == "list":
            distance_matrix = dm_serial["values"]
        elif dm_serial.get("type") == "dataframe":
            distance_matrix = pd.DataFrame(dm_serial["data"])
        else:
            distance_matrix = dm_serial.get("values")

        obj = cls.__new__(cls)
        obj.network = StreetNetwork.load(directory / "network")
        obj.pois = PointsOfInterest.load(directory / "pois")
        obj.distance_matrix = distance_matrix
        obj.poi_quality_column = manifest.get("poi_quality_column")
        obj.min_edge_length = manifest.get("min_edge_length", 0.0)
        obj.max_dist = manifest.get("max_dist")
        obj._points_with_osmid = None
        obj._result_network = (
            StreetNetwork.load(directory / "result_network")
            if manifest.get("has_result") else None
        )
        return obj

    def __repr__(self) -> str:
        return (
            f"AccessibilityAnalyzer("
            f"pois={len(self.pois)}, "
            f"network_type='{self.network.network_type}', "
            f"ran={self._result_network is not None})"
        )