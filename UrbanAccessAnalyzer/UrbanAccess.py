"""
urban_access.py
===============
Clases de alto nivel para análisis de accesibilidad urbana.

Organización:
    - AreaOfInterest        → AOI con geocodificación y descarga de límites
    - StreetNetwork         → Grafo de calles (descarga, simplificación, isocronas)
    - PointsOfInterest      → POIs desde OSM o GeoDataFrame externo
    - H3Grid                → Rejilla hexagonal H3 (rasterización y agregación)
    - PopulationLayer       → Población WorldPop o GeoDataFrame propio
    - AccessibilityAnalyzer → Orquestador: une red + POIs → accesibilidad

Cada clase expone:
    .data          → GeoDataFrame / DataFrame / grafo con el resultado principal
    .plot()        → Mapa Folium interactivo
    .to_h3()       → Convierte .data a H3Grid
    .save(path)    → Persiste el objeto en disco
    .load(path)    → (classmethod) Restaura el objeto desde disco

════════════════════════════════════════════════════════════════════════════════
Estrategia de persistencia
════════════════════════════════════════════════════════════════════════════════

Cada instancia se guarda como un directorio con:

    <path>/
        manifest.json         → clase, versión, formato, metadatos escalares
        <datos específicos>   → ver tabla abajo

Clase                Archivos generados
──────────────────   ────────────────────────────────────────────────────────
AreaOfInterest       aoi.<ext>
StreetNetwork        nodes.<ext>  edges.<ext>  graph_attrs.json
PointsOfInterest     pois.<ext>
H3Grid               grid.parquet              (siempre Parquet; sin geometría)
PopulationLayer      population.<ext>  (si GeoDataFrame) | referencia (si raster)
AccessibilityAnalyzer
                     network/  pois/  [result_network/]  manifest.json

Formatos geoespaciales soportados (parámetro ``fmt``):
    "geoparquet"  → GeoParquet + zstd + geoarrow  [DEFAULT — backend óptimo]
    "parquet"     → alias de "geoparquet"
    "fgb"         → FlatGeobuf                     [streaming / map tiles]
    "gpkg"        → GeoPackage                     [compatible QGIS / GDAL]
    "geojson"     → GeoJSON                        [interoperabilidad web]
    "shp"         → Shapefile                      [legacy GIS]

H3Grid usa siempre Parquet plano (sin geometría). La geometría se reconstruye
desde el índice h3_cell bajo demanda → archivo mínimo, consultable con
DuckDB / Polars / pandas sin dependencias geoespaciales.

StreetNetwork separa nodos y aristas en ficheros independientes:
    • Nodos en GeoParquet/Parquet: sin columna geometry (x,y son suficientes),
      ahorrando ~30 % de espacio. En otros formatos se guarda con Point.
    • Aristas: siempre con geometría LineString.
    • graph_attrs.json: atributos del grafo (crs, name, …).
    • El grafo NetworkX se reconstruye con ox.graph_from_gdfs() al cargar.
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

# ── Lazy imports ───────────────────────────────────────────────────────────────
try:
    import osmnx as ox
except ImportError:
    ox = None  # type: ignore

try:
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
except ImportError:
    import geometry_utils
    import graph_processing
    import h3_utils
    import isochrones
    import osm
    import plot_helpers
    import poi_utils
    import population as pop_module
    import quality
    import utils


# ──────────────────────────────────────────────────────────────────────────────
# Tipos y constantes
# ──────────────────────────────────────────────────────────────────────────────

GeoFmt = Literal["geoparquet", "parquet", "fgb", "gpkg", "geojson", "shp"]

_FORMAT_VERSION = "1.0"

# Mapeo formato → extensión
_FMT_EXT: Dict[str, str] = {
    "geoparquet": ".geoparquet",
    "parquet":    ".geoparquet",   # alias
    "fgb":        ".fgb",
    "gpkg":       ".gpkg",
    "geojson":    ".geojson",
    "shp":        ".shp",
}


# ──────────────────────────────────────────────────────────────────────────────
# I/O de bajo nivel — GeoDataFrame
# ──────────────────────────────────────────────────────────────────────────────

def _save_gdf(gdf: gpd.GeoDataFrame, directory: Path, stem: str, fmt: str) -> str:
    """
    Guarda un GeoDataFrame en el formato indicado.

    Preserva el índice nombrado como columna para que la carga sea exacta.
    Devuelve el nombre del archivo relativo al directorio (para el manifest).
    """
    fmt = fmt.lower()
    if fmt not in _FMT_EXT:
        raise ValueError(
            f"Formato no soportado: '{fmt}'. "
            f"Opciones: {list(_FMT_EXT)}"
        )
    ext = _FMT_EXT[fmt]
    filename = f"{stem}{ext}"
    dest = directory / filename

    # Preservar índice con nombre como columna
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

    return filename


def _load_gdf(directory: Path, filename: str) -> gpd.GeoDataFrame:
    """Carga un GeoDataFrame detectando el formato por extensión."""
    p = directory / filename
    if not p.exists():
        raise FileNotFoundError(f"No se encontró: {p}")
    ext = p.suffix.lower()
    if ext in (".geoparquet", ".parquet"):
        return gpd.read_parquet(p)
    return gpd.read_file(p)


# ──────────────────────────────────────────────────────────────────────────────
# I/O de bajo nivel — DataFrame plano (H3Grid)
# ──────────────────────────────────────────────────────────────────────────────

def _save_df(df: pd.DataFrame, directory: Path, stem: str) -> str:
    """Guarda un DataFrame sin geometría como Parquet comprimido."""
    filename = f"{stem}.parquet"
    df.to_parquet(directory / filename, compression="zstd")
    return filename


def _load_df(directory: Path, stem: str) -> pd.DataFrame:
    return pd.read_parquet(directory / f"{stem}.parquet")


# ──────────────────────────────────────────────────────────────────────────────
# I/O de bajo nivel — Grafo OSMnx
# ──────────────────────────────────────────────────────────────────────────────

def _sanitize_col(series: pd.Series) -> pd.Series:
    """Convierte listas/dicts a JSON string para formatos columnares."""
    if series.dtype != object:
        return series
    return series.apply(
        lambda x: json.dumps(x, default=str) if isinstance(x, (list, dict)) else x
    )


def _deserialize_col(series: pd.Series) -> pd.Series:
    """Recupera listas/dicts desde JSON string si procede."""
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
    Persiste un grafo OSMnx como par nodos + aristas.

    Decisiones de diseño
    ────────────────────
    GeoParquet/Parquet
      • Nodos: solo columnas escalares + x, y. Se omite la columna geometry
        Point (redundante: se reconstruye desde x,y al cargar).
        Ahorro ≈ 30 % de espacio respecto a guardar la geometría.
      • Aristas: con geometría LineString + columnas aplanadas desde MultiIndex.
        Tipos complejos (listas OSM) → JSON string.

    Otros formatos (fgb, gpkg, geojson, shp)
      • Nodos: con geometría Point (necesario para QGIS y herramientas GIS).
      • Aristas: con geometría LineString.

    graph_attrs.json almacena los atributos del grafo (crs, name, etc.)
    y es siempre JSON independientemente del fmt elegido.

    Devuelve un dict con los nombres de archivo para el manifest.
    """
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    # Serializar columnas con tipos complejos
    for col in nodes_gdf.select_dtypes("object").columns:
        nodes_gdf[col] = _sanitize_col(nodes_gdf[col])
    for col in edges_gdf.select_dtypes("object").columns:
        edges_gdf[col] = _sanitize_col(edges_gdf[col])

    fmt_norm = fmt.lower()

    # ── Nodos ─────────────────────────────────────────────────────────────────
    nodes_flat = nodes_gdf.reset_index()  # osmid → columna

    if fmt_norm in ("geoparquet", "parquet"):
        # Sin geometría: x e y son suficientes y más compactos
        nodes_df = pd.DataFrame(nodes_flat.drop(columns=["geometry"]))
        nodes_file = "nodes.parquet"
        nodes_df.to_parquet(directory / nodes_file, compression="zstd")
    else:
        nodes_file = _save_gdf(nodes_flat, directory, "nodes", fmt)

    # ── Aristas ───────────────────────────────────────────────────────────────
    edges_flat = edges_gdf.reset_index()  # u, v, key → columnas
    edges_file = _save_gdf(edges_flat, directory, "edges", fmt)

    # ── Atributos del grafo ───────────────────────────────────────────────────
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
    Reconstruye el grafo OSMnx desde los archivos de nodos y aristas.

    Tolerante: recalcula 'length' si no está en las aristas.
    """
    graph_attrs = json.loads(
        (directory / "graph_attrs.json").read_text(encoding="utf-8")
    )

    # ── Nodos ─────────────────────────────────────────────────────────────────
    nodes_path = directory / nodes_file

    # Parquet plano (sin geometría) vs GeoParquet/otro formato con geometría
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

    # ── Aristas ───────────────────────────────────────────────────────────────
    edges_gdf = _load_gdf(directory, edges_file)
    for col in edges_gdf.select_dtypes("object").columns:
        edges_gdf[col] = _deserialize_col(edges_gdf[col])

    for c in ("u", "v", "key"):
        if c in edges_gdf.columns:
            edges_gdf[c] = edges_gdf[c].astype(int)
    edges_gdf = edges_gdf.set_index(["u", "v", "key"])

    if "length" not in edges_gdf.columns:
        warnings.warn(
            "Columna 'length' no encontrada en aristas; recalculando.", UserWarning
        )
        edges_gdf["length"] = edges_gdf.geometry.length

    return ox.graph_from_gdfs(nodes_gdf, edges_gdf, graph_attrs=graph_attrs)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de manifest y directorio
# ──────────────────────────────────────────────────────────────────────────────

def _write_manifest(directory: Path, class_name: str, meta: dict) -> None:
    manifest = {"class": class_name, "version": _FORMAT_VERSION, **meta}
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


def _read_manifest(directory: Path) -> dict:
    p = directory / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"No se encontró manifest.json en '{directory}'")
    return json.loads(p.read_text(encoding="utf-8"))


def _check_class(manifest: dict, expected: str) -> None:
    found = manifest.get("class")
    if found != expected:
        raise TypeError(
            f"Se intentó cargar un '{expected}' pero el directorio "
            f"contiene un '{found}'. Usa la clase correcta para load()."
        )


def _ensure_dir(path: Union[str, Path], overwrite: bool) -> Path:
    p = Path(path)
    if p.exists():
        if not overwrite:
            raise FileExistsError(
                f"El directorio '{p}' ya existe. "
                f"Usa overwrite=True para sobreescribir."
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
        raise TypeError(f"Tipo no soportado: {type(data)}")
    if crs is not None:
        gdf = gdf.to_crs(crs)
    return gdf


# ──────────────────────────────────────────────────────────────────────────────
# 1. AreaOfInterest
# ──────────────────────────────────────────────────────────────────────────────

class AreaOfInterest:
    """
    Define y gestiona el área de interés (AOI) del análisis.

    Parameters
    ----------
    data : str | GeoDataFrame | GeoSeries
        Nombre de ciudad (geocodifica automáticamente), ruta a archivo, o geodata.
    crs : int, optional
        CRS de salida. Por defecto 4326.
    buffer : float, optional
        Buffer en metros aplicado tras cargar la geometría.

    Persistencia
    ------------
    >>> aoi.save("output/my_aoi")                  # GeoParquet por defecto
    >>> aoi.save("output/my_aoi", fmt="geojson")   # GeoJSON
    >>> aoi.save("output/my_aoi", fmt="gpkg")      # GeoPackage
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
            gdf.geometry = gdf.geometry.buffer(buffer)
            gdf = gdf.to_crs(4326)
        if crs != 4326:
            gdf = gdf.to_crs(crs)

        self.data: gpd.GeoDataFrame = gdf
        self.crs = crs
        self._buffer = buffer

    @property
    def geometry(self) -> gpd.GeoSeries:
        return self.data.geometry

    @property
    def union(self):
        return self.data.union_all()

    @property
    def bounds(self) -> tuple:
        return tuple(self.data.total_bounds)

    def clip(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = gdf.to_crs(self.data.crs)
        return gdf[gdf.intersects(self.union)]

    def plot(self, **kwargs):
        return plot_helpers.general_map(aoi=self.data, **kwargs)

    # ── Persistencia ──────────────────────────────────────────────────────────

    def save(
        self,
        path: Union[str, Path],
        fmt: GeoFmt = "geoparquet",
        overwrite: bool = True,
    ) -> Path:
        """
        Guarda el AOI en disco.

        Parameters
        ----------
        path : str | Path
            Directorio de salida (se crea si no existe).
        fmt : GeoFmt
            Formato de salida: "geoparquet" | "fgb" | "gpkg" | "geojson" | "shp".
        overwrite : bool
            Si True (defecto), sobreescribe el directorio existente.

        Returns
        -------
        Path del directorio creado.

        Examples
        --------
        >>> aoi.save("data/berlin_aoi")
        >>> aoi.save("data/berlin_aoi", fmt="geojson")
        """
        directory = _ensure_dir(path, overwrite)
        filename = _save_gdf(self.data, directory, "aoi", fmt)
        _write_manifest(directory, "AreaOfInterest", {
            "fmt": fmt,
            "crs": self.crs,
            "buffer": self._buffer,
            "aoi_file": filename,
        })
        return directory

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AreaOfInterest":
        """
        Carga un AreaOfInterest desde un directorio creado con ``save()``.

        Parameters
        ----------
        path : str | Path
            Directorio de persistencia.

        Examples
        --------
        >>> aoi = AreaOfInterest.load("data/berlin_aoi")
        """
        directory = Path(path)
        manifest = _read_manifest(directory)
        _check_class(manifest, "AreaOfInterest")

        gdf = _load_gdf(directory, manifest["aoi_file"])
        obj = cls.__new__(cls)
        obj.data = gdf
        obj.crs = manifest.get("crs", 4326)
        obj._buffer = manifest.get("buffer", 0.0)
        return obj

    def __repr__(self):
        return f"AreaOfInterest(geometries={len(self.data)}, crs={self.data.crs.to_epsg()})"


# ──────────────────────────────────────────────────────────────────────────────
# 2. StreetNetwork
# ──────────────────────────────────────────────────────────────────────────────

class StreetNetwork:
    """
    Grafo de red de calles basado en OSMnx.

    Parameters
    ----------
    aoi : AreaOfInterest | GeoDataFrame | GeoSeries
        Área de interés para la descarga.
    network_type : str
        "walk" | "bike" | "drive" | "all".
    custom_filter : str, optional
        Filtro OSM personalizado (sintaxis osmnx).
    min_edge_length : float, optional
        Simplificación mínima de arista en metros. 0 = sin simplificación.
    G : networkx.MultiDiGraph, optional
        Grafo ya construido (omite la descarga).

    Persistencia
    ------------
    >>> net.save("data/berlin_walk")                # GeoParquet (defecto)
    >>> net.save("data/berlin_walk", fmt="fgb")     # FlatGeobuf
    >>> net.save("data/berlin_walk", fmt="gpkg")    # GeoPackage (QGIS)
    >>> net = StreetNetwork.load("data/berlin_walk")

    Estructura del directorio:
        nodes.parquet       (GeoParquet) o nodes.<ext>
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
        bounds = aoi.data if isinstance(aoi, AreaOfInterest) else _ensure_gdf(aoi)

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

    # ── Propiedades (lazy) ────────────────────────────────────────────────────

    @property
    def graph(self):
        return self._G

    @property
    def nodes(self) -> gpd.GeoDataFrame:
        if self._nodes is None:
            self._nodes, self._edges = ox.graph_to_gdfs(self._G)
        return self._nodes

    @property
    def edges(self) -> gpd.GeoDataFrame:
        if self._edges is None:
            self._nodes, self._edges = ox.graph_to_gdfs(self._G)
        return self._edges

    @property
    def data(self) -> gpd.GeoDataFrame:
        return self.edges

    def _invalidate_cache(self):
        self._nodes = None
        self._edges = None

    # ── Métodos ───────────────────────────────────────────────────────────────

    def simplify(
        self, min_edge_length=0, min_edge_separation=0,
        loops=True, multi=True, undirected=False,
    ) -> "StreetNetwork":
        new_G = graph_processing.simplify_graph(
            self._G, min_edge_length=min_edge_length,
            min_edge_separation=min_edge_separation,
            loops=loops, multi=multi, undirected=undirected,
        )
        return StreetNetwork(aoi=self.nodes, G=new_G, network_type=self.network_type)

    def crop(self, aoi) -> "StreetNetwork":
        aoi_gdf = aoi.data if isinstance(aoi, AreaOfInterest) else aoi
        new_G = graph_processing.crop_by_aoi(aoi_gdf, G=self._G)
        return StreetNetwork(aoi=aoi_gdf, G=new_G, network_type=self.network_type)

    def nearest_nodes(self, geometries, max_dist=None) -> list:
        return graph_processing.nearest_nodes(geometries, self._G, max_dist=max_dist)

    def add_points(self, points, max_dist=None, min_edge_length=0):
        new_G, osmids = graph_processing.add_points_to_graph(
            points, self._G, max_dist=max_dist, min_edge_length=min_edge_length
        )
        return StreetNetwork(aoi=self.nodes, G=new_G, network_type=self.network_type), osmids

    def plot(self, aoi=None, column=None, cmap=None, **kwargs):
        aoi_gdf = aoi.data if isinstance(aoi, AreaOfInterest) else aoi
        return plot_helpers.general_map(
            aoi=aoi_gdf, gdfs=[self.edges], column=column, cmap=cmap, **kwargs
        )

    # ── Persistencia ──────────────────────────────────────────────────────────

    def save(
        self,
        path: Union[str, Path],
        fmt: GeoFmt = "geoparquet",
        overwrite: bool = True,
    ) -> Path:
        """
        Guarda la red de calles en disco.

        En formato GeoParquet/Parquet los nodos se guardan sin columna
        geometry (solo x, y) ahorrando ~30 % de espacio. En otros formatos
        se incluye la geometría Point para compatibilidad GIS.
        Las aristas siempre se guardan con geometría LineString.

        Parameters
        ----------
        path : str | Path
            Directorio de salida.
        fmt : GeoFmt
            "geoparquet" | "fgb" | "gpkg" | "geojson" | "shp".
        overwrite : bool
            Sobreescribir si existe.

        Examples
        --------
        >>> net.save("data/berlin_walk")
        >>> net.save("data/berlin_walk", fmt="gpkg")   # abre en QGIS
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
        Carga una StreetNetwork desde un directorio creado con ``save()``.

        Parameters
        ----------
        path : str | Path
            Directorio de persistencia.

        Examples
        --------
        >>> net = StreetNetwork.load("data/berlin_walk")
        """
        directory = Path(path)
        manifest = _read_manifest(directory)
        _check_class(manifest, "StreetNetwork")

        G = _load_graph(
            directory,
            nodes_file=manifest["nodes_file"],
            edges_file=manifest["edges_file"],
        )
        obj = cls.__new__(cls)
        obj._G = G
        obj.network_type = manifest.get("network_type", "walk")
        obj._custom_filter = manifest.get("custom_filter")
        obj._nodes = None
        obj._edges = None
        return obj

    def __repr__(self):
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
    Carga y preprocesa puntos de interés (POIs).

    Parameters
    ----------
    data : GeoDataFrame | str | None
        GeoDataFrame de POIs, ruta a archivo, o None (usar from_osm_*).
    quality_column : str, optional
        Columna con puntuación de calidad [0, 1].
    aoi : AreaOfInterest | GeoDataFrame, optional
        Filtra los POIs al AOI.

    Persistencia
    ------------
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
            aoi_gdf = aoi.data if isinstance(aoi, AreaOfInterest) else aoi
            self._data = self._data[
                self._data.to_crs(aoi_gdf.crs).intersects(aoi_gdf.union_all())
            ]

    # ── Constructores OSM ─────────────────────────────────────────────────────

    @classmethod
    def _from_osm(cls, fn, aoi, **kwargs) -> "PointsOfInterest":
        bounds = aoi.data if isinstance(aoi, AreaOfInterest) else aoi
        return cls(data=fn(bounds, **kwargs), aoi=aoi)

    @classmethod
    def from_osm_parks(cls, aoi, **kwargs) -> "PointsOfInterest":
        """Descarga parques y áreas verdes desde OSM."""
        return cls._from_osm(osm.green_areas, aoi, **kwargs)

    @classmethod
    def from_osm_bus_stops(cls, aoi) -> "PointsOfInterest":
        """Descarga paradas de autobús desde OSM."""
        return cls._from_osm(osm.bus_stops, aoi)

    @classmethod
    def from_osm_schools(cls, aoi) -> "PointsOfInterest":
        """Descarga colegios desde OSM."""
        return cls._from_osm(osm.schools, aoi)

    @classmethod
    def from_osm_healthcare(cls, aoi) -> "PointsOfInterest":
        """Descarga centros sanitarios desde OSM."""
        return cls._from_osm(osm.healthcare, aoi)

    @classmethod
    def from_osm_groceries(cls, aoi) -> "PointsOfInterest":
        """Descarga supermercados desde OSM."""
        return cls._from_osm(osm.groceries, aoi)

    @classmethod
    def from_osm_restaurants(cls, aoi) -> "PointsOfInterest":
        """Descarga restaurantes y bares desde OSM."""
        return cls._from_osm(osm.restaurants, aoi)

    @classmethod
    def from_overpass_query(cls, aoi, query: str) -> "PointsOfInterest":
        """Ejecuta una consulta Overpass personalizada."""
        bounds = aoi.data if isinstance(aoi, AreaOfInterest) else aoi
        return cls(data=osm.overpass_api_query(query, bounds), aoi=aoi)

    # ── Propiedad principal ───────────────────────────────────────────────────

    @property
    def data(self) -> gpd.GeoDataFrame:
        if self._data is None:
            raise ValueError("No hay POIs cargados. Usa un método from_osm_* primero.")
        return self._data

    # ── Preprocesado ──────────────────────────────────────────────────────────

    def assign_quality(
        self, quality_func: Callable, column_name: str = "poi_quality"
    ) -> "PointsOfInterest":
        """Asigna calidad con una función personalizada ``f(gdf) → Series``."""
        self._data = self._data.copy()
        self._data[column_name] = quality_func(self._data)
        self.quality_column = column_name
        return self

    def assign_quality_by_area(
        self, area_steps, large_is_better=True, column_name="poi_quality"
    ) -> "PointsOfInterest":
        """Asigna calidad [0,1] según área de la geometría."""
        self._data = self._data.copy()
        self._data[column_name] = poi_utils.quality_by_area(
            self._data, area_steps, large_is_better=large_is_better
        )
        self.quality_column = column_name
        return self

    def assign_quality_by_values(
        self, column, value_priority, quality_column_name="poi_quality"
    ) -> "PointsOfInterest":
        """Asigna calidad [0,1] según orden de prioridad de valores categóricos."""
        self._data = self._data.copy()
        self._data[quality_column_name] = poi_utils.quality_by_values(
            self._data[column], value_priority
        )
        self.quality_column = quality_column_name
        return self

    def to_points(self, street_edges: gpd.GeoDataFrame) -> "PointsOfInterest":
        """Convierte polígonos a puntos proyectados en la red de calles."""
        return PointsOfInterest(
            data=poi_utils.polygons_to_points(self._data, street_edges),
            quality_column=self.quality_column,
        )

    def clip(self, aoi) -> "PointsOfInterest":
        aoi_gdf = aoi.data if isinstance(aoi, AreaOfInterest) else aoi
        clipped = self._data[
            self._data.to_crs(aoi_gdf.crs).intersects(aoi_gdf.union_all())
        ]
        return PointsOfInterest(data=clipped, quality_column=self.quality_column)

    def to_h3(self, resolution, columns=None, method="max", buffer=0.0) -> "H3Grid":
        cols = columns or ([self.quality_column] if self.quality_column else [])
        df = h3_utils.from_gdf(
            self._data, resolution=resolution, columns=cols,
            method=method, buffer=buffer,
        )
        return H3Grid(data=df, resolution=resolution)

    def plot(self, aoi=None, column=None, cmap=None, **kwargs):
        aoi_gdf = aoi.data if isinstance(aoi, AreaOfInterest) else aoi
        col = column or self.quality_column
        return plot_helpers.general_map(
            aoi=aoi_gdf, pois=[self._data],
            poi_column=col,
            poi_cmap=cmap or ("viridis" if col else None),
            **kwargs,
        )

    # ── Persistencia ──────────────────────────────────────────────────────────

    def save(
        self,
        path: Union[str, Path],
        fmt: GeoFmt = "geoparquet",
        overwrite: bool = True,
    ) -> Path:
        """
        Guarda los POIs en disco.

        Parameters
        ----------
        path : str | Path
            Directorio de salida.
        fmt : GeoFmt
            "geoparquet" | "fgb" | "gpkg" | "geojson" | "shp".
        overwrite : bool
            Sobreescribir si existe.

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
        Carga un PointsOfInterest desde disco.

        Examples
        --------
        >>> pois = PointsOfInterest.load("data/berlin_parks")
        """
        directory = Path(path)
        manifest = _read_manifest(directory)
        _check_class(manifest, "PointsOfInterest")

        gdf = _load_gdf(directory, manifest["pois_file"])
        obj = cls.__new__(cls)
        obj._data = gdf
        obj.quality_column = manifest.get("quality_column")
        return obj

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        n = len(self._data) if self._data is not None else 0
        return f"PointsOfInterest(n={n}, quality_column='{self.quality_column}')"


# ──────────────────────────────────────────────────────────────────────────────
# 4. H3Grid
# ──────────────────────────────────────────────────────────────────────────────

class H3Grid:
    """
    Rejilla hexagonal H3 con datos agregados.

    El dato en disco es siempre Parquet plano (sin geometría). La geometría
    hexagonal se reconstruye bajo demanda desde el índice h3_cell, manteniendo
    el archivo mínimo y directamente consultable con DuckDB / Polars / pandas.

    Parameters
    ----------
    data : pd.DataFrame | gpd.GeoDataFrame
        DataFrame indexado por ``h3_cell`` con los datos de la rejilla.
    resolution : int, optional
        Resolución H3 de la rejilla.

    Persistencia
    ------------
    H3Grid usa siempre Parquet (no tiene sentido ofrecer GeoJSON/SHP porque
    el índice h3_cell ya codifica la geometría de cada hexágono).

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

    # ── Constructores ─────────────────────────────────────────────────────────

    @classmethod
    def from_gdf(cls, gdf, resolution, columns=None, value_order=None,
                 buffer=0.0, method="max") -> "H3Grid":
        df = h3_utils.from_gdf(
            gdf, resolution=resolution, columns=columns,
            value_order=value_order, buffer=buffer, method=method,
        )
        return cls(data=df, resolution=resolution)

    @classmethod
    def from_raster(cls, raster, resolution, aoi=None, method="distribute",
                    nodata=None) -> "H3Grid":
        if isinstance(aoi, AreaOfInterest):
            aoi = aoi.data
        df = h3_utils.from_raster(
            raster, aoi=aoi, resolution=resolution, method=method, nodata=nodata
        )
        return cls(data=df, resolution=resolution)

    # ── Propiedades ───────────────────────────────────────────────────────────

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """GeoDataFrame con geometrías hexagonales (conversión lazy)."""
        if isinstance(self._data, gpd.GeoDataFrame) and "geometry" in self._data.columns:
            return self._data
        return h3_utils.to_gdf(self._data)

    # ── Métodos ───────────────────────────────────────────────────────────────

    def resample(self, target_resolution, columns=None, method="max") -> "H3Grid":
        df = h3_utils.resample(self._data, target_resolution, columns=columns, method=method)
        return H3Grid(data=df, resolution=target_resolution)

    def join(self, other: "H3Grid", how="left") -> "H3Grid":
        merged = self._data.join(other._data, how=how, rsuffix="_right")
        return H3Grid(data=merged, resolution=self.resolution)

    def clip(self, aoi) -> "H3Grid":
        aoi_gdf = aoi.data if isinstance(aoi, AreaOfInterest) else aoi
        gdf = self.gdf.to_crs(aoi_gdf.crs)
        mask = gdf.intersects(aoi_gdf.union_all())
        return H3Grid(data=self._data.loc[mask], resolution=self.resolution)

    def plot(self, column=None, cmap="viridis", aoi=None, **kwargs):
        aoi_gdf = aoi.data if isinstance(aoi, AreaOfInterest) else aoi
        return plot_helpers.general_map(
            aoi=aoi_gdf, gdfs=[self._data], column=column, cmap=cmap, **kwargs
        )

    # ── Persistencia ──────────────────────────────────────────────────────────

    def save(
        self,
        path: Union[str, Path],
        overwrite: bool = True,
    ) -> Path:
        """
        Guarda la rejilla H3 como Parquet plano.

        El formato es siempre Parquet (sin geometría). Esto permite consultas
        directas con DuckDB/Polars sin dependencias geoespaciales, y hace el
        archivo 3-5× más pequeño que un GeoJSON equivalente.
        La geometría se reconstruye al llamar a ``.gdf``.

        Parameters
        ----------
        path : str | Path
            Directorio de salida.
        overwrite : bool
            Sobreescribir si existe.

        Examples
        --------
        >>> grid.save("data/berlin_h3_res9")
        >>> grid = H3Grid.load("data/berlin_h3_res9")
        """
        directory = _ensure_dir(path, overwrite)

        # Descartar geometría si es GeoDataFrame
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
        Carga una H3Grid desde disco.

        Examples
        --------
        >>> grid = H3Grid.load("data/berlin_h3_res9")
        """
        directory = Path(path)
        manifest = _read_manifest(directory)
        _check_class(manifest, "H3Grid")

        df = _load_df(directory, "grid")
        obj = cls.__new__(cls)
        obj._data = df
        obj.resolution = manifest.get("resolution")
        return obj

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return (
            f"H3Grid(cells={len(self._data)}, "
            f"resolution={self.resolution}, "
            f"columns={list(self._data.columns)})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 5. PopulationLayer
# ──────────────────────────────────────────────────────────────────────────────

class PopulationLayer:
    """
    Capa de población (WorldPop o datos propios).

    Parameters
    ----------
    data : str | ndarray | GeoDataFrame
        Ruta a raster WorldPop, ndarray, o GeoDataFrame con columna de población.
    population_column : str, optional
        Columna de población si ``data`` es un GeoDataFrame.
    aoi : AreaOfInterest | GeoDataFrame, optional
        AOI para recorte.

    Persistencia
    ------------
    Si ``data`` es GeoDataFrame → se guarda en el formato elegido.
    Si ``data`` es ruta a raster → se guarda la ruta en el manifest
    (usa ``copy_raster=True`` para copiar el archivo).
    Si ``data`` es ndarray → usa ``copy_raster=True`` (guarda como .npy).

    >>> pop.save("data/berlin_pop")
    >>> pop.save("data/berlin_pop", fmt="gpkg", copy_raster=True)
    >>> pop = PopulationLayer.load("data/berlin_pop")
    """

    def __init__(self, data, population_column=None, aoi=None):
        self._data = data
        self.population_column = population_column
        self._aoi = aoi.data if isinstance(aoi, AreaOfInterest) else aoi
        self._transform = None
        self._crs = None

    @classmethod
    def from_worldpop(cls, aoi, year: int, folder=None,
                      resolution="100m") -> "PopulationLayer":
        """Descarga el raster WorldPop para el AOI y año dados."""
        from datetime import datetime
        aoi_gdf = aoi.data if isinstance(aoi, AreaOfInterest) else aoi
        path = pop_module.download_worldpop_population(
            aoi_gdf, date=datetime(year=year, month=1, day=1),
            folder=folder, resolution=resolution,
        )
        return cls(data=path, aoi=aoi)

    @property
    def data(self):
        return self._data

    def filter_by_streets(self, network, street_buffer=50,
                           min_population=0, scale=True) -> "PopulationLayer":
        result = pop_module.filter_population_by_streets(
            streets_gdf=network.edges, population=self._data,
            street_buffer=street_buffer, aoi=self._aoi,
            transform=self._transform, crs=self._crs,
            min_population=min_population, scale=scale,
            population_column=self.population_column or "population",
        )
        return PopulationLayer(
            data=result, population_column=self.population_column, aoi=self._aoi
        )

    def density(self, buffer=0, return_raster=False, min_value=0):
        return pop_module.density(
            population_data=self._data, aoi=self._aoi, buffer=buffer,
            population_column=self.population_column, min_value=min_value,
            transform=self._transform, crs=self._crs, return_raster=return_raster,
        )

    def to_h3(self, resolution, method="distribute") -> "H3Grid":
        df = h3_utils.from_raster(
            self._data, aoi=self._aoi, resolution=resolution, method=method
        )
        return H3Grid(data=df, resolution=resolution)

    def plot(self, column=None, cmap="YlOrRd", aoi=None, **kwargs):
        if not isinstance(self._data, gpd.GeoDataFrame):
            raise TypeError("plot() solo disponible cuando data es GeoDataFrame.")
        aoi_gdf = aoi.data if isinstance(aoi, AreaOfInterest) else aoi
        return plot_helpers.general_map(
            aoi=aoi_gdf, gdfs=[self._data],
            column=column or self.population_column, cmap=cmap, **kwargs,
        )

    # ── Persistencia ──────────────────────────────────────────────────────────

    def save(
        self,
        path: Union[str, Path],
        fmt: GeoFmt = "geoparquet",
        overwrite: bool = True,
        copy_raster: bool = False,
    ) -> Path:
        """
        Guarda la capa de población en disco.

        Comportamiento según el tipo de ``data``:
          - GeoDataFrame → geodata en el formato elegido.
          - ruta str     → referencia en manifest (copia si copy_raster=True).
          - ndarray      → .npy (solo si copy_raster=True).

        Parameters
        ----------
        path : str | Path
            Directorio de salida.
        fmt : GeoFmt
            "geoparquet" | "fgb" | "gpkg" | "geojson" | "shp".
        overwrite : bool
            Sobreescribir si existe.
        copy_raster : bool
            Si True, copia el archivo raster al directorio de persistencia.

        Examples
        --------
        >>> pop.save("data/berlin_pop")                     # GDF → GeoParquet
        >>> pop.save("data/berlin_pop", copy_raster=True)   # raster → copia
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
                dest = directory / src.name
                shutil.copy2(src, dest)
                meta["raster_file"] = src.name
                meta["storage"] = "raster_copy"
            else:
                meta["raster_path"] = str(src.resolve())
                meta["storage"] = "raster_reference"

        elif isinstance(self._data, np.ndarray):
            if not copy_raster:
                raise ValueError(
                    "Para persistir un ndarray usa copy_raster=True."
                )
            np.save(directory / "population_array.npy", self._data)
            meta["storage"] = "ndarray"

        else:
            raise TypeError(f"Tipo de data no persistible: {type(self._data)}")

        _write_manifest(directory, "PopulationLayer", meta)
        return directory

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PopulationLayer":
        """
        Carga una PopulationLayer desde disco.

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
            raise ValueError(f"storage desconocido: {storage}")

        return obj

    def __repr__(self):
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
    Orquestador de análisis de accesibilidad urbana.

    Parameters
    ----------
    network : StreetNetwork
    pois : PointsOfInterest
    distance_matrix : pd.DataFrame | list
        Matriz distancia-calidad-accesibilidad o lista de distancias simples.
    poi_quality_column : str, optional
    min_edge_length : float, optional
    max_dist : float, optional

    Persistencia
    ------------
    Guarda todos los componentes en sub-directorios independientes, lo que
    permite al backend cargar solo la parte que necesita (p.ej. solo el
    resultado sin volver a cargar la red original).

    Estructura:
        <path>/
            manifest.json       → metadatos + distance_matrix serializada
            network/            → StreetNetwork de entrada
            pois/               → PointsOfInterest
            result_network/     → StreetNetwork resultado (si run() fue llamado)

    >>> analyzer.save("data/berlin_analysis")
    >>> analyzer.save("data/berlin_analysis", fmt="fgb")
    >>> analyzer = AccessibilityAnalyzer.load("data/berlin_analysis")

    # Carga solo el resultado sin cargar la red original:
    >>> result = StreetNetwork.load("data/berlin_analysis/result_network")
    """

    def __init__(
        self,
        network: StreetNetwork,
        pois: PointsOfInterest,
        distance_matrix,
        poi_quality_column=None,
        min_edge_length=0.0,
        max_dist=None,
    ):
        self.network = network
        self.pois = pois
        self.distance_matrix = distance_matrix
        self.poi_quality_column = poi_quality_column or pois.quality_column
        self.min_edge_length = min_edge_length
        self.max_dist = max_dist
        self._result_network: Optional[StreetNetwork] = None
        self._points_with_osmid: Optional[gpd.GeoDataFrame] = None

    # ── Ejecución ─────────────────────────────────────────────────────────────

    def run(self, verbose=True) -> StreetNetwork:
        """Calcula isocronas de accesibilidad sobre la red."""
        pts = self.pois.data.copy()
        result = isochrones.graph(
            self.network.graph, pts,
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

    def run_buffers(self, accessibility_values=None, verbose=True) -> gpd.GeoDataFrame:
        """Calcula isocronas como buffers euclídeos (sin red de calles)."""
        return isochrones.buffers(
            service_geoms=self.pois.data,
            distance_matrix=self.distance_matrix,
            accessibility_values=accessibility_values,
            poi_quality_col=self.poi_quality_column,
            verbose=verbose,
        )

    def default_distance_matrix(self) -> pd.DataFrame:
        if self.poi_quality_column is None:
            raise ValueError("Se necesita quality_column en los POIs.")
        dm, _ = isochrones.default_distance_matrix(
            self.pois.data,
            distance_steps=(
                self.distance_matrix if isinstance(self.distance_matrix, list)
                else list(self.distance_matrix.columns[:-1])
            ),
            poi_quality_column=self.poi_quality_column,
        )
        return dm

    # ── Resultados ────────────────────────────────────────────────────────────

    @property
    def result_network(self) -> Optional[StreetNetwork]:
        return self._result_network

    @property
    def result_nodes(self) -> Optional[gpd.GeoDataFrame]:
        return self._result_network.nodes if self._result_network else None

    @property
    def result_edges(self) -> Optional[gpd.GeoDataFrame]:
        return self._result_network.edges if self._result_network else None

    def to_h3(self, resolution, column="accessibility", method="max") -> H3Grid:
        if self._result_network is None:
            raise RuntimeError("Ejecuta run() antes de llamar a to_h3().")
        edges = self._result_network.edges
        if column not in edges.columns:
            raise ValueError(f"Columna '{column}' no encontrada en las aristas.")
        df = h3_utils.from_gdf(
            edges[[column, "geometry"]], resolution=resolution,
            columns=[column], method=method,
        )
        return H3Grid(data=df, resolution=resolution)

    def plot(self, column="accessibility", cmap="RdYlGn", aoi=None,
             show_pois=True, **kwargs):
        if self._result_network is None:
            raise RuntimeError("Ejecuta run() antes de plotear.")
        aoi_gdf = aoi.data if isinstance(aoi, AreaOfInterest) else aoi
        return plot_helpers.general_map(
            aoi=aoi_gdf, gdfs=[self._result_network.edges],
            column=column, cmap=cmap,
            pois=[self.pois.data] if show_pois else [],
            **kwargs,
        )

    # ── Persistencia ──────────────────────────────────────────────────────────

    def save(
        self,
        path: Union[str, Path],
        fmt: GeoFmt = "geoparquet",
        overwrite: bool = True,
    ) -> Path:
        """
        Guarda el análisis completo en disco.

        Cada componente se guarda en su propio sub-directorio, permitiendo
        carga independiente desde el backend sin cargar el análisis entero.

        Parameters
        ----------
        path : str | Path
            Directorio raíz de salida.
        fmt : GeoFmt
            Formato aplicado a todos los componentes geodata.
        overwrite : bool
            Sobreescribir si existe.

        Examples
        --------
        >>> analyzer.save("data/berlin_analysis")
        >>> analyzer.save("data/berlin_analysis", fmt="fgb")

        # Carga parcial desde backend:
        >>> result = StreetNetwork.load("data/berlin_analysis/result_network")
        """
        directory = _ensure_dir(path, overwrite)

        self.network.save(directory / "network", fmt=fmt, overwrite=True)
        self.pois.save(directory / "pois", fmt=fmt, overwrite=True)

        has_result = self._result_network is not None
        if has_result:
            self._result_network.save(directory / "result_network", fmt=fmt, overwrite=True)

        # Serializar distance_matrix
        if isinstance(self.distance_matrix, list):
            dm_serial = {"type": "list", "values": self.distance_matrix}
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
        Carga un AccessibilityAnalyzer desde disco.

        Parameters
        ----------
        path : str | Path
            Directorio generado por ``save()``.

        Examples
        --------
        >>> analyzer = AccessibilityAnalyzer.load("data/berlin_analysis")
        """
        directory = Path(path)
        manifest = _read_manifest(directory)
        _check_class(manifest, "AccessibilityAnalyzer")

        network = StreetNetwork.load(directory / "network")
        pois = PointsOfInterest.load(directory / "pois")

        dm_serial = manifest.get("distance_matrix", {})
        if dm_serial.get("type") == "list":
            distance_matrix = dm_serial["values"]
        elif dm_serial.get("type") == "dataframe":
            distance_matrix = pd.DataFrame(dm_serial["data"])
        else:
            distance_matrix = dm_serial.get("values")

        obj = cls.__new__(cls)
        obj.network = network
        obj.pois = pois
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

    def __repr__(self):
        return (
            f"AccessibilityAnalyzer("
            f"pois={len(self.pois)}, "
            f"network_type='{self.network.network_type}', "
            f"ran={self._result_network is not None})"
        )