"""Places (points of interest) from OpenStreetMap or your own files, as GeoDataFrames.

A *source* can be a ready-made OpenStreetMap kind (``"schools"``), an OSM tag ``("amenity", "dentist")``, an
Overpass filter, a file of your own, or your own function. Typical use::

    from UrbanAccessAnalyzer import osm

    schools = osm.fetch("schools", area)              # a GeoDataFrame with the schools
    pois = [osm.poi("schools", "schools", importance=10), osm.poi("parks", "green_areas")]
    gdfs = osm.fetch_all(pois, area)                  # {"schools": GeoDataFrame, "parks": GeoDataFrame}

Scoring the places is in :mod:`UrbanAccessAnalyzer.grading`; the OpenStreetMap queries are in
:mod:`UrbanAccessAnalyzer.osm_io`.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

import geopandas as gpd
import pandas as pd
import polars as pl
import shapely

from .api import PointsOfInterest

REQUIRED_KEYS = ("name", "query", "geometry", "importance", "max_distance")


def _check_specs(specs: Sequence[dict]) -> None:
    names = [s.get("name") for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"POI names must be unique, got {names}")
    for s in specs:
        missing = [k for k in REQUIRED_KEYS if k not in s]
        if missing:
            raise ValueError(f"POI config {s.get('name')!r} is missing {missing}")
        if s["geometry"] not in ("Point", "Polygon"):
            raise ValueError(f"POI {s['name']!r}: geometry must be 'Point' or 'Polygon', got {s['geometry']!r}")


OSM_KINDS = (
    "schools", "healthcare", "groceries", "shops", "restaurants", "libraries",
    "pharmacies", "gyms", "cinemas", "bus_stops", "green_areas",
)


def poi(
    name: str,
    source,
    geometry: Optional[str] = None,
    importance: float = 5,
    max_distance: float = 1000,
    icon: str = "📍",
    name_column: Optional[str] = None,
) -> dict:
    """One type of place for :func:`fetch_all` / :func:`grade`, from almost any source.

    ``source`` can be:

    - a ready-made OSM kind: ``"schools"``, ``"groceries"``, ``"green_areas"``... (see :data:`OSM_KINDS`)
    - any OSM tag, as ``(key, value)``: ``("amenity", "dentist")`` (``"*"`` = any value)
    - your own file of points or polygons (``.gpkg``, ``.geojson``, ``.shp``, ``.parquet``..., any CRS):
      ``"my_places.gpkg"``. In Google Colab, if the file is not there, a window opens to choose it from
      your computer.
    - an **Overpass filter**, for several tags or conditions: ``'nwr["amenity"~"hospital|clinic"]'`` (see
      :func:`UrbanAccessAnalyzer.osm_io.overpass`)
    - your own function ``query(bounds) -> table with a WKB geometry column``.

    Args:
        name: Label of the type (also the output column and the map legend).
        source: See above.
        geometry: ``"Point"`` or ``"Polygon"`` (areas such as parks). Default: ``"Polygon"`` for
            ``"green_areas"``, otherwise ``"Point"``.
        importance: Weight of this type in the final score (relative to the others).
        max_distance: Meters walked beyond which the place is worthless.
        icon: Emoji shown on the maps.
        name_column: For your own file: the column with the names of the places.

    Returns:
        The config dict (``name``, ``query``, ``geometry``, ``importance``, ``max_distance``, ``icon``).
    """
    from . import osm_io

    if callable(source):
        query = source
    elif isinstance(source, (tuple, list)) and len(source) == 2:
        query = osm_io.tag_query(*source)
    elif isinstance(source, str) and source in OSM_KINDS:
        query = getattr(osm_io, source)
    elif isinstance(source, str) and (source.lstrip().startswith(("nwr", "node", "way", "relation", "[out:")) or "{{bbox}}" in source):
        query = osm_io.overpass(source)
    elif isinstance(source, str):
        query = query_from_file(source, name_column=name_column)
    else:
        raise TypeError(f"source must be a kind, a (key, value) OSM tag, an Overpass filter, a file path or a function; got {source!r}")
    if geometry is None:
        geometry = "Polygon" if source == "green_areas" else "Point"
    return {"name": name, "query": query, "geometry": geometry, "importance": importance,
            "max_distance": max_distance, "icon": icon}


def query_from_file(path: str = "", name_column: Optional[str] = None) -> Callable:
    """Build a POI ``query`` from your own vector file (points or polygons).

    Any format GeoPandas reads works: GeoPackage, GeoJSON, Shapefile, GeoParquet,
    FlatGeobuf, ... in any CRS. Only the features inside the download area are kept.

    Args:
        path: Path to the file. In Google Colab, leave it empty (or wrong) and the upload dialog
            opens automatically to pick the file from your computer (see
            :func:`UrbanAccessAnalyzer.utils.select_file`).
        name_column: Column with the POI names (shown on the maps), if any.

    Returns:
        A function ``query(bounds)`` to use as the ``"query"`` of a POI config.
    """
    from . import geometry_ops, utils

    path = utils.select_file(path)

    def _query(bounds: gpd.GeoDataFrame) -> pl.DataFrame:
        gdf = geometry_ops.read_geofile(path, bounds=bounds)
        if gdf.crs is None:
            raise ValueError(f"{path} has no CRS; set one first, e.g. gdf.set_crs(4326).to_file(...)")
        gdf = gdf.to_crs(4326)
        gdf = gdf[gdf.geometry.intersects(bounds.to_crs(4326).union_all())]
        df = pl.DataFrame({"geometry": shapely.to_wkb(gdf.geometry.to_numpy())}).with_columns(
            pl.col("geometry").cast(pl.Binary)
        )
        if name_column:
            df = df.with_columns(pl.Series("name", gdf[name_column].astype(str).to_numpy()))
        return df

    _query.__name__ = "file_" + str(path).replace("/", "_")
    return _query


def fetch_all(
    specs: Sequence[dict],
    bounds: gpd.GeoDataFrame,
    crs=None,
    max_retries: int = 3,
    retry_wait: float = 5.0,
) -> dict[str, gpd.GeoDataFrame]:
    """Run every config's ``query`` over ``bounds`` and return one GeoDataFrame per POI type.

    Args:
        specs: POI configs (see module docstring).
        bounds: Download area.
        crs: CRS to project the results to (normally ``network.crs``); ``None`` keeps EPSG:4326.
        max_retries: Attempts per query (public Overpass servers are often busy).
        retry_wait: Seconds between attempts.

    Returns:
        ``{name: GeoDataFrame}``. ``Point`` configs are collapsed to one point
        per feature; ``Polygon`` configs keep their polygons. A type whose
        query fails or returns nothing maps to an empty GeoDataFrame.
    """
    _check_specs(specs)
    out: dict[str, gpd.GeoDataFrame] = {}
    for s in specs:
        df = None
        for attempt in range(1, max_retries + 1):
            try:
                df = s["query"](bounds)
                break
            except Exception as e:  # noqa: BLE001 - network errors of any kind
                print(f"[{s['name']}] attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(retry_wait)
        if df is None or df.height == 0:
            print(f"[{s['name']}] no POIs found")
            out[s["name"]] = gpd.GeoDataFrame(geometry=[], crs=crs)
            continue
        gdf = PointsOfInterest(df, crs="EPSG:4326").to_gdf()
        if crs is not None:
            gdf = gdf.to_crs(crs)
        gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].reset_index(drop=True)
        if s["geometry"] == "Point":
            utm = gdf.estimate_utm_crs() if gdf.crs.is_geographic and len(gdf) else None
            gdf["geometry"] = (gdf.to_crs(utm).geometry.centroid.to_crs(gdf.crs) if utm is not None else gdf.geometry.centroid)
        out[s["name"]] = gdf
        print(f"[{s['name']}] {len(gdf)} features")
    return out


def pins(gdf: gpd.GeoDataFrame, type: str = "places", icon: str = "📍") -> gpd.GeoDataFrame:
    """The places of one type as a table for the maps: EPSG:4326 with ``type``, ``name`` and ``icon`` columns.

    Args:
        gdf: The places (any CRS); names come from its ``name`` column if it has one.
        type: Label of the type (the legend of the map).
        icon: Emoji drawn on every place.
    """
    if gdf.empty:
        return gpd.GeoDataFrame({"type": [], "name": [], "icon": []}, geometry=[], crs=4326)
    name_cols = [c for c in gdf.columns if c == "name" or c.startswith("name:")]
    names = gdf[name_cols].bfill(axis=1).iloc[:, 0] if name_cols else pd.Series("", index=gdf.index)
    return gpd.GeoDataFrame(
        {"type": type, "name": names.fillna("").astype(str).to_numpy(), "icon": icon},
        geometry=gdf.geometry.to_crs(4326).to_numpy(),
        crs=4326,
    )


def table(specs: Sequence[dict], pois: dict[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """One EPSG:4326 table of all the places of every config, for mapping (see :func:`pins`).

    ``icon`` is the config's optional ``"icon"`` emoji (default ``"📍"``), which
    :func:`UrbanAccessAnalyzer.plotting.general_map` draws on each place
    (``poi_column="type", poi_icon_column="icon"``).
    """
    parts = [pins(pois[s["name"]], s["name"], s.get("icon", "📍")) for s in specs if not pois[s["name"]].empty]
    if not parts:
        return pins(gpd.GeoDataFrame(geometry=[], crs=4326))
    return pd.concat(parts, ignore_index=True)


def fetch(
    source,
    area: gpd.GeoDataFrame,
    crs=None,
    geometry: str = "Point",
    name_column: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """The places of one source inside ``area``, as a GeoDataFrame.

    Example: ``schools = osm.fetch("schools", area)``.

    Args:
        source: A ready-made OSM kind (``"schools"``), an OSM tag ``("amenity", "dentist")``, an Overpass filter,
            a file of your own (in Google Colab a window opens to choose it if it is not there) or a function.
            See :func:`poi`.
        area: Area to search (a GeoDataFrame, any CRS).
        crs: CRS of the result (default EPSG:4326).
        geometry: ``"Point"`` (default: polygons become one point each) or ``"Polygon"`` (keep the polygons).
        name_column: For your own file: the column with the names.

    Returns:
        GeoDataFrame with the places (and their OSM tags as columns).
    """
    config = poi("places", source, geometry=geometry, name_column=name_column)
    return fetch_all([config], area, crs)["places"]
