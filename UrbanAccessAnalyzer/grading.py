"""Grade street accessibility to several types of points of interest (POIs).

Each POI type is described by a small config dict::

    {
        "name": "groceries",           # label, also the output column name
        "query": osm_io.groceries,     # callable(bounds) -> POI table, see osm_io
        "geometry": "Point",           # "Point" or "Polygon" (e.g. parks)
        "importance": 8,               # relative weight in the final score
        "max_distance": 1500,          # meters walked beyond which the POI is worthless
        "icon": "🛒",                  # optional emoji shown on the maps
    }

``query`` can be an OSM query (:mod:`UrbanAccessAnalyzer.osm_io`,
:func:`UrbanAccessAnalyzer.osm_io.tag_query`) or your own file
(:func:`query_from_file`).

Grading, per POI type:

1. Distances ``max_distance * j / n_steps`` (``j = 1..n_steps``) are the
   walk-distance tiers; the score of a street node is ``100 * (n_steps - j + 1) /
   n_steps`` for the closest tier it falls in (so ``100`` right next to a POI
   and ``100 / n_steps`` at ``max_distance``), and ``0`` beyond ``max_distance``.
2. ``Point`` POIs are snapped onto the street network. ``Polygon`` POIs
   (parks) enter the network wherever their boundary touches a street, and
   every node inside the polygon scores ``100``.

The final score of a node is the importance-weighted mean of its per-type
scores; an edge takes the mean of its two end nodes.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import shapely

from . import isochrones, scoring
from .api import PointsOfInterest, StreetNetwork

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


def fetch_pois(
    specs: Sequence[dict],
    bounds: gpd.GeoDataFrame,
    crs,
    max_retries: int = 3,
    retry_wait: float = 5.0,
) -> dict[str, gpd.GeoDataFrame]:
    """Run every config's ``query`` over ``bounds`` and return one GeoDataFrame per POI type.

    Args:
        specs: POI configs (see module docstring).
        bounds: Download area.
        crs: CRS to project the results to (normally ``network.crs``).
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
        gdf = PointsOfInterest(df, crs="EPSG:4326").to_gdf().to_crs(crs)
        gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].reset_index(drop=True)
        if s["geometry"] == "Point":
            gdf["geometry"] = gdf.geometry.centroid
        out[s["name"]] = gdf
        print(f"[{s['name']}] {len(gdf)} features")
    return out


def grade(
    network: StreetNetwork,
    specs: Sequence[dict],
    pois: dict[str, gpd.GeoDataFrame],
    n_steps: int = 5,
    max_snap_dist: Optional[float] = 150.0,
    min_edge_length: float = 30.0,
    score_col: str = "access_score",
    aoi: Optional[gpd.GeoDataFrame] = None,
) -> tuple[gpd.GeoDataFrame, StreetNetwork]:
    """Grade every street edge by its walk distance to each POI type, then combine.

    Args:
        network: Street network (projected CRS, meters).
        specs: POI configs (see module docstring).
        pois: Output of :func:`fetch_pois`, in ``network.crs``.
        n_steps: Number of distance tiers per POI type.
        max_snap_dist: Max distance (m) from a POI to the nearest street.
        min_edge_length: Minimum edge length (m) created when snapping POIs.
        score_col: Name of the combined ``0-100`` score column.
        aoi: If given, only the edges inside this area are returned.

    Returns:
        ``(edges_gdf, snapped_network)``. ``edges_gdf`` has one row per street
        edge with one ``0-100`` column per POI type (named ``name``) plus the
        importance-weighted ``score_col``.
    """
    _check_specs(specs)
    street_edges = network.to_gdf()

    # 1. Access points: the point itself, or where a polygon touches a street.
    parts = []
    for s in specs:
        gdf = pois[s["name"]]
        if gdf.empty:
            continue
        if s["geometry"] == "Polygon":
            gdf = scoring.polygons_to_points(gdf, street_edges)
        geoms = gdf.geometry.centroid if (gdf.geom_type != "Point").any() else gdf.geometry
        parts.append(pd.DataFrame({"geometry": shapely.to_wkb(geoms.values), "poi_type": s["name"]}))
    if not parts:
        raise ValueError("No POIs were found for any type; nothing to grade.")
    entry = pl.from_pandas(pd.concat(parts)).with_columns(
        pl.col("geometry").cast(pl.Binary), pl.lit(1.0).alias("poi_score")
    )
    network, points = network.snap_points(
        PointsOfInterest(entry, crs=network.crs), max_dist=max_snap_dist, min_edge_length=min_edge_length
    )
    nodes, edges = network.nodes, network.edges

    # 2. Per-type node score from the distance tiers.
    node_scores = nodes.select("node_id")
    for s in specs:
        name = s["name"]
        pts = points.df.filter(pl.col("poi_type") == name).drop_nulls("node_id")
        if pts.height == 0:
            node_scores = node_scores.with_columns(pl.lit(0.0).alias(name))
            continue
        steps = [s["max_distance"] * (j + 1) / n_steps for j in range(n_steps)]
        levels = [100.0 * (n_steps - j) / n_steps for j in range(n_steps)]
        order = isochrones.distance_matrix_to_processing_order(steps, levels)
        reached, _ = isochrones.compute_node_access(nodes, edges, pts, order, "poi_score", True, False)
        node_scores = node_scores.join(
            reached.select("node_id", pl.col("access_score").alias(name)), on="node_id", how="left"
        )
        if s["geometry"] == "Polygon":
            inside = shapely.contains_xy(pois[name].union_all(), nodes["x"].to_numpy(), nodes["y"].to_numpy())
            node_scores = node_scores.with_columns(
                pl.when(pl.Series(inside)).then(100.0).otherwise(pl.col(name)).alias(name)
            )
        node_scores = node_scores.with_columns(pl.col(name).fill_null(0.0))

    # 3. Importance-weighted mean per node, then per edge (mean of its end nodes).
    total_importance = float(sum(s["importance"] for s in specs))
    combined = sum(pl.col(s["name"]) * s["importance"] for s in specs) / total_importance
    node_scores = node_scores.with_columns(combined.alias(score_col))

    cols = [s["name"] for s in specs] + [score_col]
    u_scores = node_scores.rename({c: f"u_{c}" for c in cols}).rename({"node_id": "u"})
    v_scores = node_scores.rename({c: f"v_{c}" for c in cols}).rename({"node_id": "v"})
    edges = (
        edges.join(u_scores, on="u", how="left")
        .join(v_scores, on="v", how="left")
        .with_columns([((pl.col(f"u_{c}") + pl.col(f"v_{c}")) / 2).alias(c) for c in cols])
        .drop([f"u_{c}" for c in cols] + [f"v_{c}" for c in cols])
    )
    geom = shapely.from_wkb(edges["geometry_wkb"].to_numpy())
    edges_gdf = gpd.GeoDataFrame(edges.drop("geometry_wkb").to_pandas(), geometry=geom, crs=network.crs)
    if aoi is not None:  # keep the study area only
        edges_gdf = edges_gdf[edges_gdf.intersects(aoi.to_crs(network.crs).union_all())].reset_index(drop=True)
    return edges_gdf, network


def poi_table(specs: Sequence[dict], pois: dict[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """One EPSG:4326 table of all POIs for mapping, with columns ``type``, ``name``, ``icon``.

    ``icon`` is the config's optional ``"icon"`` emoji (default ``"📍"``), which
    :func:`UrbanAccessAnalyzer.plotting.general_map` draws on each POI
    (``poi_column="type", poi_icon_column="icon"``).
    """
    parts = []
    for s in specs:
        gdf = pois[s["name"]]
        if gdf.empty:
            continue
        name_cols = [c for c in gdf.columns if c == "name" or c.startswith("name:")]
        names = gdf[name_cols].bfill(axis=1).iloc[:, 0] if name_cols else pd.Series("", index=gdf.index)
        parts.append(
            gpd.GeoDataFrame(
                {"type": s["name"], "name": names.fillna("").astype(str).to_numpy(), "icon": s.get("icon", "📍")},
                geometry=gdf.geometry.to_crs(4326).to_numpy(),
                crs=4326,
            )
        )
    if not parts:
        return gpd.GeoDataFrame({"type": [], "name": [], "icon": []}, geometry=[], crs=4326)
    return pd.concat(parts, ignore_index=True)


def tiers_table(specs: Sequence[dict], n_steps: int = 5) -> pd.DataFrame:
    """How each type of POI is scored: for every distance step, the score of the streets inside it.

    A street within the first step of a POI scores 100; each next step scores 100/``n_steps`` less, up to
    ``max_distance``; beyond it the score is 0. ``weight %`` is the ``importance`` share in the final score.

    Returns:
        One row per POI type and step: ``type``, ``icon``, ``within (m)``, ``score`` and ``weight %``.
    """
    _check_specs(specs)
    total = float(sum(s["importance"] for s in specs))
    rows = []
    for s in specs:
        for j in range(1, n_steps + 1):
            rows.append({
                "type": s["name"], "icon": s.get("icon", "📍"),
                "within (m)": round(s["max_distance"] * j / n_steps),
                "score": round(100 * (n_steps - j + 1) / n_steps),
                "weight %": round(100 * s["importance"] / total, 1),
            })
    return pd.DataFrame(rows)


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
    """One type of place for :func:`fetch_pois` / :func:`grade`, from almost any source.

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


def score_summary(edges: gpd.GeoDataFrame, specs: Sequence[dict], score_col: str = "access_score") -> pd.DataFrame:
    """Summary statistics (0-100) of the score of every type of place and of the final score."""
    return edges[[s["name"] for s in specs] + [score_col]].describe().round(1)
