"""OpenStreetMap data ingestion, entirely into Polars.

This module is the single entry point for getting OpenStreetMap data into the
package as Polars DataFrames. It replaces the previous osmnx/networkx +
``osmium`` CLI pipeline with two independent paths:

1. **Street networks** (:func:`load_pbf`): a ``.osm.pbf`` extract is parsed
   directly by `DuckDB's spatial extension
   <https://duckdb.org/docs/extensions/spatial/functions.html#st_readosm>`_
   (``ST_ReadOSM``), which streams the protobuf file straight into Arrow
   record batches. Polars reads those batches zero-copy
   (``duckdb_relation.pl()``), so no pandas, geopandas, osmnx or networkx
   object is ever created. Nodes and ways are resolved into a directed edge
   table (``u, v, length_m, highway, oneway, geometry_wkb``) using vectorised
   Polars list/struct expressions (tag extraction, way-to-consecutive-node-pair
   explosion) instead of a Python loop over ways. Edge length is computed with
   the haversine great-circle formula so no UTM (re)projection is needed just
   to route.

2. **Points of interest** (:func:`overpass_query` and the named convenience
   wrappers below it, e.g. :func:`schools`): POI payloads from Overpass are
   small (hundreds to low thousands of features), so a full DuckDB pipeline
   buys nothing there; these keep the existing raw-HTTP-to-Overpass +
   ``osm2geojson`` path, but the result is converted to a Polars DataFrame
   with a WKB ``geometry`` column immediately, instead of a GeoDataFrame.

Sources / methodology:

- OSM PBF structure and tag semantics: https://wiki.openstreetmap.org/wiki/PBF_Format
- Highway tag classification for walk/bike/drive network profiles follows the
  same tag sets osmnx and the previous ``osmium`` filter used:
  https://wiki.openstreetmap.org/wiki/Key:highway
- Great-circle (haversine) distance:
  ``d = 2r * asin(sqrt(sin²(Δφ/2) + cos(φ1)cos(φ2)sin²(Δλ/2)))``, r = 6371008.8 m
  (mean Earth radius, WGS84 authalic radius approximation).
- Geofabrik region index: https://download.geofabrik.de/index-v1.json
- Overpass API: https://wiki.openstreetmap.org/wiki/Overpass_API
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Optional

import duckdb
import geopandas as gpd
import numpy as np
import polars as pl
import requests
import shapely
from osm2geojson import json2geojson

from . import utils

EARTH_RADIUS_M = 6_371_008.8

WALK_HIGHWAYS = {
    "footway", "pedestrian", "path", "living_street", "steps",
    "residential", "service", "unclassified", "track",
    # Pedestrians can legally walk along (usually via a sidewalk/shoulder) every
    # vehicle-carrying road type except motorways/motorway_links -- the previous
    # narrower set silently dropped primary/secondary/tertiary/trunk roads from
    # the walk network, which fragments pedestrian routing anywhere those are
    # the only road serving a block (a real issue in car-oriented street grids
    # like Guadalajara's, where sidewalks along arterials are the only walkable
    # connection between residential clusters).
    "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link",
}
BIKE_HIGHWAYS = {
    "cycleway", "path", "residential", "living_street",
    "unclassified", "service", "track",
}
DRIVE_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary",
    "primary_link", "secondary", "secondary_link", "tertiary",
    "tertiary_link", "residential", "unclassified", "service", "living_street",
}
PRIMARY_HIGHWAYS = {
    "trunk", "trunk_link", "primary", "primary_link", "secondary",
    "secondary_link", "tertiary", "tertiary_link", "residential",
    "unclassified", "service", "living_street",
}

NETWORK_PROFILES = {
    "walk": {"highways": WALK_HIGHWAYS, "extra_tags": {"foot": {"yes"}}},
    "bike": {"highways": BIKE_HIGHWAYS, "extra_tags": {"bicycle": {"yes"}}},
    "drive": {"highways": DRIVE_HIGHWAYS, "extra_tags": {}},
    "all": {"highways": None, "extra_tags": {}},
    "walk+bike": {
        "highways": WALK_HIGHWAYS | BIKE_HIGHWAYS,
        "extra_tags": {"foot": {"yes"}, "bicycle": {"yes"}},
    },
    "walk+bike+primary": {
        "highways": WALK_HIGHWAYS | BIKE_HIGHWAYS | PRIMARY_HIGHWAYS,
        "extra_tags": {"foot": {"yes"}, "bicycle": {"yes"}},
    },
}


def _write_poly_file(aoi: gpd.GeoDataFrame | gpd.GeoSeries, poly_path: str) -> None:
    """Write AOI geometry to a .poly file in Osmosis format.

    Handles Polygons, MultiPolygons, and interior rings (holes). Kept for
    compatibility with Geofabrik/osmium-style clipping workflows; the
    duckdb-based loader in this module clips by bounding box + point-in-polygon
    instead and does not need this file.

    Args:
        aoi: AOI geometry.
        poly_path: Destination path for the .poly file.
    """
    geom = aoi.to_crs(4326).union_all()
    with open(poly_path, "w") as f:
        f.write("aoi\n")

        def write_ring(coords, ring_id, is_hole=False):
            prefix = "!" if is_hole else ""
            f.write(f"{prefix}{ring_id}\n")
            for x, y in coords:
                f.write(f"  {x:.7f} {y:.7f}\n")
            f.write("END\n")

        ring_counter = 1
        polygons = []
        if geom.geom_type == "Polygon":
            polygons.append(geom)
        elif geom.geom_type == "MultiPolygon":
            polygons.extend(geom.geoms)
        else:
            raise ValueError(f"Unsupported geometry type for .poly file: {geom.geom_type}")

        for poly in polygons:
            write_ring(poly.exterior.coords, ring_counter)
            ring_counter += 1
            for interior in poly.interiors:
                write_ring(interior.coords, ring_counter, is_hole=True)
                ring_counter += 1
        f.write("END\n")


def download_geofabrik(aoi: gpd.GeoDataFrame | gpd.GeoSeries, output_folder: Optional[str] = None) -> str:
    """Download the smallest Geofabrik region that fully contains an AOI.

    Args:
        aoi: AOI geometry, in any CRS.
        output_folder: Folder to save the downloaded ``.osm.pbf`` into. If
            ``None``, saves to the current working directory.

    Returns:
        Path to the downloaded (or already-cached) ``.osm.pbf`` file.

    Raises:
        ValueError: If no Geofabrik region contains the AOI.
    """
    aoi = aoi.to_crs(4326)
    aoi_geom = aoi.union_all()
    if not aoi_geom.is_valid:
        print("Validity problem:", shapely.validation.explain_validity(aoi_geom))

    url = "https://download.geofabrik.de/index-v1.json"
    print(f"Fetching Geofabrik index from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    candidate_regions = []
    for feature in data["features"]:
        properties = feature.get("properties", {})
        if not properties.get("urls", {}).get("pbf") or not feature.get("geometry"):
            continue
        try:
            region_geom = shapely.geometry.shape(feature["geometry"])
        except Exception as e:
            print(f"Warning: could not process geometry for {properties.get('name', 'N/A')}: {e}")
            continue
        if region_geom.contains(aoi_geom):
            candidate_regions.append((region_geom.area, properties))

    if not candidate_regions:
        raise ValueError("No Geofabrik region was found to contain the AOI.")

    candidate_regions.sort(key=lambda x: x[0])
    best_region = candidate_regions[0][1]
    pbf_url = best_region.get("urls", {}).get("pbf")

    safe_name = utils.sanitize_filename(best_region.get("name", "region"))
    filename = f"{safe_name}.osm.pbf"
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, filename)
    else:
        output_file = filename

    if os.path.exists(output_file):
        print(f"File '{output_file}' already exists. Skipping download.")
        return output_file

    print(f"Downloading '{best_region['name']}' from {pbf_url} ...")
    pbf_response = requests.get(pbf_url, stream=True)
    pbf_response.raise_for_status()
    with open(output_file, "wb") as f:
        for chunk in pbf_response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded geofabrik to {output_file}")
    return output_file


def _duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with the spatial extension loaded."""
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    return con


def _tag_value(column: str, key: str) -> pl.Expr:
    """Polars expression extracting a single OSM tag value from a tags-list column.

    Args:
        column: Name of the ``list[struct[key, value]]`` column produced by
            ``ST_ReadOSM``.
        key: OSM tag key to extract (e.g. ``"highway"``).

    Returns:
        A ``pl.Expr`` yielding the tag's string value, or ``null`` if absent.
    """
    return (
        pl.col(column)
        .list.eval(pl.element().filter(pl.element().struct.field("key") == key).struct.field("value"))
        .list.first()
    )


def load_pbf(
    pbf_path: str,
    aoi: Optional[gpd.GeoDataFrame | gpd.GeoSeries] = None,
    network_type: str = "walk",
    bbox_margin_deg: float = 0.02,
    ignore_oneway: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load a street network from a ``.osm.pbf`` file directly into Polars.

    Uses DuckDB's ``ST_ReadOSM`` table function to parse the PBF into Arrow,
    then resolves ways into a directed edge table entirely with Polars
    expressions (no per-way Python loop, no networkx/osmnx/geopandas).

    Args:
        pbf_path: Path to a local ``.osm.pbf`` file (e.g. from
            :func:`download_geofabrik`).
        aoi: Optional AOI used to bbox-prefilter nodes before resolving ways,
            for speed on large regional extracts. If ``None``, the whole file
            is loaded.
        network_type: One of ``"walk"``, ``"bike"``, ``"drive"``, ``"all"``,
            ``"walk+bike"``, ``"walk+bike+primary"`` (see
            :data:`NETWORK_PROFILES`).
        bbox_margin_deg: Degrees of padding added around the AOI bounding box
            before filtering nodes, so edges that straddle the AOI boundary
            aren't dropped prematurely (final precise cropping belongs to
            ``graph_ops.crop_by_aoi``).
        ignore_oneway: If True, every kept way is emitted as a directed edge
            in BOTH directions regardless of its OSM ``oneway`` tag -- for
            walking-access graphs, where a pedestrian can generally walk
            against traffic direction, a vehicle-only ``oneway=yes`` should
            not restrict traversal. The returned ``oneway`` column still
            reports the original OSM tag value either way; only which
            direction(s) get an edge row changes. Defaults to False (the
            original OSM-faithful behavior), which existing drive/bike
            callers rely on.

    Returns:
        A tuple ``(nodes, edges)``:

        - ``nodes``: columns ``node_id (i64), lon (f64), lat (f64)``, one row
          per node referenced by a kept edge.
        - ``edges``: columns ``u (i64), v (i64), length_m (f64),
          highway (str), oneway (bool), geometry_wkb (binary), edge_uid
          (i64)``, one directed row per traversable direction (two rows per
          two-way segment sharing the same ``edge_uid``, one row for a
          ``oneway=yes`` segment). ``edge_uid`` lets downstream code (e.g.
          ``graph_ops.simplify``) recognize the forward/backward rows of the
          same original way segment as one edge.

    Raises:
        ValueError: If ``network_type`` is not a known profile.
    """
    if network_type not in NETWORK_PROFILES:
        raise ValueError(f"Unknown network_type: {network_type!r}. Choose one of {list(NETWORK_PROFILES)}.")
    profile = NETWORK_PROFILES[network_type]

    con = _duckdb_connection()
    where_bbox = ""
    if aoi is not None:
        minx, miny, maxx, maxy = aoi.to_crs(4326).total_bounds
        where_bbox = (
            f" AND lat BETWEEN {miny - bbox_margin_deg} AND {maxy + bbox_margin_deg}"
            f" AND lon BETWEEN {minx - bbox_margin_deg} AND {maxx + bbox_margin_deg}"
        )

    nodes_all = con.execute(
        f"SELECT id AS node_id, lat, lon FROM ST_ReadOSM('{pbf_path}') WHERE kind = 'node'{where_bbox}"
    ).pl()

    empty_result = (
        pl.DataFrame(schema={"node_id": pl.Int64, "lon": pl.Float64, "lat": pl.Float64}),
        pl.DataFrame(
            schema={
                "u": pl.Int64, "v": pl.Int64, "length_m": pl.Float64,
                "highway": pl.Utf8, "oneway": pl.Boolean, "geometry_wkb": pl.Binary,
                "edge_uid": pl.Int64,
            }
        ),
    )
    if nodes_all.is_empty():
        con.close()
        return empty_result

    # Filter ways *inside* the SQL query rather than pulling every way in the file into
    # Python first: a whole-region/country .osm.pbf's way table includes every building,
    # landuse polygon, etc, not just roads, and can be tens of millions of rows -- loading
    # all of it (with a full tags map + node-ref list per row) before any AOI/road-type
    # filtering is what caused real OOM crashes on large regional extracts. Every profile's
    # final filter below requires a non-null `highway` tag anyway (see `keep_mask &
    # pl.col("highway").is_not_null()`), so requiring it here is not a behavior change --
    # and requiring at least one referenced node to be in the (already bbox-filtered)
    # `nodes_all` set drops every way entirely outside the AOI's bounding box. Also extract
    # only the four tag values actually used (`highway`/`foot`/`bicycle`/`oneway`) instead
    # of materializing the full tags map for every kept way.
    con.register("_nodes_bbox", nodes_all)
    ways = con.execute(
        """
        SELECT
            id AS way_id,
            refs,
            tags['highway'] AS highway,
            tags['foot'] AS foot,
            tags['bicycle'] AS bicycle,
            tags['oneway'] AS oneway_tag
        FROM ST_ReadOSM(?) w
        WHERE kind = 'way' AND refs IS NOT NULL
          AND tags['highway'] IS NOT NULL
          AND EXISTS (
              SELECT 1 FROM UNNEST(w.refs) AS ref(node_id)
              WHERE ref.node_id IN (SELECT node_id FROM _nodes_bbox)
          )
        """,
        [pbf_path],
    ).pl()
    con.close()

    if ways.is_empty():
        return empty_result

    allowed_highways = profile["highways"]
    keep_mask = pl.lit(True) if allowed_highways is None else pl.col("highway").is_in(list(allowed_highways))
    for tag_key, allowed_values in profile["extra_tags"].items():
        keep_mask = keep_mask | pl.col(tag_key).is_in(list(allowed_values))
    ways = ways.filter(keep_mask & pl.col("highway").is_not_null())

    if ways.is_empty():
        return (
            pl.DataFrame(schema={"node_id": pl.Int64, "lon": pl.Float64, "lat": pl.Float64}),
            pl.DataFrame(
                schema={
                    "u": pl.Int64, "v": pl.Int64, "length_m": pl.Float64,
                    "highway": pl.Utf8, "oneway": pl.Boolean, "geometry_wkb": pl.Binary,
                    "edge_uid": pl.Int64,
                }
            ),
        )

    # Explode each way's node-ref list into consecutive (u, v) pairs.
    segments = (
        ways.with_row_index("way_row")
        .explode("refs")
        .rename({"refs": "u"})
        .with_columns(pl.col("u").shift(-1).over("way_row").alias("v"))
        .filter(pl.col("v").is_not_null())
    )

    # Attach endpoint coordinates; drop segments referencing out-of-bbox nodes.
    segments = (
        segments.join(nodes_all.rename({"node_id": "u", "lon": "u_lon", "lat": "u_lat"}), on="u", how="inner")
        .join(nodes_all.rename({"node_id": "v", "lon": "v_lon", "lat": "v_lat"}), on="v", how="inner")
    )

    if segments.is_empty():
        return (
            pl.DataFrame(schema={"node_id": pl.Int64, "lon": pl.Float64, "lat": pl.Float64}),
            pl.DataFrame(
                schema={
                    "u": pl.Int64, "v": pl.Int64, "length_m": pl.Float64,
                    "highway": pl.Utf8, "oneway": pl.Boolean, "geometry_wkb": pl.Binary,
                    "edge_uid": pl.Int64,
                }
            ),
        )

    # Haversine great-circle length in meters, vectorized.
    lat1, lat2 = pl.col("u_lat").radians(), pl.col("v_lat").radians()
    dlat = lat2 - lat1
    dlon = (pl.col("v_lon") - pl.col("u_lon")).radians()
    a = (dlat / 2).sin() ** 2 + lat1.cos() * lat2.cos() * (dlon / 2).sin() ** 2
    length_expr = 2 * EARTH_RADIUS_M * a.sqrt().arcsin()

    # `oneway=yes` is a vehicle-traffic restriction; pedestrians are not bound by it
    # (they can walk along a one-way street's sidewalk/shoulder in either direction).
    # Applying it to the "walk" profile silently made large chunks of the walk network
    # directional -- e.g. any AOI with a one-way arterial grid (again, common in
    # Guadalajara) had walking routes forced the "wrong" way or blocked outright.
    # Only respect `oneway` for motorized profiles (bike/drive/etc) -- and,
    # regardless of profile name, never when the caller explicitly asked for
    # a walking-access graph via `ignore_oneway=True` (e.g. transitlos's
    # `prepare_street_network` now defaults to `network_type="all"` so it
    # includes every OSM street/way type, not just the "walk" profile's
    # whitelist -- it still needs the "walk" profile's oneway-ignoring
    # behavior even though its `network_type` is no longer literally "walk").
    oneway_applies = pl.lit(network_type != "walk" and not ignore_oneway)
    segments = segments.with_columns(
        length_expr.alias("length_m"),
        (oneway_applies & (pl.col("oneway_tag") == "yes")).fill_null(False).alias("is_oneway"),
        pl.int_range(pl.len()).alias("edge_uid"),
    )

    forward = segments.select("u", "v", "length_m", "highway", "u_lon", "u_lat", "v_lon", "v_lat", "is_oneway", "edge_uid")
    backward = (
        segments.filter(~pl.col("is_oneway"))
        .select(
            pl.col("v").alias("u"), pl.col("u").alias("v"), "length_m", "highway",
            pl.col("v_lon").alias("u_lon"), pl.col("v_lat").alias("u_lat"),
            pl.col("u_lon").alias("v_lon"), pl.col("u_lat").alias("v_lat"),
            "is_oneway", "edge_uid",
        )
    )
    edges = pl.concat([forward, backward], how="vertical")

    # Build WKB LineString geometry for every directed edge in one vectorized shapely call.
    coords = edges.select("u_lon", "u_lat", "v_lon", "v_lat").to_numpy()
    lines = shapely.linestrings(coords.reshape(-1, 2, 2))
    wkb = shapely.to_wkb(lines)

    edges = edges.with_columns(
        pl.Series("geometry_wkb", wkb, dtype=pl.Binary),
        pl.col("is_oneway").alias("oneway"),
    ).drop("u_lon", "u_lat", "v_lon", "v_lat", "is_oneway")

    used_node_ids = pl.concat([edges.select(pl.col("u").alias("node_id")), edges.select(pl.col("v").alias("node_id"))]).unique()
    nodes = nodes_all.join(used_node_ids, on="node_id", how="inner")

    return nodes, edges


def overpass_query(query: str, bounds: gpd.GeoDataFrame | gpd.GeoSeries, timeout: int = 120) -> pl.DataFrame:
    """Run a raw Overpass API query and return results as a Polars DataFrame.

    Args:
        query: An Overpass QL query string containing a ``{{bbox}}``
            placeholder (substituted with ``south,west,north,east``).
        bounds: Geometry defining the query bounding box; the result is also
            spatially filtered to intersect this geometry.
        timeout: HTTP timeout in seconds per Overpass mirror attempt.

    Returns:
        A Polars DataFrame with a WKB ``geometry`` column (EPSG:4326) plus one
        column per flattened OSM tag.

    Raises:
        RuntimeError: If every Overpass mirror fails.
    """
    bbox = bounds.to_crs(4326).total_bounds
    bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    query = query.replace("{{bbox}}", bbox_str).replace("[out:xml]", "[out:json]")

    overpass_urls = [
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]

    response_json = None
    for i, url in enumerate(overpass_urls, start=1):
        try:
            response = requests.get(
                url, params={"data": query}, timeout=timeout,
                headers={"User-Agent": "Python Overpass Client"},
            )
            if response.status_code != 200:
                print(f"Warning: server {i}/{len(overpass_urls)} failed ({url}) HTTP {response.status_code}")
                continue
            response_json = response.json()
            if "elements" not in response_json:
                print(f"Warning: server {i}/{len(overpass_urls)} missing 'elements'")
                response_json = None
                continue
            break
        except Exception as e:
            print(f"Warning: server {i}/{len(overpass_urls)} request error ({e})")
            time.sleep(1)
            continue

    if response_json is None:
        raise RuntimeError("All Overpass servers failed.")

    geojson_response = json2geojson(response_json)
    gdf = gpd.GeoDataFrame.from_features(geojson_response, crs="EPSG:4326").reset_index(drop=True)

    if len(gdf) == 0:
        print("Warning: no OSM features found for this query.")
        return _gdf_to_polars(gdf.set_geometry("geometry"))

    if "tags" in gdf.columns:
        tags = gdf["tags"].apply(lambda t: t if isinstance(t, dict) else {})
        tags_df = gpd.pd.json_normalize(tags).rename(columns={"type": "geometry_type"})
        gdf = gpd.pd.concat([gdf.drop(columns=["tags"]), tags_df], axis=1)

    gdf = gdf.loc[:, ~gdf.columns.duplicated()]
    bounds_geom = bounds.to_crs(4326).union_all()
    gdf = gdf[gdf.geometry.intersects(bounds_geom)]

    return _gdf_to_polars(gdf)


def _gdf_to_polars(gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    """Convert a GeoDataFrame to a Polars DataFrame with a WKB ``geometry`` column.

    This is the sanctioned conversion point from the geopandas I/O boundary
    into the package's internal Polars representation.

    Args:
        gdf: Source GeoDataFrame.

    Returns:
        Polars DataFrame with all non-geometry columns preserved and a binary
        ``geometry`` column holding WKB.
    """
    wkb = shapely.to_wkb(gdf.geometry.to_numpy())
    df = pl.from_pandas(gdf.drop(columns=[gdf.geometry.name]))
    return df.with_columns(pl.Series("geometry", wkb, dtype=pl.Binary))


def green_areas(bounds, min_area: float = 200, min_width: float = 10, buffer: float = 5) -> pl.DataFrame:
    """Fetch green-space polygons (parks, gardens, forests, grass) from OSM.

    Applies an erosion-dilation-erosion sequence (shrink by ``min_width``,
    grow back by ``buffer + min_width``, shrink by ``buffer``) to merge
    adjacent slivers and drop features narrower than ``min_width``, matching
    common walkable-green-space definitions used in accessibility studies.

    Args:
        bounds: AOI geometry.
        min_area: Minimum polygon area (m²) to keep.
        min_width: Minimum feature width (m) to keep (erosion radius).
        buffer: Extra buffer (m) applied during the dilation step.

    Returns:
        Polars DataFrame with a WKB ``geometry`` column, in the CRS of ``bounds``.
    """
    query = """
        [out:json][timeout:25];
        (
        node[leisure = "garden"]({{bbox}});
        node[leisure = "park"]({{bbox}});
        node[landuse = "greenfield"]({{bbox}});
        node[landuse = "grass"]({{bbox}});
        node[landuse = "forest"]({{bbox}});
        way[leisure = "garden"]({{bbox}});
        way[leisure = "park"]({{bbox}});
        way[landuse = "greenfield"]({{bbox}});
        way[landuse = "grass"]({{bbox}});
        way[landuse = "forest"]({{bbox}});
        relation[leisure = "garden"]({{bbox}});
        relation[leisure = "park"]({{bbox}});
        relation[landuse = "greenfield"]({{bbox}});
        relation[landuse = "grass"]({{bbox}});
        relation[landuse = "forest"]({{bbox}});
        );
        out body;
        >;
        out skel qt;
    """
    gdf = _polars_wkb_to_gdf(overpass_query(query, bounds), crs=4326)
    crs = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(crs)
    gdf = gdf[gdf.geometry.area > min_area]
    merged = gdf.geometry.union_all()
    merged = shapely.buffer(merged, -min_width, quad_segs=2)
    merged = shapely.buffer(merged, buffer + min_width, quad_segs=2)
    merged = shapely.buffer(merged, -buffer, quad_segs=2)
    result = gpd.GeoDataFrame({}, geometry=shapely.get_parts(merged), crs=crs).to_crs(bounds.crs)
    return _gdf_to_polars(result)


def _polars_wkb_to_gdf(df: pl.DataFrame, crs) -> gpd.GeoDataFrame:
    """Reconstruct a GeoDataFrame from a Polars DataFrame with a WKB ``geometry`` column."""
    geom = shapely.from_wkb(df["geometry"].to_numpy())
    return gpd.GeoDataFrame(df.drop("geometry").to_pandas(), geometry=geom, crs=crs)


def _simple_poi_query(query: str, bounds) -> pl.DataFrame:
    return overpass_query(query, bounds)


def bus_stops(bounds) -> pl.DataFrame:
    """Fetch bus stop points from OSM (``highway=bus_stop``)."""
    return _simple_poi_query(
        '[out:json][timeout:25];(node["highway"="bus_stop"]({{bbox}}););out body;>;out skel qt;', bounds
    )


def schools(bounds) -> pl.DataFrame:
    """Fetch school points/polygons from OSM (``amenity=school``)."""
    return _simple_poi_query(
        '[out:xml][timeout:25];(node["amenity"="school"]({{bbox}});way["amenity"="school"]({{bbox}});'
        'relation["amenity"="school"]({{bbox}}););(._;>;);out body;',
        bounds,
    )


def healthcare(bounds) -> pl.DataFrame:
    """Fetch healthcare facilities from OSM (hospitals, clinics, doctors, ``healthcare=*``)."""
    return _simple_poi_query(
        '[out:xml][timeout:25];(node["amenity"~"hospital|clinic|doctors|healthcare"]({{bbox}});'
        'way["amenity"~"hospital|clinic|doctors|healthcare"]({{bbox}});'
        'relation["amenity"~"hospital|clinic|doctors|healthcare"]({{bbox}});'
        'node["healthcare"]({{bbox}});way["healthcare"]({{bbox}});relation["healthcare"]({{bbox}}););'
        '(._;>;);out body;',
        bounds,
    )


def groceries(bounds) -> pl.DataFrame:
    """Fetch grocery/supermarket/market POIs from OSM."""
    return _simple_poi_query(
        '[out:xml][timeout:25];(node["shop"~"supermarket|grocery|convenience"]({{bbox}});'
        'way["shop"~"supermarket|grocery|convenience"]({{bbox}});'
        'relation["shop"~"supermarket|grocery|convenience"]({{bbox}});'
        'node["amenity"="marketplace"]({{bbox}});way["amenity"="marketplace"]({{bbox}});'
        'relation["amenity"="marketplace"]({{bbox}}););(._;>;);out body;',
        bounds,
    )


def shops(bounds) -> pl.DataFrame:
    """Fetch all ``shop=*`` POIs from OSM."""
    return _simple_poi_query(
        '[out:xml][timeout:25];(node["shop"]({{bbox}});way["shop"]({{bbox}});'
        'relation["shop"]({{bbox}}););(._;>;);out body;',
        bounds,
    )


def restaurants(bounds) -> pl.DataFrame:
    """Fetch restaurant/bar/pub/cafe/fast-food POIs from OSM."""
    return _simple_poi_query(
        '[out:xml][timeout:25];(node["amenity"~"restaurant|bar|pub|cafe|fast_food"]({{bbox}});'
        'way["amenity"~"restaurant|bar|pub|cafe|fast_food"]({{bbox}});'
        'relation["amenity"~"restaurant|bar|pub|cafe|fast_food"]({{bbox}}););(._;>;);out body;',
        bounds,
    )


def libraries(bounds) -> pl.DataFrame:
    """Fetch library POIs from OSM (``amenity=library``)."""
    return _simple_poi_query(
        '[out:xml][timeout:25];(node["amenity"="library"]({{bbox}});way["amenity"="library"]({{bbox}});'
        'relation["amenity"="library"]({{bbox}}););(._;>;);out body;',
        bounds,
    )


def pharmacies(bounds) -> pl.DataFrame:
    """Fetch pharmacy POIs from OSM (``amenity=pharmacy``)."""
    return _simple_poi_query(
        '[out:xml][timeout:25];(node["amenity"="pharmacy"]({{bbox}});way["amenity"="pharmacy"]({{bbox}});'
        'relation["amenity"="pharmacy"]({{bbox}}););(._;>;);out body;',
        bounds,
    )


def gyms(bounds) -> pl.DataFrame:
    """Fetch gym POIs from OSM (``amenity=gym``)."""
    return _simple_poi_query(
        '[out:xml][timeout:25];(node["amenity"="gym"]({{bbox}});way["amenity"="gym"]({{bbox}});'
        'relation["amenity"="gym"]({{bbox}}););(._;>;);out body;',
        bounds,
    )


def cinemas(bounds) -> pl.DataFrame:
    """Fetch cinema POIs from OSM (``amenity=cinema``)."""
    return _simple_poi_query(
        '[out:xml][timeout:25];(node["amenity"="cinema"]({{bbox}});way["amenity"="cinema"]({{bbox}});'
        'relation["amenity"="cinema"]({{bbox}}););(._;>;);out body;',
        bounds,
    )
