"""General vector-geometry utilities: file I/O, area, spatial joins, aggregation.

File I/O (:func:`read_geofile`) and the spatial-predicate join
(:func:`source_ids_to_dst_geometry`) stay on geopandas/pyogrio/shapely: file
format support (shapefile, GeoPackage, KML, ...) and spatial-predicate joins
are exactly what those libraries are for, and this module is the package's
sanctioned geometry I/O boundary (see module docstrings of
:mod:`UrbanAccessAnalyzer.osm_io` and :mod:`UrbanAccessAnalyzer.graph_ops` for
the same boundary on the OSM/graph side).

:func:`aggregate` is the one piece of this module that used to be duplicated
almost verbatim between the old ``geometry_utils.py`` (pandas, keyed by an
arbitrary id column) and ``h3_utils.py``/``h3_polars.py`` (keyed by H3 cell).
It is now a single generic Polars implementation keyed by any id column
(scalar or list-valued, for one-to-many source-to-destination assignments);
:func:`UrbanAccessAnalyzer.h3_ops.aggregate` is a thin wrapper around it.

Methodology:

- **Geodesic area** (:func:`geodesic_area`): for AOIs too large for a single
  UTM zone to stay accurate (:func:`is_utm_reasonable` checks width/height
  against configurable thresholds), area is computed on the ellipsoid
  directly via ``pyproj.Geod.geometry_area_perimeter`` rather than a
  projected approximation.
- **Containment/join modes** shared across :func:`source_ids_to_dst_geometry`
  and :mod:`UrbanAccessAnalyzer.h3_ops` (``center``, ``full``, ``overlap``,
  ``bbox_overlap``, ``centroid``, ``center_overlap``) follow the same
  semantics everywhere in the package: ``center_overlap`` tries strict
  centroid-in-polygon first and falls back to any-overlap only for sources
  that matched nothing (e.g. a destination cell smaller than a source
  feature's centroid-exclusion sliver).
"""

from typing import Dict, List, Literal, Optional, Union
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pyogrio
import shapely
from pyproj import Geod
from shapely.geometry import box

ContainMode = Literal["center", "full", "overlap", "bbox_overlap", "centroid", "center_overlap"]


def read_geofile(path, bounds=None) -> gpd.GeoDataFrame:
    """Load a vector geo file, auto-detecting format and pushing down a bbox filter.

    Tries GeoParquet first (fastest, with true bbox pushdown via
    ``geopandas.read_parquet(bbox=...)``), then falls back through
    FlatGeobuf/GeoPackage/Shapefile/GeoJSON/... via pyogrio.

    Args:
        path: File path, with or without extension. If given without an
            extension, every file sharing that base name is considered,
            fastest format first.
        bounds: Optional bbox filter: a shapely geometry, a
            ``[xmin, ymin, xmax, ymax]`` list/tuple (assumed EPSG:4326), or a
            GeoDataFrame/GeoSeries.

    Returns:
        The loaded GeoDataFrame, spatially filtered to ``bounds`` if given.

    Raises:
        FileNotFoundError: If no candidate file could be read.
    """
    def _normalize_bounds(bounds):
        if bounds is None:
            return bounds
        if isinstance(bounds, shapely.geometry.base.BaseGeometry):
            return gpd.GeoSeries([bounds], crs="EPSG:4326")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            return gpd.GeoSeries([box(*bounds)], crs="EPSG:4326")
        if isinstance(bounds, gpd.GeoDataFrame):
            return bounds.geometry
        return bounds

    path = Path(path)
    priority_exts = [
        ".geoparquet", ".parquet", ".fgb", ".gpkg", ".shp", ".geojson", ".json",
        ".gml", ".kml", ".kmz", ".tab", ".mif", ".mid", ".dxf", ".vrt",
    ]
    bounds_gs = _normalize_bounds(bounds)

    if path.suffix:
        base = path.with_suffix("")
        candidates = [base.with_suffix(ext) for ext in priority_exts]
        if base.parent.exists():
            for p in base.parent.glob(base.name + ".*"):
                if p.suffix not in priority_exts:
                    candidates.append(p)
    else:
        candidates = list(path.parent.glob(path.name + ".*"))
        candidates = sorted(candidates, key=lambda p: priority_exts.index(p.suffix) if p.suffix in priority_exts else len(priority_exts))

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            suffix = candidate.suffix.lower()
            if suffix in (".geoparquet", ".parquet"):
                bbox = None
                if bounds_gs is not None:
                    meta = gpd.read_parquet(candidate, columns=[])
                    if meta.crs is None:
                        raise ValueError("No CRS in GeoParquet")
                    bounds_proj = bounds_gs.to_crs(meta.crs)
                    bbox = tuple(bounds_proj.total_bounds)
                gdf = gpd.read_parquet(candidate, bbox=bbox)
                if gdf.geometry is None or gdf.crs is None:
                    raise ValueError("Invalid GeoParquet")
                if bbox is not None:
                    gdf = gdf[gdf.intersects(bounds_proj.union_all())]
                return gdf
            else:
                bbox = None
                if bounds_gs is not None:
                    info = pyogrio.read_info(candidate)
                    if info["crs"] is None:
                        raise ValueError("No CRS in file")
                    if info["geometry_type"] is None:
                        raise ValueError("No geometry column")
                    bounds_proj = bounds_gs.to_crs(info["crs"])
                    bbox = tuple(bounds_proj.total_bounds)
                gdf = gpd.read_file(candidate, engine="pyogrio", use_arrow=True, bbox=bbox)
                if gdf.crs is None or gdf.geometry is None:
                    raise ValueError("Invalid geodataframe")
                if bbox is not None:
                    gdf = gdf[gdf.intersects(bounds_proj.union_all())]
                return gdf
        except Exception:
            continue

    raise FileNotFoundError(f"No valid geofile found for base path: {path}")


def geodesic_area(geom, geod: Geod = Geod(ellps="WGS84")) -> float:
    """Compute ellipsoidal (geodesic) area of a Polygon/MultiPolygon in EPSG:4326.

    Args:
        geom: Shapely Polygon or MultiPolygon in lon/lat coordinates.
        geod: ``pyproj.Geod`` defining the ellipsoid.

    Returns:
        Area in square meters. ``0.0`` for empty/non-polygonal geometry.
    """
    if geom is None or geom.is_empty:
        return 0.0
    if geom.geom_type == "Polygon":
        area, _ = geod.geometry_area_perimeter(geom)
        return abs(area)
    if geom.geom_type == "MultiPolygon":
        return sum(abs(geod.geometry_area_perimeter(p)[0]) for p in geom.geoms)
    return 0.0


def is_utm_reasonable(gdf: gpd.GeoDataFrame, max_width_m: float = 750_000, max_height_m: float = 2_000_000, ellps=None) -> bool:
    """Check whether a GeoDataFrame's extent is small enough for one UTM zone to stay accurate.

    Args:
        gdf: GeoDataFrame in a geographic CRS.
        max_width_m: Maximum east-west extent (meters) considered reasonable.
        max_height_m: Maximum north-south extent (meters) considered reasonable.
        ellps: Ellipsoid name for the geodesic distance calculation; defaults
            to the CRS's own ellipsoid.

    Returns:
        ``True`` if the extent fits within the given thresholds.

    Raises:
        ValueError: If ``gdf`` is not in a geographic CRS.
    """
    if gdf.crs is None or not gdf.crs.is_geographic:
        raise ValueError("GeoDataFrame must have geographic CRS (degrees).")
    minx, miny, maxx, maxy = gdf.total_bounds
    if ellps is None:
        ellps = gdf.crs.ellipsoid.name.replace(" ", "")
    geod = Geod(ellps=ellps)
    midy = (miny + maxy) / 2
    _, _, width_m = geod.inv(minx, midy, maxx, midy)
    _, _, height_m = geod.inv(minx, miny, minx, maxy)
    return width_m <= max_width_m and height_m <= max_height_m


def area(gdf: gpd.GeoDataFrame, max_width_m: float = 750_000, max_height_m: float = 2_000_000, ellps=None, geod: Geod = Geod(ellps="WGS84")):
    """Compute per-feature area, using UTM when reasonable and geodesic area otherwise.

    Args:
        gdf: GeoDataFrame (any CRS).
        max_width_m: See :func:`is_utm_reasonable`.
        max_height_m: See :func:`is_utm_reasonable`.
        ellps: Ellipsoid override for :func:`is_utm_reasonable`.
        geod: ``pyproj.Geod`` used for the geodesic fallback.

    Returns:
        A pandas Series of areas in square meters.
    """
    if gdf.crs.is_projected:
        return gdf.geometry.area
    gdf = gdf.to_crs(4326)
    if is_utm_reasonable(gdf, max_width_m, max_height_m, ellps):
        return gdf.geometry.to_crs(gdf.estimate_utm_crs()).area
    return gdf.geometry.map(lambda geom: geodesic_area(geom, geod=geod))


def intersects_all_with_all(G: Union[gpd.GeoDataFrame, gpd.GeoSeries], g: Union[gpd.GeoDataFrame, gpd.GeoSeries]) -> np.ndarray:
    """Fully vectorized pairwise intersection matrix between two geometry collections.

    Args:
        G: Target geometries (one output row each).
        g: Source geometries (one output column each); reprojected to
            ``G``'s CRS.

    Returns:
        Boolean array of shape ``(len(G), len(g))``.
    """
    g = g.to_crs(G.crs)
    _g = np.repeat(np.array(g.geometry)[np.newaxis, :], len(G), axis=1)
    _G = list(G.geometry)
    shapely.prepare(_G)
    shapely.prepare(_g)
    return shapely.intersects(_G, _g).transpose()


def intersects_xy_all_with_all(G: Union[gpd.GeoDataFrame, gpd.GeoSeries], x, y=None) -> np.ndarray:
    """Vectorized intersection test between geometries and many point coordinates.

    Args:
        G: Target geometries.
        x: Iterable of x-coordinates, iterable of ``(x, y)`` pairs, or a
            GeoDataFrame/GeoSeries (whose centroids are used).
        y: Y-coordinates, required unless ``x`` already carries both.

    Returns:
        Boolean array of shape ``(len(G), n_points)``.
    """
    if isinstance(x, (gpd.GeoDataFrame, gpd.GeoSeries)):
        centroids = x.geometry.centroid
        x, y = list(centroids.x), list(centroids.y)
    if y is None:
        x, y = list(zip(*x))
    _x = np.repeat(np.array(x)[np.newaxis, :], len(G), axis=1)
    _y = np.repeat(np.array(y)[np.newaxis, :], len(G), axis=1)
    _G = list(G.geometry)
    shapely.prepare(_G)
    return shapely.intersects_xy(_G, x=_x, y=_y).transpose()


def source_ids_to_dst_geometry(
    source_gdf: Union[gpd.GeoDataFrame, gpd.GeoSeries],
    dst_gdf: Union[gpd.GeoDataFrame, gpd.GeoSeries],
    buffer_source: float = 0.0,
    buffer_dst: float = 0.0,
    contain: ContainMode = "center_overlap",
    id_column: Optional[str] = None,
    simplify_tol: Optional[float] = None,
    clip_to_dst_bbox: bool = True,
) -> gpd.GeoDataFrame:
    """Assign the list of intersecting source-feature ids to each destination geometry.

    Args:
        source_gdf: Source features to assign.
        dst_gdf: Destination geometries to receive assigned ids.
        buffer_source: Buffer applied to source geometries before the join.
        buffer_dst: Buffer applied to destination geometries before the join.
        contain: Spatial relationship rule; see :mod:`h3_ops` for the shared
            containment-mode semantics used across the package.
        id_column: Column in ``source_gdf`` to collect; defaults to its index.
        simplify_tol: Optional geometry simplification tolerance applied to
            ``source_gdf`` before the join (speeds up complex polygons).
        clip_to_dst_bbox: Pre-filter ``source_gdf`` to ``dst_gdf``'s bounding
            box before the (potentially expensive) join.

    Returns:
        ``dst_gdf`` with an added ``id_column`` list column of matched
        source ids (empty list where nothing matched).

    Raises:
        NotImplementedError: For an unrecognized ``contain`` mode.
    """
    source_gdf = gpd.GeoDataFrame(geometry=source_gdf, crs=source_gdf.crs) if isinstance(source_gdf, gpd.GeoSeries) else source_gdf.copy()
    dst_gdf = gpd.GeoDataFrame(geometry=dst_gdf, crs=dst_gdf.crs) if isinstance(dst_gdf, gpd.GeoSeries) else dst_gdf.copy()

    if id_column is None:
        if source_gdf.index.name is None:
            id_column = "index"
            source_gdf[id_column] = source_gdf.index
        else:
            id_column = source_gdf.index.name
            source_gdf = source_gdf.reset_index()
    if id_column not in source_gdf.columns:
        raise ValueError(f"ID column {id_column} not found in source_gdf {list(source_gdf.columns)}.")

    dst_gdf = dst_gdf.to_crs(source_gdf.crs)
    if simplify_tol is not None:
        source_gdf.geometry = source_gdf.geometry.simplify(simplify_tol)
    if clip_to_dst_bbox:
        source_gdf = source_gdf[source_gdf.intersects(box(*dst_gdf.total_bounds))].copy()
    if buffer_source > 0:
        if source_gdf.crs and source_gdf.crs.is_geographic:
            source_gdf = source_gdf.to_crs(source_gdf.estimate_utm_crs())
        dst_gdf = dst_gdf.to_crs(source_gdf.crs)
        source_gdf.geometry = source_gdf.geometry.buffer(buffer_source, resolution=4)
    if buffer_dst > 0:
        if dst_gdf.crs and dst_gdf.crs.is_geographic:
            dst_gdf = dst_gdf.to_crs(dst_gdf.estimate_utm_crs())
        source_gdf = source_gdf.to_crs(dst_gdf.crs)
        dst_gdf.geometry = dst_gdf.geometry.buffer(buffer_dst, resolution=4)

    if contain == "center":
        left = source_gdf.copy()
        left.geometry = left.geometry.centroid
        joined = gpd.sjoin(left, dst_gdf, predicate="within", how="inner")
    elif contain == "centroid":
        right = dst_gdf.copy()
        right.geometry = right.geometry.centroid
        joined = gpd.sjoin(source_gdf, right, predicate="contains", how="inner")
    elif contain in ("overlap", "full"):
        joined = gpd.sjoin(source_gdf, dst_gdf, predicate="intersects", how="inner")
    elif contain == "center_overlap":
        left = source_gdf.copy()
        left.geometry = left.geometry.centroid
        joined_center = gpd.sjoin(left, dst_gdf, predicate="within", how="inner")
        remaining = source_gdf.loc[~source_gdf.index.isin(joined_center.index.unique())]
        joined_overlap = gpd.sjoin(remaining, dst_gdf, predicate="intersects", how="inner")
        joined = pd.concat([joined_center, joined_overlap], axis=0)
    elif contain == "bbox_overlap":
        src_bbox, dst_bbox = source_gdf.geometry.bounds, dst_gdf.geometry.bounds
        source_gdf_tmp, dst_gdf_tmp = source_gdf.copy(), dst_gdf.copy()
        source_gdf_tmp.geometry = gpd.GeoSeries([box(*b) for b in src_bbox.values], crs=source_gdf.crs)
        dst_gdf_tmp.geometry = gpd.GeoSeries([box(*b) for b in dst_bbox.values], crs=dst_gdf.crs)
        joined = gpd.sjoin(source_gdf_tmp, dst_gdf_tmp, predicate="intersects", how="inner")
    else:
        raise NotImplementedError(f"Contain mode '{contain}' not implemented")

    id_col_in_joined = id_column if id_column in joined.columns else f"{id_column}_left"
    result = (
        joined.groupby("index_right")[id_col_in_joined]
        .apply(list)
        .reindex(dst_gdf.index)
        .apply(lambda x: x if isinstance(x, list) else [])
    )
    dst_gdf[id_column] = result.values
    return dst_gdf


def aggregate(
    df: pl.DataFrame,
    id_column: str,
    columns: Optional[List[str]] = None,
    value_order: Optional[Union[List, Dict[str, List]]] = None,
    method: Union[str, Dict[str, str]] = "max",
) -> pl.DataFrame:
    """Explode a list-valued (or scalar) id column and aggregate values per id.

    Generic Polars group-by-and-reduce used by both plain geometry-to-geometry
    resampling (:func:`resample_gdf`) and H3 rasterization
    (:func:`UrbanAccessAnalyzer.h3_ops.aggregate`, a thin wrapper around this
    function).

    Args:
        df: DataFrame with an ``id_column`` (scalar or ``list[...]``) plus
            attribute columns to aggregate.
        id_column: Column to group by (exploded first if list-valued).
        columns: Columns to aggregate; defaults to every other column.
        value_order: Explicit category ordering for non-numeric columns
            (needed for ``first``/``last``/``min``/``max`` to be meaningful),
            either one list shared by all ``columns`` or a
            ``{column: order}`` dict.
        method: Aggregation method per column: ``first``, ``last``, ``min``,
            ``max``, ``mean``, ``sum``, ``density`` (requires an ``area``
            column in m², total-preserving), or ``distribute`` (splits a
            value evenly across every id a row maps to, total-preserving).

    Returns:
        Polars DataFrame with one row per distinct ``id_column`` value.
    """
    columns = columns or [c for c in df.columns if c != id_column]
    if not columns:
        df = df.with_columns(pl.lit(0).alias("_count"))
        columns = ["_count"]

    df = df.filter(~pl.all_horizontal([pl.col(c).is_null() for c in columns]))

    if not isinstance(value_order, dict):
        value_order = {col: value_order for col in columns}
    value_order = {col: value_order.get(col) for col in columns}
    if not isinstance(method, dict):
        method = {col: method for col in columns}

    mapped_cols: Dict[str, str] = {}
    select_cols = [id_column]
    for col in columns:
        order = value_order.get(col)
        if order:
            mapping = {v: i for i, v in enumerate(order)}
            df = df.with_columns(pl.col(col).replace_strict(mapping, default=None).alias(f"_{col}_int"))
            mapped_cols[col] = f"_{col}_int"
            select_cols.append(f"_{col}_int")
        else:
            select_cols.append(col)

    agg_dict: Dict[str, pl.Expr] = {}
    col_totals: Dict[str, float] = {}
    list_valued = df.schema[id_column].base_type() == pl.List

    for col, m in method.items():
        target = mapped_cols.get(col, col)
        if m == "first":
            agg_dict[target] = pl.first(target)
        elif m == "last":
            agg_dict[target] = pl.last(target)
        elif m == "max":
            agg_dict[target] = pl.max(target)
        elif m == "min":
            agg_dict[target] = pl.min(target)
        elif m == "mean":
            df = df.with_columns(pl.col(target).cast(pl.Float64))
            agg_dict[target] = pl.mean(target)
        elif m == "sum":
            df = df.with_columns(pl.col(target).cast(pl.Float64))
            agg_dict[target] = pl.sum(target)
        elif m == "density":
            if "area" not in df.columns:
                raise ValueError("method 'density' requires an 'area' (m^2) column.")
            df = df.with_columns(pl.col(target).cast(pl.Float64))
            col_totals[target] = float(df[target].sum())
            df = df.with_columns((pl.col(target) / pl.col("area")).alias(target))
            agg_dict[target] = pl.mean(target)
        elif m == "distribute":
            df = df.with_columns(pl.col(target).cast(pl.Float64))
            if list_valued:
                df = df.with_columns((pl.col(target) / pl.col(id_column).list.len().clip(1)).alias(target))
            else:
                df = df.with_columns((pl.col(target) / pl.len().over(id_column)).alias(target))
            agg_dict[target] = pl.sum(target)
        else:
            raise NotImplementedError(f"Aggregation method '{m}' not implemented")

    df = df.select(list(dict.fromkeys(select_cols)))
    if list_valued:
        df = df.explode(id_column)

    result = df.group_by(id_column).agg(list(agg_dict.values()))

    for target, total in col_totals.items():
        current_sum = float(result[target].sum())
        if current_sum > 0:
            result = result.with_columns((pl.col(target) * (total / current_sum)).alias(target))

    for col, target in mapped_cols.items():
        order = value_order[col]
        reverse = {i: v for i, v in enumerate(order)}
        result = result.with_columns(pl.col(target).replace_strict(reverse, default=None).alias(col)).drop(target)

    value_cols = [c for c in result.columns if c != id_column]
    return result.filter(~pl.all_horizontal([pl.col(c).is_null() for c in value_cols]))


def resample_gdf(
    source_gdf: gpd.GeoDataFrame,
    dst_gdf: Union[gpd.GeoDataFrame, gpd.GeoSeries],
    columns: Optional[List[str]] = None,
    value_order: Optional[Union[List, Dict[str, List]]] = None,
    buffer_source: float = 0.0,
    buffer_dst: float = 0.0,
    contain: ContainMode = "center_overlap",
    method: Union[str, Dict[str, str]] = "max",
    id_column: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Spatially resample attributes from a source layer onto a destination geometry layer.

    Combines :func:`source_ids_to_dst_geometry` (spatial join) with
    :func:`aggregate` (Polars reduction), then rejoins the result to
    ``dst_gdf``'s geometry.

    Args:
        source_gdf: Source geometries and attributes.
        dst_gdf: Destination geometries to receive resampled attributes.
        columns: Columns from ``source_gdf`` to aggregate.
        value_order: See :func:`aggregate`.
        buffer_source: Buffer applied to source geometries before the join.
        buffer_dst: Buffer applied to destination geometries before the join.
        contain: Spatial relationship rule.
        method: Aggregation method(s); see :func:`aggregate`. ``"density"``
            automatically computes source-feature area in m².
        id_column: Identifier column on ``dst_gdf`` (defaults to its index).

    Returns:
        GeoDataFrame with ``dst_gdf``'s geometry and the aggregated columns.
    """
    source_gdf = source_gdf.copy()
    dst_gdf = gpd.GeoDataFrame({}, geometry=dst_gdf, crs=dst_gdf.crs) if isinstance(dst_gdf, gpd.GeoSeries) else dst_gdf.copy()

    if id_column is None:
        if dst_gdf.index.name is None:
            id_column = "index"
            dst_gdf["index"] = dst_gdf.index
        else:
            id_column = dst_gdf.index.name
            dst_gdf = dst_gdf.reset_index()

    if method == "density":
        source_gdf = source_gdf.to_crs(source_gdf.estimate_utm_crs())
        source_gdf["area"] = source_gdf.geometry.area

    joined = source_ids_to_dst_geometry(
        dst_gdf, source_gdf, buffer_source=buffer_dst, buffer_dst=buffer_source, contain=contain, id_column=id_column
    )
    attr_cols = [id_column] + (columns or [c for c in source_gdf.columns if c != source_gdf.geometry.name])
    attr_cols = [c for c in dict.fromkeys(attr_cols) if c in joined.columns]
    pl_df = pl.from_pandas(pd.DataFrame(joined[attr_cols]))

    result = aggregate(pl_df, id_column=id_column, columns=columns, value_order=value_order, method=method)
    result_pd = result.to_pandas()
    merged = dst_gdf.merge(result_pd, on=id_column, how="left", suffixes=("_dst", ""))
    return gpd.GeoDataFrame(merged, geometry=dst_gdf.geometry.name, crs=dst_gdf.crs).set_index(id_column)
