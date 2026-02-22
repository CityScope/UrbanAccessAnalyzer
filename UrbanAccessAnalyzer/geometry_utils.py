from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import geopandas as gpd

import shapely
from shapely.geometry import box
from pyproj import Geod

geod_wgs84 = Geod(ellps="WGS84")

def geodesic_area(geom,geod=geod_wgs84) -> float:
    if geom is None or geom.is_empty:
        return 0.0

    # Handle polygons & multipolygons
    if geom.geom_type == "Polygon":
        area, _ = geod_wgs84.geometry_area_perimeter(geom)
        return abs(area)

    if geom.geom_type == "MultiPolygon":
        return sum(
            abs(geod_wgs84.geometry_area_perimeter(p)[0])
            for p in geom.geoms
        )

    return 0.0

def is_utm_reasonable(gdf: gpd.GeoDataFrame,
                      max_width_m: float = 750_000,
                      max_height_m: float = 2_000_000,
                      ellps=None
                    ) -> bool:
    """
    Check if a GeoDataFrame is reasonable for a single UTM projection.
    Uses the CRS ellipsoid directly.
    """
    if gdf.crs is None or not gdf.crs.is_geographic:
        raise ValueError("GeoDataFrame must have geographic CRS (degrees).")

    minx, miny, maxx, maxy = gdf.total_bounds

    if ellps is None:
        # Extract ellipsoid axes
        ellps = gdf.crs.ellipsoid.name.replace(" ","")

    geod = Geod(ellps=ellps)

    # Compute width and height in meters along approximate edges
    midy = (miny + maxy) / 2
    # geod.inv returns az1, az2, distance
    _, _, width_m = geod.inv(minx, midy, maxx, midy)
    _, _, height_m = geod.inv(minx, miny, minx, maxy)

    return width_m <= max_width_m and height_m <= max_height_m

def intersects_all_with_all(
    G: gpd.GeoDataFrame | gpd.GeoSeries, g: gpd.GeoDataFrame | gpd.GeoSeries
):
    """
    Compute a full pairwise spatial intersection matrix between two geometry
    collections.

    Each geometry in ``G`` is tested against every geometry in ``g`` using
    Shapely's vectorized ``intersects`` operation.

    Parameters
    ----------
    G : geopandas.GeoDataFrame or geopandas.GeoSeries
        Target geometries. Each row corresponds to one output row.
    g : geopandas.GeoDataFrame or geopandas.GeoSeries
        Source geometries. Each column corresponds to one source geometry.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(len(G), len(g))`` where ``[i, j]`` is True
        if geometry ``G.iloc[i]`` intersects geometry ``g.iloc[j]``.

    Notes
    -----
    - ``g`` is automatically reprojected to the CRS of ``G``.
    - Prepared geometries are used internally for performance.
    """
    g = g.to_crs(G.crs)
    _g = np.array(
        np.repeat(np.transpose(np.array(g.geometry)[np.newaxis, :]), len(G), axis=1)
    )
    _G = list(G.geometry)
    shapely.prepare(_G)
    shapely.prepare(_g)
    return shapely.intersects(_G, _g).transpose()


def intersects_xy_all_with_all(G: gpd.GeoDataFrame | gpd.GeoSeries, x, y=None):
    """
    Compute spatial intersections between geometries and multiple point
    coordinates.

    Points may be provided explicitly as coordinates or implicitly via
    centroids of GeoSeries or GeoDataFrames.

    Parameters
    ----------
    G : geopandas.GeoDataFrame or geopandas.GeoSeries
        Target geometries to test against.
    x : iterable, geopandas.GeoDataFrame, or geopandas.GeoSeries
        X coordinates, iterable of ``(x, y)`` tuples, or geometries whose
        centroids are used as points.
    y : iterable, optional
        Y coordinates. Required if ``x`` is a list of x-values only.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(len(G), number_of_points)`` indicating
        whether each point intersects each geometry.

    Notes
    -----
    - If ``x`` is a GeoDataFrame or GeoSeries, centroids are computed.
    - Intersection tests are fully vectorized using Shapely.
    """
    if isinstance(x, (gpd.GeoDataFrame, gpd.GeoSeries)):
        x = x.geometry.centroid
        y = list(x.y)
        x = list(x.x)

    if y is None:
        x, y = list(zip(*x))

    _x = np.array(np.repeat(np.transpose(np.array(x)[np.newaxis, :]), len(G), axis=1))
    _y = np.array(np.repeat(np.transpose(np.array(y)[np.newaxis, :]), len(G), axis=1))
    _G = list(G.geometry)
    shapely.prepare(_G)
    return shapely.intersects_xy(_G, x=_x, y=_y).transpose()


def source_ids_to_dst_geometry(
    source_gdf: Union[gpd.GeoDataFrame, gpd.GeoSeries],
    dst_gdf: Union[gpd.GeoDataFrame, gpd.GeoSeries],
    buffer_source: float = 0.0,
    buffer_dst: float = 0.0,
    contain: Literal[
        "center",
        "full",
        "overlap",
        "bbox_overlap",
        "centroid",
        "center_overlap",
    ] = "center_overlap",
    id_column: str | None = None,
    simplify_tol: float | None = None,
    clip_to_dst_bbox: bool = True,
) -> gpd.GeoDataFrame:
    # ---------------------------
    # Normalize inputs
    # ---------------------------
    if isinstance(source_gdf, gpd.GeoSeries):
        source_gdf = gpd.GeoDataFrame(geometry=source_gdf, crs=source_gdf.crs)
    else:
        source_gdf = source_gdf.copy()

    if isinstance(dst_gdf, gpd.GeoSeries):
        dst_gdf = gpd.GeoDataFrame(geometry=dst_gdf, crs=dst_gdf.crs)
    else:
        dst_gdf = dst_gdf.copy()

    # ---------------------------
    # ID handling
    # ---------------------------
    if id_column is None:
        if source_gdf.index.name is None:
            id_column = "index"
            source_gdf[id_column] = source_gdf.index
        else:
            id_column = source_gdf.index.name
            source_gdf = source_gdf.reset_index()

    if id_column not in source_gdf.columns:
        raise Exception(
            f"ID column {id_column} not found in source_gdf {source_gdf.columns}."
        )

    # ---------------------------
    # CRS alignment
    # ---------------------------
    dst_gdf = dst_gdf.to_crs(source_gdf.crs)

    # ---------------------------
    # Optional simplification
    # ---------------------------
    if simplify_tol is not None:
        source_gdf.geometry = source_gdf.geometry.simplify(simplify_tol)

    # ---------------------------
    # Optional clipping
    # ---------------------------
    if clip_to_dst_bbox:
        dst_total_bounds = dst_gdf.total_bounds  # xmin, ymin, xmax, ymax
        dst_box = box(*dst_total_bounds)
        source_gdf = source_gdf[source_gdf.intersects(dst_box)].copy()

    # ---------------------------
    # Buffer (safe handling)
    # ---------------------------
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

    # ---------------------------
    # Containment logic
    # ---------------------------
    if contain == "center":
        left = source_gdf.copy()
        left.geometry = left.geometry.centroid
        joined = gpd.sjoin(left, dst_gdf, predicate="within", how="inner")

    elif contain == "centroid":
        right = dst_gdf.copy()
        right.geometry = right.geometry.centroid
        joined = gpd.sjoin(source_gdf, right, predicate="contains", how="inner")

    elif contain == "overlap" or contain == "full":
        joined = gpd.sjoin(source_gdf, dst_gdf, predicate="intersects", how="inner")

    elif contain == "center_overlap":
        # First pass: centroid-in-polygon
        left = source_gdf.copy()
        left.geometry = left.geometry.centroid
        joined_center = gpd.sjoin(left, dst_gdf, predicate="within", how="inner")
        matched_sources = joined_center.index.unique()

        # Second pass: fallback to intersects for unmatched
        remaining = source_gdf.loc[~source_gdf.index.isin(matched_sources)]
        joined_overlap = gpd.sjoin(
            remaining, dst_gdf, predicate="intersects", how="inner"
        )

        joined = pd.concat([joined_center, joined_overlap], axis=0)

    elif contain == "bbox_overlap":
        # Use bounding boxes for fast spatial join
        # Add temporary bounding boxes
        src_bbox = source_gdf.geometry.bounds
        dst_bbox = dst_gdf.geometry.bounds

        # Construct rectangles as GeoSeries
        source_rects = gpd.GeoSeries(
            [box(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in src_bbox.values],
            crs=source_gdf.crs,
        )
        dst_rects = gpd.GeoSeries(
            [box(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in dst_bbox.values],
            crs=dst_gdf.crs,
        )

        # Replace geometry temporarily
        source_gdf_tmp = source_gdf.copy()
        source_gdf_tmp.geometry = source_rects
        dst_gdf_tmp = dst_gdf.copy()
        dst_gdf_tmp.geometry = dst_rects

        joined = gpd.sjoin(
            source_gdf_tmp, dst_gdf_tmp, predicate="intersects", how="inner"
        )

    else:
        raise NotImplementedError(f"Contain mode '{contain}' not implemented")

    # ---------------------------
    # Aggregate source IDs per destination
    # ---------------------------
    if id_column not in joined.columns:
        _id_column = id_column + "_left"
    else:
        _id_column = id_column

    result = (
        joined.groupby("index_right")[_id_column]
        .apply(list)
        .reindex(dst_gdf.index, fill_value=[])
    )

    dst_gdf[id_column] = result.values

    return dst_gdf


def aggregate(
    df: gpd.GeoDataFrame | pd.DataFrame,
    geometries: gpd.GeoDataFrame,
    columns: List[str] = None,
    value_order: Union[List, Dict[str, List]] = None,
    method: Union[str, Dict[str, str]] = "max",
    id_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aggregate attribute values by identifier after exploding list-valued
    spatial relationships.

    This function groups rows by ``id_column`` after exploding list-valued
    identifiers and applies column-wise aggregation rules.

    Parameters
    ----------
    df : geopandas.GeoDataFrame or pandas.DataFrame
        Input data containing attributes and list-valued identifiers.
    geometries : geopandas.GeoDataFrame
        Geometry reference table keyed by ``id_column``.
    columns : list of str, optional
        Columns to aggregate. Defaults to all non-identifier columns.
    value_order : list or dict, optional
        Explicit ordering for categorical aggregation.
    method : str or dict, default "max"
        Aggregation method(s) to apply per column.
    id_column : str, optional
        Identifier column used for grouping.

    Returns
    -------
    pandas.DataFrame
        Aggregated values indexed by ``id_column`` and joined with geometry.

    Notes
    -----
    - Supports numeric, categorical, density, and distributive aggregation.
    - Geometry is preserved via a final join with ``geometries``.
    """
    if columns is None:
        columns = []
    if value_order is None:
        value_order = {}
    df = df.copy()

    if id_column is None:
        if df.index.name is None:
            raise Exception("Param id_column is needed or index should be named.")
        else:
            id_column = df.index.name
            df = df.reset_index()

    if id_column not in geometries.columns:
        if id_column == geometries.index.name:
            geometries = geometries.reset_index()
        else:
            raise Exception(f"geometries does not have column {id_column}")

    df = df.replace(["nan", "None", np.nan], None)

    if (columns is None) or (len(columns) == 0):
        columns = [
            c
            for c in df.columns
            if c != id_column and not isinstance(df[c], gpd.GeoSeries)
        ]

    if len(columns) == 0:
        df["idx"] = df.index
        columns = ["idx"]

    df = df.dropna(how="all", subset=columns)

    if not isinstance(value_order, dict):
        if value_order is None:
            value_order: Dict[str, list | None] = {}
        else:
            if not isinstance(value_order, list):
                value_order = [value_order]
            value_order = {col: value_order for col in columns}

    for col in columns:
        if col not in value_order.keys():
            value_order[col] = None

    mapped_cols: Dict[str, str] = {}
    all_columns = [id_column]

    for col in value_order:
        if value_order[col] is not None and len(value_order[col]) > 0:
            non_null = [v for v in value_order[col] if v is not None]

            if all(isinstance(v, str) for v in non_null):
                common_type = str
            elif all(isinstance(v, (int, float)) for v in non_null):
                common_type = (
                    float if any(isinstance(v, float) for v in non_null) else int
                )
            else:
                common_type = object

            if common_type in (int, float, str):
                mask = df[col].notna()
                df.loc[mask, col] = df.loc[mask, col].astype(common_type)
            else:
                df[col] = df[col].astype(object)

            mapping = {
                v if v is None else common_type(v): i
                for i, v in enumerate(value_order[col])
            }
            df[f"_{col}_int"] = (
                df[col]
                .map(mapping)
                .where(df[col].isin(value_order[col]), len(value_order[col]))
            )
            mapped_cols[col] = f"_{col}_int"
            all_columns.append(f"_{col}_int")
        else:
            all_columns.append(col)

    if not isinstance(method, dict):
        method = {col: method for col in columns}

    agg_dict: Dict[str, str] = {}
    col_totals: Dict[str, str] = {}

    for col, m in method.items():
        if col in mapped_cols:
            col = mapped_cols[col]

        if m in {"first", "last", "max", "min"}:
            if m in {"max", "min"}:
                df[col] = pd.to_numeric(df[col])

            agg_dict[col] = m
        elif m == "mean":
            df[col] = df[col].astype(float)
            agg_dict[col] = "mean"
        elif m == "sum":
            s = pd.to_numeric(df[col], errors="coerce")
            df[col] = s.astype(int) if (s.dropna() % 1 == 0).all() else s.astype(float)
            agg_dict[col] = "sum"
        elif m == "density":
            if not isinstance(df, gpd.GeoDataFrame):
                raise Exception("method 'density' requires h3_df to be a GeoDataFrame.")

            if df.crs and df.crs.is_geographic:
                df = df.to_crs(df.estimate_utm_crs())

            agg_dict[col] = "mean"
            agg_dict[f"{id_column}_area"] = "first"
            df[col] = df[col].astype(float)
            col_totals[col] = df[col].sum()
            df[col] = df[col] / df.area
        elif m == "distribute":
            agg_dict[col] = "sum"
            df[col] = df[col].astype(float)
            df[col] = df[col] / df[id_column].apply(
                lambda x: len(x) if isinstance(x, list) else 1
            )
        else:
            raise NotImplementedError(f"Aggregation method '{m}' not implemented")

    df = df[all_columns].explode(id_column).reset_index(drop=True)
    result = df.groupby(id_column).agg(agg_dict).reset_index()

    if len(col_totals) > 0:
        for col in col_totals:
            result[col] *= result[f"{id_column}_area"]
            result[col] *= col_totals[col] / result[col].sum()

    if len(mapped_cols) > 0:
        for col in value_order:
            if value_order[col] is not None and len(value_order[col]) > 0:
                mapping = {i: v for i, v in enumerate(value_order[col])}
                result[col] = result[f"_{col}_int"].map(mapping)
                result = result.drop(columns=[f"_{col}_int"])

    result = result.dropna(
        how="all",
        subset=[col for col in result.columns if col != id_column],
    )
    columns = [col for col in geometries.columns if col not in result.columns]
    if id_column not in columns:
        columns.append(id_column)

    if geometries.geometry.name not in columns:
        columns.append(geometries.geometry.name)

    result = result.merge(geometries[columns], on=id_column, how="right")

    result = gpd.GeoDataFrame(
        result, geometry=geometries.geometry.name, crs=geometries.crs
    )
    result = result.set_index(id_column)
    return result


def resample_gdf(
    source_gdf: gpd.GeoDataFrame,
    dst_gdf: gpd.GeoDataFrame | gpd.GeoSeries,
    columns: Optional[List[str]] = None,
    value_order: Optional[Union[List, Dict[str, List]]] = None,
    buffer_source: float = 0.0,
    buffer_dst: float = 0.0,
    contain: Literal[
        "center",
        "full",
        "overlap",
        "bbox_overlap",
        "centroid",
        "center_overlap",
    ] = "center_overlap",
    method: Union[str, Dict[str, str]] = "max",
    id_column: str | None = None,
) -> pd.DataFrame:
    """
    Spatially resample attributes from a source GeoDataFrame onto a destination
    geometry layer.

    This is a high-level convenience wrapper that:
    1. Assigns source feature IDs to destination geometries based on spatial
       relationships.
    2. Aggregates source attributes per destination geometry.

    Parameters
    ----------
    source_gdf : geopandas.GeoDataFrame
        Source geometries and attributes.
    dst_gdf : geopandas.GeoDataFrame or geopandas.GeoSeries
        Destination geometries.
    columns : list of str, optional
        Columns from ``source_gdf`` to aggregate.
    value_order : list or dict, optional
        Category ordering for categorical aggregation.
    buffer_source : float, default 0.0
        Optional buffer applied to source geometries before resampling.
    buffer_dst : float, default 0.0
        Optional buffer applied to destination geometries before resampling.
    contain : Literal[
        "center", "full", "overlap", "bbox_overlap",
        "centroid", "center_overlap"
    ], default "center_overlap"
        Spatial relationship rule.
    method : str or dict, default "max"
        Aggregation method(s).
    id_column : str, optional
        Identifier column used to join and aggregate.

    Returns
    -------
    pandas.DataFrame
        Aggregated attributes indexed by destination geometry identifier.
    """
    source_gdf = source_gdf.copy()

    if isinstance(dst_gdf, gpd.GeoSeries):
        dst_gdf = gpd.GeoDataFrame(
            {}, geometry=dst_gdf, crs=dst_gdf.crs, index=dst_gdf.index
        )

    if id_column is None:
        if dst_gdf.index.name is None:
            id_column = "index"
            dst_gdf["index"] = dst_gdf.index
        else:
            id_column = dst_gdf.index.name
            dst_gdf = dst_gdf.reset_index()

    if method == "density":
        source_gdf = source_gdf.to_crs(source_gdf.estimate_utm_crs())
        source_gdf[f"{id_column}_area"] = source_gdf.geometry.area

    source_gdf = source_ids_to_dst_geometry(
        dst_gdf,
        source_gdf,
        buffer_source=buffer_dst,
        buffer_dst=buffer_source,
        contain=contain,
        id_column=id_column,
    )

    result = aggregate(
        source_gdf,
        geometries=dst_gdf,
        columns=columns,
        value_order=value_order,
        method=method,
        id_column=id_column,
    )
    return result
