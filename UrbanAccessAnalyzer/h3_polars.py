import warnings
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import polars as pl
import geopandas as gpd
import pandas as pd
import h3

from shapely.geometry import Polygon
from shapely.ops import unary_union

from rasterio.transform import Affine
from rasterio.crs import CRS

from . import raster_utils

# ------------------------------------------------------------------------------
# Geometry type groups
# ------------------------------------------------------------------------------
# Explicit geometry type groupings used to determine how each geometry
# should be rasterized into H3 cells.

"""TODO: Relies on H3 experimental APIs which may change behavior."""

POLYGON_TYPES: tuple[str, ...] = ("Polygon", "MultiPolygon")
POINT_TYPES: tuple[str, ...] = ("Point",)
BUFFER_TYPES: tuple[str, ...] = (
    "LineString",
    "MultiLineString",
    "LinearRing",
    "MultiPoint",
)
GEOMETRY_COLLECTION_TYPE: str = "GeometryCollection"


# ------------------------------------------------------------------------------
# Rasterization
# ------------------------------------------------------------------------------

def cells_in_geometry(
    gdf: Union[pl.DataFrame, pl.LazyFrame],
    resolution: int,
    contain: Literal[
        "center",
        "full",
        "overlap",
        "bbox_overlap",
        "centroid",
        "center_overlap",
    ] = "center_overlap",
) -> pl.LazyFrame:
    """
    Rasterize vector geometries into H3 hexagonal cells.

    This function converts geometries into H3 cell indices at a given
    resolution. It supports polygons, points, buffered line geometries,
    and geometry collections.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame or geopandas.GeoSeries
        Input geometries. If a GeoDataFrame is provided, only the
        ``geometry`` column is used.
    resolution : int
        H3 resolution level (0–15). Higher values produce smaller hexagons.
    buffer : float, default 0.0
        Buffer distance (in CRS units). Line- and point-like geometries
        are buffered into polygons before rasterization.
    contain : Literal[
        "center", "full", "overlap", "bbox_overlap",
        "centroid", "center_overlap"
    ], default "center_overlap"
        Containment rule used by H3 polygon filling.

        - ``center``: cell center must be inside geometry
        - ``full``: entire cell must be inside geometry
        - ``overlap``: any overlap counts
        - ``bbox_overlap``: bounding box overlap
        - ``centroid``: rasterize centroids instead of shapes
        - ``center_overlap``: try ``center`` first, then fallback to ``overlap``

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with an added ``h3_cells`` column containing a list
        of H3 cell indices for each input geometry.

    Notes
    -----
    - Buffering in a geographic CRS automatically reprojects to UTM.
    - GeometryCollections are flattened using unary union.
    - Uses experimental H3 polygon fill APIs when required.

    Warnings
    --------
    Relies on H3 experimental APIs which may change behavior.
    """

    # Handle CRS for buffering vs. non-buffering cases
    if buffer > 0:
        if gdf.crs and gdf.crs.is_geographic:
            gdf = gdf.to_crs(gdf.estimate_utm_crs())
    else:
        # Ensure WGS84 when no buffering is applied
        gdf = gdf.to_crs(4326)
        
    # Prepare output column
    gdf["h3_cells"] = None

    # Apply explicit buffer when requested
    if buffer > 0:
        gdf["geometry"] = gdf.geometry.buffer(buffer)

    # Replace geometries with centroids if requested
    if contain == "centroid":
        gdf = gdf.centroid

    # Flatten GeometryCollections into a single geometry
    mask_gc = gdf.geom_type == GEOMETRY_COLLECTION_TYPE
    if mask_gc.any():
        gdf.loc[mask_gc, "geometry"] = gdf.loc[mask_gc, "geometry"].apply(
            lambda g: unary_union([geom for geom in g.geoms if not geom.is_empty])
        )

    # Slightly buffer line- and multipoint-like geometries
    mask_buffer = gdf.geom_type.isin(BUFFER_TYPES)
    if mask_buffer.any():
        if gdf.crs.is_geographic:
            gdf = gdf.to_crs(gdf.estimate_utm_crs())
        gdf.loc[mask_buffer, "geometry"] = (
            gdf.loc[mask_buffer, "geometry"].buffer(0.01)
        )

    # Reproject back to WGS84 for H3 operations
    gdf = gdf.to_crs(4326)

    # ------------------------------------------------------------------
    # Polygon rasterization
    # ------------------------------------------------------------------
    mask_polygons = gdf.geom_type.isin(POLYGON_TYPES)

    if mask_polygons.any():
        # Resolve containment mode
        if contain == "center_overlap":
            contain_mode = "center"
        elif contain == "centroid":
            raise ValueError("Polygons remain even though contain='centroid'")
        else:
            contain_mode = contain

        # Standard (non-experimental) center containment
        if contain_mode == "center":
            gdf.loc[mask_polygons, "h3_cells"] = gdf.loc[
                mask_polygons
            ].apply(
                lambda row: h3.h3shape_to_cells(
                    h3.geo_to_h3shape(row.geometry),
                    res=resolution,
                ),
                axis=1,
            )
        # Experimental containment modes
        else:
            gdf.loc[mask_polygons, "h3_cells"] = gdf.loc[
                mask_polygons
            ].apply(
                lambda row: h3.h3shape_to_cells_experimental(
                    h3.geo_to_h3shape(row.geometry),
                    res=resolution,
                    contain=contain_mode,
                ),
                axis=1,
            )

        # Fallback to overlap mode when center containment yields no cells
        if contain == "center_overlap":
            fallback_mask = mask_polygons & (
                gdf["h3_cells"].isna()
                | gdf["h3_cells"].apply(lambda x: isinstance(x, list) and len(x) == 0)
            )
            gdf.loc[fallback_mask, "h3_cells"] = gdf.loc[
                fallback_mask
            ].apply(
                lambda row: h3.h3shape_to_cells_experimental(
                    h3.geo_to_h3shape(row.geometry),
                    res=resolution,
                    contain="overlap",
                ),
                axis=1,
            )

    # ------------------------------------------------------------------
    # Point rasterization
    # ------------------------------------------------------------------
    mask_points = gdf.geom_type.isin(POINT_TYPES)
    if mask_points.any():
        gdf.loc[mask_points, "h3_cells"] = gdf.loc[
            mask_points
        ].apply(
            lambda row: [
                h3.latlng_to_cell(
                    lat=row.geometry.y,
                    lng=row.geometry.x,
                    res=resolution,
                )
            ],
            axis=1,
        )

    return gdf


# ------------------------------------------------------------------------------
# Explode & aggregate
# ------------------------------------------------------------------------------


def aggregate(
    h3_df: pl.DataFrame | pl.LazyFrame,
    columns: List[str] = [],
    value_order: Union[List, Dict[str, List]] = {},
    method: Union[str, Dict[str, str]] = "max",
    h3_column: Optional[str] = 'h3_cell'
) -> pl.LazyFrame:
    """
    Explode H3 cell lists and aggregate values per cell.

    Parameters
    ----------
    h3_df : geopandas.GeoDataFrame or pandas.DataFrame
        Output of :func:`cells_in_geometry`, must contain ``h3_cells``.
    columns : list of str, optional
        Columns to aggregate. Defaults to all non-geometry, non-H3 columns.
    value_order : list or dict, optional
        Explicit ordering for categorical values.
    method : str or dict
        Aggregation method per column.

        Supported methods:
        - ``first``, ``last``, ``min``, ``max``
        - ``mean``, ``sum``
        - ``density`` (value per unit area)
        - ``distribute`` (evenly distribute value across cells)

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by ``h3_cell`` with aggregated values.
    """
    if h3_column is None:
        if "h3_cells" in h3_df.collect_schema().names():
            h3_column = "h3_cells" 
        elif "h3_cell" in h3_df.collect_schema().names():
            h3_column = "h3_cell" 
        else:
            raise Exception("Param h3_column is needed as h3 cell column could not be infered.")

    if h3_column != "h3_cell": 
        if "h3_cell" in h3_df.columns:
            warnings.warn(f"h3_cell column exists in h3_df. Dropping h3_cell column and renaming {h3_column} to h3_cell.")
            h3_df = h3_df.drop("h3_cell")

        h3_df = h3_df.rename({h3_column: "h3_cell"}) 

    # Replace "nan" and "None" strings with None
    h3_df = h3_df.with_columns([
        pl.when((pl.col(c) == "nan") | (pl.col(c) == "None"))
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in h3_df.collect_schema().names()
    ])

    # Determine aggregation columns
    if (columns is None) or (len(columns) == 0):
        columns = [
            c for c in h3_df.columns
            if c != "h3_cell" and not isinstance(h3_df[c], gpd.GeoSeries)
        ]


    # Fallback when no columns are provided
    delete_columns = []
    if len(columns) == 0:
        h3_df = h3_df.with_columns(
            pl.lit(0).alias("column")
        )
        columns = ["column"]
        delete_columns.append("column")

    h3_df = h3_df.filter(
        ~pl.all(columns).is_null()
    )

    # Normalize value_order into dict form
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

    # Encode ordered categorical columns as integers
    mapped_cols: Dict[str, str] = {}
    all_columns = ["h3_cell"]

    for col in value_order:
        if value_order[col] is not None and len(value_order[col]) > 0:
            # Filter out None values for type inference
            non_null = [v for v in value_order[col] if v is not None]

            # Determine the common type
            if all(isinstance(v, str) for v in non_null):
                # All strings
                common_type = str
            elif all(isinstance(v, (int, float)) for v in non_null):
                # All numbers -> promote to float if needed
                if any(isinstance(v, float) for v in non_null):
                    common_type = float
                else:
                    common_type = int
            else:
                # Mixed types or other -> fallback to object
                common_type = object
            

            # Cast the column, keeping None safe
            if common_type in (int, float, str):
                h3_df = h3_df.with_columns(
                    pl.col(col).cast(common_type)
                )

            mapping = {
                v if v is None else common_type(v): i
                for i, v in enumerate(value_order[col])
            }
            # Use pl.when().then() chain
            expr = None
            for v, i in mapping.items():
                if expr is None:
                    expr = pl.when(pl.col(col) == v).then(i)
                else:
                    expr = expr.when(pl.col(column) == v).then(i)

            # Handle nulls / unmapped values
            expr = expr.otherwise(pl.lit(None))
        
            h3_df = h3_df.with_columns(
                expr.alias(f"_{col}_int")
            )
            mapped_cols[col] = f"_{col}_int"
            all_columns.append(f"_{col}_int")
        else:
            all_columns.append(col)

    # Normalize aggregation method specification
    if not isinstance(method, dict):
        method = {col: method for col in columns}

    for col in columns:
        if col not in method:
            raise Exception(f"Value missing for column {col} in param method {method}")

    agg_dict = {}
    col_totals = {}

    # Build aggregation instructions
    for col, m in method.items():
        if col in mapped_cols:
            col = mapped_cols[col]

        if m == "first":
            agg_dict[col] = pl.first()
        elif m == "last":
            agg_dict[col] = pl.last()
        elif m == "max":
            agg_dict[col] = pl.max()
        elif m == "min":
            agg_dict[col] = pl.min()
        elif m == "mean":
            h3_df = h3_df.with_columns(
                pl.col(col).cast(float)
            )
            agg_dict[col] = pl.mean()
        elif m == "sum":
            h3_df = h3_df.with_columns(
                pl.col(col).cast(float)
            )
            agg_dict[col] = pl.sum()
        elif m == "density":
            agg_dict[col] = pl.mean()
            agg_dict["h3_cell_area"] = pl.first()
            h3_df = h3_df.with_columns(
                pl.col(col).cast(float)
            )
            col_totals[col] = h3_df.select(pl.col(col).sum())
            if "area" not in h3_df.collect_schema().names():
                raise Exception("area must be in columns of h3_df to apply density method.")
            
            h3_df = h3_df.with_columns(
                (pl.col(col) / pl.col("area")).alias(col)
            )
            if "h3_cell_area" not in h3_df.collect_schema().names():
                h3_df = h3_df.with_columns(
                    pl.col(col)
                    .map_elements(lambda x: h3.cell_area(x, unit="m^2"))
                    .alias("h3_cell_area")
                )
        elif m == "distribute":
            agg_dict[col] = pl.sum()
            is_list = isinstance(h3_df.schema["h3_cell"], pl.List)
            if is_list:
                h3_df = h3_df.with_columns(
                    (
                        pl.col(col).cast(float)
                        / pl.col("h3_cell").list.len()
                    ).alias(col)
                )
            else:
                h3_df = h3_df.with_columns(
                    (
                        pl.col(col).cast(float)
                        / pl.len().over("h3_cell")
                    ).alias(col)
                )
        else:
            raise NotImplementedError(f"Aggregation method '{m}' not implemented")

    # Explode list-valued H3 cells into rows
    h3_df = h3_df.select(all_columns)
    h3_df = h3_df.explode("h3_cell")

    # Aggregate per H3 cell
    result = h3_df.group_by("h3_cell").agg(agg_dict)

    # Re-normalize density totals if needed
    if len(col_totals) > 0:
        for col in col_totals:
            result = result.with_columns(
                (pl.col(col) * pl.col("h3_cell_area")).alias(col)
            )
            result = result.with_columns(
                (pl.col(col) * (col_totals[col] / pl.col(col).sum())).alias(col)
            )

    # Decode categorical columns back to original values
    if len(mapped_cols) > 0:
        for col in value_order:
            if value_order[col] is not None and len(value_order[col]) > 0:
                mapping = {
                    i: v for i, v in enumerate(value_order[col])
                }
                # Use pl.when().then() chain
                expr = None
                for v, i in mapping.items():
                    if expr is None:
                        expr = pl.when(pl.col(f"_{col}_int") == i).then(v)
                    else:
                        expr = expr.when(pl.col(f"_{col}_int") == i).then(v)

                # Handle nulls / unmapped values
                expr = expr.otherwise(pl.lit(None))
            
                result = result.with_columns(
                    expr.alias(col)
                ).drop(f"_{col}_int")
                
    # Drop rows with no aggregated values
    cols = [c for c in result.columns if c != "h3_cell"]

    result = result.filter(
        ~pl.any_horizontal(pl.col(cols).is_null())
    )

    return result


# ------------------------------------------------------------------------------
# High-level convenience wrapper
# ------------------------------------------------------------------------------

def from_gdf(
    gdf: gpd.GeoDataFrame,
    resolution: int,
    columns: Optional[List[str]] = None,
    value_order: Optional[Union[List, Dict[str, List]]] = None,
    buffer: float = 0.0,
    contain: Literal[
        "center",
        "full",
        "overlap",
        "bbox_overlap",
        "centroid",
        "center_overlap",
    ] = "center_overlap",
    method: Union[str, Dict[str, str]] = "max",
):
    """
    Rasterize a GeoDataFrame directly into aggregated H3 cells.

    This is a convenience wrapper combining :func:`cells_in_geometry`
    and :func:`explode`.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input geometries and attributes.
    resolution : int
        H3 resolution.
    columns : list of str, optional
        Columns to aggregate.
    value_order : list or dict, optional
        Category ordering.
    buffer : float, default 0.0
        Buffer distance applied before rasterization.
    contain : Literal[
        "center", "full", "overlap", "bbox_overlap",
        "centroid", "center_overlap"
    ], default "center_overlap"
        Containment rule for polygon filling.
    method : str or dict, default "max"
        Aggregation method per column.

        Supported methods:
        - ``first``, ``last``, ``min``, ``max``
        - ``mean``, ``sum``
        - ``density`` (value per unit area)
        - ``distribute`` (evenly distribute value across cells)


    Returns
    -------
    pandas.DataFrame
        Aggregated values indexed by H3 cell.
    """

    h3_df = cells_in_geometry(
        gdf,
        resolution=resolution,
        buffer=buffer,
        contain=contain,
    )
    if method == "density":
        if h3_df.crs.is_projected == False:
            h3_df = h3_df.to_crs(h3_df.estimate_utm_crs())

        h3_df["area"] = h3_df.geometry.area

    if "geometry" in h3_df.columns:
        h3_df = h3_df.drop(columns="geometry")

    h3_df = pl.from_pandas(pd.DataFrame(h3_df))
    return aggregate(
        h3_df,
        columns=columns,
        value_order=value_order,
        method=method,
    )


def from_raster(
    raster: np.ndarray | str,
    aoi=None,
    resolution: int = 10,
    contain: Literal[
        "center",
        "full",
        "overlap",
        "bbox_overlap",
        "centroid",
        "center_overlap",
    ] = "center_overlap",
    method: Union[str, Dict[str, str]] = "distribute",
    value_order: Optional[List] = None,
    transform: Affine | None = None,
    crs: CRS | None = None,
    nodata=None,
):
    """
    Rasterize a raster dataset into aggregated H3 cells.

    This function converts a raster (either provided as a file path or as
    an in-memory NumPy array) into vector polygons, then aggregates raster
    values into H3 hexagonal cells using :func:`from_gdf`.

    The raster is first vectorized (one polygon per contiguous pixel region),
    and raster values are propagated to H3 cells according to the specified
    aggregation method.

    Parameters
    ----------
    raster : numpy.ndarray or str
        Either:
        - A NumPy array representing raster values, or
        - A file path to a raster readable by ``raster_utils.read_raster``.
    aoi : optional
        Area of interest used to crop the raster when ``raster`` is a file path.
        Ignored (with warning) when a raster array is passed directly.
    resolution : int, default 10
        H3 resolution level (0–15). Higher values produce smaller hexagons.
    contain : Literal[
        "center", "full", "overlap", "bbox_overlap",
        "centroid", "center_overlap"
    ], default "center_overlap"
        Containment rule for polygon filling during raster-to-H3 conversion.

        - ``center``: cell center must be inside geometry
        - ``full``: entire cell must be inside geometry
        - ``overlap``: any overlap counts
        - ``bbox_overlap``: bounding box overlap
        - ``centroid``: rasterize centroids instead of shapes
        - ``center_overlap``: try ``center`` first, fallback to ``overlap``
    method : str or dict, default "distribute"
        Aggregation method passed to :func:`explode` via :func:`from_gdf`.

        Supported methods:
        - ``first``, ``last``, ``min``, ``max``
        - ``mean``, ``sum``
        - ``density`` (value per unit area)
        - ``distribute`` (evenly distribute raster values across H3 cells)
    value_order : list, optional
        Explicit ordering for categorical raster values.
    transform : Affine, optional
        Affine transform of the raster. Required when ``raster`` is a NumPy array.
    crs : CRS, optional
        Coordinate reference system of the raster. Required when ``raster``
        is a NumPy array.
    nodata : optional
        Nodata value to ignore during raster reading and vectorization.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by H3 cell containing aggregated raster values.

    Notes
    -----
    - When ``raster`` is a file path, raster reading, cropping, and CRS handling
      are delegated to ``raster_utils.read_raster``.
    - When ``raster`` is a NumPy array, both ``transform`` and ``crs`` must be
      provided explicitly.
    - The resulting GeoDataFrame must contain a ``value`` column to be compatible
      with :func:`from_gdf`.
    """

    # ------------------------------------------------------------------
    # Raster loading and vectorization
    # ------------------------------------------------------------------
    if isinstance(raster, str):
        # Read raster from disk and optionally crop to AOI
        raster, transform, crs = raster_utils.read_raster(
            raster,
            aoi=aoi,
            nodata=nodata,
        )

        # Convert raster pixels to vector geometries
        gdf = raster_utils.vectorize(
            raster_array=raster,
            transform=transform,
            crs=crs,
            aoi=None,
            keep_nodata=False,
            nodata=nodata,
        )
    else:
        # AOI cropping is not supported for in-memory rasters
        if aoi is not None:
            warnings.warn(
                "aoi cropping is not allowed when passing a loaded raster. "
                "Pass the raster path."
            )

        # Transform and CRS are mandatory for array-based rasters
        if (transform is None) or (crs is None):
            raise Exception(
                "If inputing a raster array keywords transform and crs are mandatory."
            )

        # Vectorize raster array into polygons
        gdf = raster_utils.vectorize(
            raster_array=raster,
            transform=transform,
            crs=crs,
            aoi=aoi,
            keep_nodata=False,
            nodata=None,
        )

    # ------------------------------------------------------------------
    # Cleanup and aggregation
    # ------------------------------------------------------------------

    # Remove features without raster values
    gdf = gdf.dropna(subset=["value"]).reset_index(drop=True)
    if value_order is not None:
        value_order = {'value':value_order}

    # Raster-to-H3 aggregation using existing GeoDataFrame pipeline
    df = from_gdf(
        gdf,
        columns=["value"],
        resolution=resolution,
        value_order=value_order,
        contain=contain,
        method=method,
    )

    return df

def resample(df,target_resolution,columns=[],value_order={},method='max',h3_column="h3_cell"):
    if h3_column not in df.collect_schema().names():
        raise Exception(f"H3 cell column {h3_column} not in {df}")

    df = df.with_columns(
        pl.col(h3_column).map_elements(lambda x: h3.cell_to_parent(x, target_resolution)).alias(h3_column)
    )
    df = aggregate(df,columns,value_order,method,h3_column=h3_column)
    return df


def to_gdf(df, h3_column="h3_cell"):
    if h3_column not in df.collect_schema().names():
        raise Exception(f"H3 cell column {h3_column} not in {df}")

    # --- Validation ---
    # Detect proper validation function (for H3 v3/v4)
    if hasattr(h3, "is_valid_cell"):
        is_valid = h3.is_valid_cell
    elif hasattr(h3, "h3_is_valid"):
        is_valid = h3.h3_is_valid
    else:
        raise AttributeError("Cannot find a valid H3 validation function in the h3 module.")

    df = df.with_columns(
        pl.col(h3_column).map_elements(is_valid).alias(h3_column)
    )
    # --- Build geometries ---
    def cell_to_polygon(cell):
        boundary = h3.cell_to_boundary(cell)
        return Polygon([(lng, lat) for lat, lng in boundary])

    df = df.with_columns(
        pl.col(h3_column).map_elements(cell_to_polygon).alias("geometry")
    )
    if isinstance(df,pl.LazyFrame):
        df = df.collect()

    gdf = gpd.GeoDataFrame(df.to_pandas(), geometry="geometry", crs="EPSG:4326")
    return gdf
