import geopandas as gpd
import pandas as pd
import numpy as np
import h3
import warnings
import shapely 
from rasterio.transform import Affine
from rasterio.crs import CRS
from . import raster_utils

from shapely.geometry import (
    Point, MultiPoint,
    LineString, MultiLineString, LinearRing,
    Polygon, MultiPolygon,
    GeometryCollection
)
from shapely.ops import unary_union

"""TODO: using h3 h3shape_to_cells_experimental function so rasterize might break"""

# Geometry groups
polygon_types = ("Polygon", "MultiPolygon")
point_types = ("Point",)
buffer_types = ("LineString", "MultiLineString", "LinearRing", "MultiPoint")
gc_type = "GeometryCollection"


def from_gdf(
    geoms: gpd.GeoDataFrame,
    resolution: int,
    value_column: str | None = None,
    value_order: list | str | None = None,
    buffer: float = 0,
    centroid: bool = False,
    method: str = 'first'
) -> pd.DataFrame:
    """
    Rasterize vector geometries (GeoDataFrame) into H3 hexagonal cells.

    This function converts input geometries into H3 cells at a given resolution.
    Optionally, it buffers the geometries, assigns categorical or numerical values,
    and aggregates results based on a specified attribute.

    Parameters
    ----------
    geoms : geopandas.GeoDataFrame
        Input GeoDataFrame containing geometries to be rasterized.
    resolution : int
        H3 resolution level (0–15). Higher values produce finer grids.
    value_column : str, optional
        Column name containing the values to group geometries by.
        If None, all geometries are assigned a default value of 0.
    value_order : list, optional
        List defining the order of values to process. If None, unique sorted values
        from `value_column` are used (sorted).
    buffer : float, optional
        Buffer distance (in the CRS units of `geoms`). If CRS is geographic,
        geometries are temporarily reprojected to UTM for accurate buffering.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with two columns:
        - `value_column`: the categorical/numerical value.
        - `h3_cell`: list of H3 cell indices covering the geometry for that value.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Polygon
    >>> gdf = gpd.GeoDataFrame({'geometry': [Polygon([(-120,35),(-119,35),(-119,36),(-120,36),(-120,35)])],
    ...                         'class': [1]}, crs="EPSG:4326")
    >>> from_gdf(gdf, resolution=6, value_column='class')
    """
    if method == 'max':
        method = 'first' 
    elif method == 'min':
        method = 'last'

    # Copy to avoid modifying original GeoDataFrame
    geoms = geoms.copy()
    categorical = False

    #buffer += h3.average_hexagon_edge_length(resolution, unit='m') * 1.2
    if buffer > 0:
        if geoms.crs.is_geographic:
            geoms = geoms.to_crs(geoms.estimate_utm_crs())
    else:
        geoms = geoms.to_crs(4326)
        
    # Assign a default value column if none provided
    if value_column is None:
        geoms["_value"] = 0
        value_column = "_value"
        value_order = [0]
        categorical = True

    geoms = geoms[[value_column,'geometry']].dropna()
    # Infer order if not provided
    if isinstance(value_order,list):
        categorical = True 
    else:
        order = list(np.unique(geoms[value_column]))
        if len(order) > (len(geoms) / 10):
            value_order = None 
            categorical = False 
        else:
            categorical = True
            if value_order == 'min':
                value_order = order
                value_order.reverse()
            elif value_order == 'max':
                value_order = order
            elif value_order is None:
                value_order = order
            else:
                raise Exception(f"Keyword value_order {value_order} is not valid.")

    if categorical == True:
        # Filter out any unexpected values not in value_order
        geoms[value_column] = geoms[value_column].astype(str)
        geoms.loc[(geoms[value_column] == 'nan') | (geoms[value_column] == 'None'), value_column] = None
        geoms = geoms.dropna(subset=[value_column])
        value_order = [str(i) for i in value_order]
        geoms = geoms[geoms[value_column].isin(value_order)]

    if centroid:
        geoms.geometry = geoms.geometry.centroid

    if categorical:
        result = geoms[[value_column,'geometry']].dissolve(value_column,as_index=False,sort=False)
    else:
        result = geoms[[value_column,'geometry']].copy()

    result = result.reset_index(drop=True)
    result["h3_cell"] = None

    if buffer > 0:
        if result.crs.is_geographic:
            result = result.to_crs(result.estimate_utm_crs())

        result["geometry"] = result.geometry.buffer(buffer)

    mask_gc = result.geom_type == gc_type

    if mask_gc.any():
        result.loc[mask_gc, "geometry"] = (
            result.loc[mask_gc, "geometry"]
            .apply(lambda g: unary_union([geom for geom in g.geoms if not geom.is_empty]))
        )

    mask_buffer = result.geom_type.isin(buffer_types)

    if mask_buffer.any():
        # Ensure correct CRS for buffering small distances
        if result.crs.is_geographic:
            result = result.to_crs(result.estimate_utm_crs())

        result.loc[mask_buffer, "geometry"] = (
            result.loc[mask_buffer, "geometry"].buffer(0.01)
        )

    result = result.to_crs(4326)

    mask_polygons = result.geom_type.isin(polygon_types)

    if mask_polygons.any():
        result.loc[mask_polygons, "h3_cell"] = (
            result.loc[mask_polygons]
            .apply(
                lambda row: h3.h3shape_to_cells_experimental(
                    h3.geo_to_h3shape(row.geometry),
                    res=resolution,
                    contain="overlap",
                ),
                axis=1,
            )
        )

    mask_points = result.geom_type.isin(point_types)

    if mask_points.any():
        result.loc[mask_points, "h3_cell"] = (
            result.loc[mask_points]
            .apply(
                lambda row: [
                    h3.h3shape_to_cells(
                    h3.geo_to_h3shape(row.geometry),
                    res=resolution,
                )],
                axis=1,
            )
        )

    result = result.explode('h3_cell')
    result = result.dropna().reset_index(drop=True)
    
    if (value_order is not None) and (len(value_order) > 0):
        mapping = {v: i for i, v in enumerate(value_order)}
        result[f"{value_column}_int"] = result[value_column].map(mapping).where(
            result[value_column].isin(value_order), len(value_order)
        )
        result = result.sort_values(f"{value_column}_int").drop(columns=[f"{value_column}_int"])
    else:
        result = result.sort_values(value_column)

    if method == 'first':
        result = result.drop_duplicates('h3_cell',keep='first')
    elif method == 'last':
        result = result.drop_duplicates('h3_cell',keep='last')
    elif method == 'mean':
        result = result.groupby("h3_cell").agg({value_column: "mean"}).reset_index()
    elif method == 'sum':
        result = result.groupby("h3_cell").agg({value_column: "sum"}).reset_index()
    elif method == 'density':
        if result.crs.is_geographic:
            result = result.to_crs(result.estimate_utm_crs())

        total = geoms[value_column].sum()
        result['density'] = result[value_column] / result.area 
        result = result.groupby("h3_cell").agg({value_column: "mean"}).reset_index()
        result['cell_area'] = result['h3_cell'].apply(lambda x: h3.cell_area(x, unit='m^2'))
        result[value_column] *= result['cell_area']
        result[value_column] *= total/result[value_column].sum()
    
    result = result[[value_column,'h3_cell']]
    if 'geometry' in result.columns:
        result = result.drop(columns='geometry')
        
    return result.dropna().set_index('h3_cell')


def from_raster(
    raster: np.ndarray|str,
    aoi=None,
    resolution:int=10,
    transform: Affine|None=None,
    crs: CRS|None=None,
    method:str = 'density',
    nodata=None,
):
    if isinstance(raster,str):
        raster, transform, crs = raster_utils.read_raster(raster,aoi=aoi,nodata=nodata)
        gdf = raster_utils.vectorize(
            raster_array = raster,
            transform = transform,
            crs = crs,
            aoi = None,
            keep_nodata = False,
            nodata = nodata,
        )
    else:
        if aoi is not None:
            warnings.warn("aoi cropping is not allowed when passing a loaded raster. Pass the raster path.")

        if (transform is None) or (crs is None):
            raise Exception("If inputing a raster array keywords transform and crs are mandatory.")
        
        gdf = raster_utils.vectorize(
            raster_array = raster,
            transform = transform,
            crs = crs,
            aoi = aoi,
            keep_nodata = False,
            nodata = None,
        )

    gdf = gdf.dropna(subset=['value']).reset_index(drop=True)

    df = from_gdf(gdf,resolution=resolution,value_column='value',method=method)

    return df 

def h3_to_gdf(df, h3_column=None):
    df = df.copy()
    if h3_column is not None:
        df = df.set_index(h3_column)

    # --- Validation ---
    # Detect proper validation function (for H3 v3/v4)
    if hasattr(h3, "is_valid_cell"):
        is_valid = h3.is_valid_cell
    elif hasattr(h3, "h3_is_valid"):
        is_valid = h3.h3_is_valid
    else:
        raise AttributeError("Cannot find a valid H3 validation function in the h3 module.")

    df = df[df.index.map(is_valid)]

    # --- Build geometries ---
    def cell_to_polygon(cell):
        boundary = h3.cell_to_boundary(cell)
        return shapely.geometry.Polygon([(lng, lat) for lat, lng in boundary])

    df["geometry"] = df.index.map(cell_to_polygon)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf


# def plot_h3_df(
#     df: pd.DataFrame,
#     value_column: str | None = None,
#     h3_column: str = "h3_cell",
#     ax=None,
#     cmap: str = "viridis",
#     alpha: float = 0.7,
#     legend: bool = True
# ):
#     import geopandas as gpd
#     import matplotlib.pyplot as plt
#     import contextily as cx
#     import shapely.geometry as geom
#     import h3

#     df = df.copy()

#     # Validate
#     if h3_column not in df.columns:
#         raise ValueError(f"DataFrame must contain an '{h3_column}' column with H3 indices.")

#     # Ensure all cells are valid strings
#     df[h3_column] = df[h3_column].astype(str)

#     # ✅ Use the correct validation function for h3>=4
#     df = df[df[h3_column].apply(h3.is_valid_cell)]

#     # Convert each H3 cell to a polygon geometry
#     def cell_to_polygon(cell):
#         boundary = h3.cell_to_boundary(cell)
#         return geom.Polygon([(lng, lat) for lat, lng in boundary])  # note lat/lng order flip

#     df["geometry"] = df[h3_column].apply(cell_to_polygon)

#     # Convert to GeoDataFrame
#     gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
#     gdf = gdf.to_crs(epsg=3857)

#     # Prepare axis
#     if ax is None:
#         _, ax = plt.subplots(figsize=(8, 8))
#     ax.set_axis_off()

#     # Plot polygons
#     gdf.plot(
#         ax=ax,
#         column=value_column,
#         cmap=cmap if value_column else None,
#         alpha=alpha,
#         edgecolor="k",
#         linewidth=0.3,
#         legend=legend if value_column else False,
#         legend_kwds={"loc": "upper left"} if value_column else None,
#     )

#     cx.add_basemap(ax, crs=gdf.crs, source=cx.providers.CartoDB.Positron)

#     return ax