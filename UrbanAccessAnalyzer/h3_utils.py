import geopandas as gpd
import pandas as pd
import numpy as np
import h3
from shapely.geometry import Polygon, MultiPolygon
import warnings
import shapely 
from rasterio.transform import Affine
from rasterio.crs import CRS
from . import raster_utils

"""TODO: using h3 h3shape_to_cells_experimental function so rasterize might break"""

def from_gdf(
    geoms: gpd.GeoDataFrame,
    resolution: int,
    value_column: str | None = None,
    value_order: list | str = 'max',
    buffer: float = 0,
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

    # Infer order if not provided
    if isinstance(value_order,list):
        categorical = True 
    else:
        order = list(np.unique(geoms[value_column]))
        if len(order) > (len(geoms) / 10):
            categorical = False 
        else:
            categorical = True
            if value_order == 'min':
                value_order = list(np.unique(geoms[value_column]))
                value_order.reverse()
            elif value_order == 'max':
                value_order = list(np.unique(geoms[value_column]))
            elif value_order is None:
                value_order = order
            else:
                raise Exception(f"Keyword value_order {value_order} is not valid.")

    if categorical == True:
        # Filter out any unexpected values not in value_order
        geoms = geoms[geoms[value_column].isin(value_order)]

    if categorical:
        # Prepare results container
        result = pd.DataFrame({value_column: value_order, "h3_cell": None})
        # Loop over each value and compute union + H3 cells
        for val in value_order:
            subset = geoms[geoms[value_column] == val]
            if subset.empty:
                continue

            # Merge all geometries into one
            union_geom = subset.geometry.union_all()
            # Apply buffering if requested
            if buffer > 0:
                union_geom = shapely.buffer(union_geom,buffer,quad_segs=2)
                union_geom = gpd.GeoSeries([union_geom],crs=geoms.crs).to_crs(4326).union_all()
                
            if union_geom.is_empty:
                continue

            # Handle both single and multi geometries
            if isinstance(union_geom, (Polygon, MultiPolygon)):
                h3_cells = h3.h3shape_to_cells_experimental(h3.geo_to_h3shape(union_geom), res=resolution, contain='overlap')
            else:
                warnings.warn(f"Skipping a geometry that is not Polygon. Got: {type(union_geom)}")
                continue

            # Assign H3 cell list to result
            result.loc[result[value_column] == val, "h3_cell"] = [list(h3_cells)]
    else:
        result = geoms[[value_column,'geometry']].copy()
        if value_order is None:
            value_order == 'max' 

        if value_order == 'max':
            result = result.sort_values(value_column,ascending=False)
        elif value_order == 'min':
            result = result.sort_values(value_column,ascending=True)
        else:
            raise Exception(f"Variable value_order {value_order} not valid for continuous data. Choose between 'max' or 'min'.")

        result = result.reset_index(drop=True)

        if buffer > 0:
            result = result.to_crs(result.estimate_utm_crs())
            result.geometry = result.geometry.buffer(buffer)

        result = result.to_crs(4326)

        result['h3_cell'] = result.apply(
            lambda row: h3.h3shape_to_cells_experimental(
                h3.geo_to_h3shape(row['geometry']),
                res=resolution,
                contain='overlap'
            ),
            axis=1
        )
        result = result.drop(columns=['geometry'])

    result = result.explode('h3_cell')
    result = result.reset_index(drop=True).dropna()
    result = result.drop_duplicates('h3_cell',keep='first')

    return result


def from_raster(
    raster: np.ndarray|str,
    aoi=None,
    resolution:int=10,
    transform: Affine|None=None,
    crs: CRS|None=None,
):
    if isinstance(raster,str):
        raster, transform, crs = raster_utils.read_raster(raster,aoi=aoi)
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
        min_value = 0,
        keep_nodata = False,
        nodata = 0,
    )
    gdf = gdf.loc[gdf['value'] > 0]
    gdf = gdf.rename(columns={'value':'population'})
    return from_gdf(gdf,resolution=resolution,value_column='population',)

def h3_to_gdf(df,h3_column='h3_cell'):
    df = df.copy()
    # --- Validation ---
    if h3_column not in df.columns:
        raise ValueError(f"DataFrame must contain an '{h3_column}' column with H3 indices.")

    df[h3_column] = df[h3_column].astype(str)

    # Detect proper validation function (for H3 v3/v4)
    if hasattr(h3, "is_valid_cell"):
        is_valid = h3.is_valid_cell
    elif hasattr(h3, "h3_is_valid"):
        is_valid = h3.h3_is_valid
    else:
        raise AttributeError("Cannot find a valid H3 validation function in the h3 module.")

    df = df[df[h3_column].apply(is_valid)]

    # --- Build geometries ---
    def cell_to_polygon(cell):
        boundary = h3.cell_to_boundary(cell)
        return shapely.geometry.Polygon([(lng, lat) for lat, lng in boundary])

    df["geometry"] = df[h3_column].apply(cell_to_polygon)
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