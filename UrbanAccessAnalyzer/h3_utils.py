import geopandas as gpd
import pandas as pd
import numpy as np
import h3
from shapely.geometry import Polygon, MultiPolygon
import warnings
import shapely 

"""TODO: using h3 h3shape_to_cells_experimental function so rasterize might break"""

def rasterize(
    geoms: gpd.GeoDataFrame,
    resolution: int,
    value_column: str | None = None,
    value_order: list | None = None,
    buffer: float = 0
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
    >>> rasterize(gdf, resolution=6, value_column='class')
    """

    # Copy to avoid modifying original GeoDataFrame
    geoms = geoms.copy()
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

    # Infer order if not provided
    if value_order is None:
        value_order = list(np.unique(geoms[value_column]))

    # Filter out any unexpected values not in value_order
    geoms = geoms[geoms[value_column].isin(value_order)]

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

    result = result.explode('h3_cell')
    result = result.reset_index(drop=True).dropna()
    result = result.drop_duplicates('h3_cell',keep='first')

    return result
