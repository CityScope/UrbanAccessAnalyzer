import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
import string 


def condense_rows(
    df: pd.DataFrame, row_values: list | None = None, columns: list | None = None
):
    """
    Condense multiple columns of a dataframe into a single 'service_type' column.

    If row_values and columns are provided, the first items in row_values are given priority.
    If multiple values exist on a row, the highest-priority value (based on order in row_values) is selected.

    Parameters:
        df (pd.DataFrame): Input DataFrame (GeoDataFrame or regular).
        row_values (list): Optional list of values in priority order.
        columns (list): Optional list of columns to condense.

    Returns:
        List: A list of condensed 'service_type' values, same length as the input.
    """
    df = df.copy()

    # Determine relevant columns (excluding 'geometry' if exists)
    data_columns = list(df.columns)
    if "geometry" in data_columns:
        data_columns.remove("geometry")

    if columns is not None:
        columns = [col for col in columns if col in data_columns]
    else:
        columns = data_columns

    # Initialize output
    service_type = [None] * len(df)

    if row_values is not None:
        # Priority-based assignment
        for val in row_values:
            mask = df[columns].eq(val).any(axis=1) & pd.isna(service_type)
            for idx in df[mask].index:
                matching_cols = df.loc[idx, columns] == val
                if matching_cols.any():
                    service_type[idx] = val
    else:
        # No priority, just take the first non-null value row-wise
        for idx in df.index:
            for col in columns:
                val = df.at[idx, col]
                if pd.notna(val):
                    service_type[idx] = val
                    break

    return service_type


def quality_by_values(values: list | pd.Series, value_priority: list):
    # Convert values to list while preserving None/np.nan
    values = pd.Series(values)
    str_values = values.astype(str).where(~values.isna(), None)

    if len(value_priority) == 0:
        raise Exception("No values in value_priority")

    if len(values) == 0:
        return []

    # Check for duplicates in priority list
    if len(set(value_priority)) != len(value_priority):
        raise Exception("value_priority has to have unique values")

    # Check for None or NaN in priority list
    if any(pd.isna(x) for x in value_priority):
        raise Exception("value_priority has None or NaN")

    # Convert priority list to strings for matching
    value_priority_str = [str(x) for x in value_priority]

    # Warn about mismatches
    unique_values = set(str_values.dropna().unique())
    not_in_values = set(value_priority_str) - unique_values
    if not_in_values:
        warnings.warn(
            f"Values {not_in_values} in value_priority are not in the input values"
        )

    not_in_priority = unique_values - set(value_priority_str)
    if not_in_priority:
        warnings.warn(
            f"Values {not_in_priority} in input values are not in value_priority. They will be set to None."
        )

    # Create mapping from value to priority index (starting at 1)
    value_to_priority = {val: i + 1 for i, val in enumerate(value_priority_str)}

    # Map values to priorities
    result = str_values.map(value_to_priority)

    return result.tolist()


def quality_by_area(gdf: gpd.GeoDataFrame | gpd.GeoSeries, area_steps: list[float], large_is_better:bool=True):
    """
    Classify geometries by area thresholds defined in area_steps.

    Parameters:
        gdf (GeoDataFrame or GeoSeries): Input geometries.
        area_steps (list): List of area breakpoints. Must be sorted (asc or desc).

    Returns:
        list: A list of class numbers corresponding to area bins.
    """
    # Create the area column
    gdf = gdf.copy()
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    gdf['area'] = gdf.geometry.area 
    area_steps = np.unique(area_steps)

    for i in range(len(area_steps)):
        j = len(area_steps) - i - 1
        if large_is_better:
            gdf.loc[gdf.geometry.area > area_steps[i], '_service_quality'] = j + 1
        else:
            gdf.loc[gdf.geometry.area > area_steps[j], '_service_quality'] = i + 1
            
    return list(gdf['_service_quality'])


def polygons_to_points(poi, street_edges):
    poi_points = poi.copy()
    if not (poi.geometry.type == 'Point').all():
        poi_points['poi_id'] = poi.index 
        polygons_bool = (
            (
                poi.geometry.type == 'Polygon'
            ) | (
                poi.geometry.type == 'MultiPolygon'
            )
        )
        poi_points.loc[polygons_bool,'geometry'] = poi_points[polygons_bool].geometry.boundary
        points_bool = (
            (
                poi.geometry.type == 'Point'
            ) | (
                poi.geometry.type == 'MultiPoint'
            )
        )
        poi_points.loc[~points_bool,'geometry'] = poi_points[~points_bool].geometry.intersection(street_edges.union_all())
        poi_points = poi_points[poi_points.geometry.is_empty == False] 

    poi_points = poi_points.explode().reset_index(drop=True)
    return poi_points