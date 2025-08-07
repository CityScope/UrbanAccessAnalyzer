import pandas as pd
import geopandas as gpd
import numpy as np
import warnings


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


def by_values(values: list | pd.Series, value_priority: list):
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


def by_area(gdf: gpd.GeoDataFrame | gpd.GeoSeries, area_steps: list):
    """
    Classify geometries by area thresholds defined in area_steps.

    Parameters:
        gdf (GeoDataFrame or GeoSeries): Input geometries.
        area_steps (list): List of area breakpoints. Must be sorted (asc or desc).

    Returns:
        list: A list of class numbers corresponding to area bins.
    """
    if len(area_steps) == 0:
        raise Exception("No values in area_steps")

    if len(gdf) == 0:
        return []

    # Convert to GeoSeries if needed
    if isinstance(gdf, gpd.GeoDataFrame):
        gdf = gdf.copy()
    else:
        gdf = gpd.GeoDataFrame({}, geometry=gdf.copy(), crs=gdf.crs)

    # Project if not projected
    if not gdf.crs.is_projected:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())

    # Check sort order
    ascending = area_steps == sorted(area_steps)
    descending = area_steps == sorted(area_steps, reverse=True)

    if not ascending and not descending:
        warnings.warn("area_steps is not sorted.")

    # Build class labels
    if descending:
        classes = np.arange(len(area_steps), 0, -1)
    else:
        classes = np.arange(1, len(area_steps) + 1)

    gdf["class"] = None

    areas = gdf.area

    # Assign lowest class
    gdf.loc[areas < area_steps[0], "class"] = classes[0]

    # Assign highest class
    gdf.loc[areas >= area_steps[-1], "class"] = classes[-1]

    # Assign intermediate classes
    for i in range(len(area_steps) - 1):
        mask = (areas >= area_steps[i]) & (areas < area_steps[i + 1])
        gdf.loc[mask, "class"] = classes[i + 1]

    return gdf["class"].tolist()
