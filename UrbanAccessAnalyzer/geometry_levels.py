"""Bridge from UAA's Polars edge results to geohierarchy's any-polygon aggregation.

UAA represents street-network edges (and their per-edge access scores) as a
Polars DataFrame with a WKB geometry column, while the ``geohierarchy``
package works with GeoPandas GeoDataFrames and aggregates line data onto an
arbitrary polygon layer (an H3 grid, administrative boundaries, or any other
polygon ``GeoDataFrame``) via its ``edges_to_level`` helper. This module does
the small amount of glue work in between: converting UAA's Polars/WKB edge
table into a GeoDataFrame (the same conversion :meth:`AccessibilityAnalyzer
.to_h3` already did before this refactor) and handling columns that aren't
plain numeric quantities -- e.g. categorical access-score labels -- which
``geohierarchy``'s aggregation strategies (``Max``/``Min``) don't natively
support since they call Polars numeric-only operations internally. This is an
optional dependency of UAA: ``geohierarchy`` is only imported when a function
in this module is actually called, behind the ``geohierarchy`` extra.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import polars as pl
import shapely

NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


def _import_geohierarchy():
    try:
        from geohierarchy import edges_to_level as gh_edges_to_level
        from geohierarchy.aggregation import Max, Min
    except ImportError as e:
        raise ImportError(
            "Polygon/H3 aggregation requires the 'geohierarchy' extra: "
            "pip install urbanaccessanalyzer[geohierarchy]"
        ) from e
    return gh_edges_to_level, Max, Min


def edges_to_level(
    edges_df: pl.DataFrame,
    level_gdf: gpd.GeoDataFrame,
    id_col: str,
    columns: Union[str, List[str]],
    agg=None,
    geometry_col: str = "geometry",
    crs=None,
) -> gpd.GeoDataFrame:
    """Aggregate columns from a Polars WKB edge table onto a polygon level.

    Converts ``edges_df`` (a Polars DataFrame with a WKB-encoded
    ``geometry_col``, as produced throughout UAA) into a GeoDataFrame, then
    delegates to :func:`geohierarchy.edges_to_level` to length-weight the
    requested ``columns`` onto ``level_gdf``.

    Non-numeric columns (e.g. categorical access-score labels) can't be
    aggregated by ``geohierarchy``'s ``Max``/``Min`` strategies directly,
    since those call Polars numeric-only expressions internally. For any
    column resolved to a ``Max``/``Min`` strategy that isn't already
    numeric, this function encodes it to integer codes ordered the same way
    Polars would sort the raw values (so the numeric max/min matches the
    string max/min UAA previously produced), aggregates, then decodes the
    result back to the original labels.

    Args:
        edges_df: Polars DataFrame with a WKB ``geometry_col`` column plus
            the attribute columns to aggregate.
        level_gdf: Target polygon GeoDataFrame to aggregate onto -- an H3
            grid, an administrative boundary layer, or any other polygon
            layer.
        id_col: Name of the unique id column in ``level_gdf``.
        columns: Column name or list of column names in ``edges_df`` to
            aggregate onto ``level_gdf``.
        agg: Aggregation strategy (a
            :class:`~geohierarchy.aggregation.AggregationStrategy` instance)
            applied to every column, or a per-column
            ``{column: AggregationStrategy}`` mapping. Defaults to
            :class:`~geohierarchy.aggregation.Mean` for every column, per
            :func:`geohierarchy.edges_to_level`.
        geometry_col: Name of the WKB geometry column in ``edges_df``.
        crs: Coordinate reference system the result should use. Defaults to
            ``level_gdf``'s own CRS.

    Returns:
        A copy of ``level_gdf`` with ``id_col``, geometry, and the
        aggregated ``columns`` -- one row per polygon.

    Raises:
        ImportError: If the optional ``geohierarchy`` dependency isn't
            installed.
    """
    gh_edges_to_level, Max, Min = _import_geohierarchy()

    if isinstance(columns, str):
        columns = [columns]

    geoms = shapely.from_wkb(edges_df[geometry_col].to_numpy())
    edges_gdf = gpd.GeoDataFrame(
        edges_df.drop(geometry_col).to_pandas(), geometry=geoms, crs=crs or level_gdf.crs
    )

    agg_by_col: Dict[str, object] = agg if isinstance(agg, dict) else {col: agg for col in columns}

    decoders: Dict[str, np.ndarray] = {}
    for col in columns:
        strategy = agg_by_col.get(col)
        if isinstance(strategy, (Max, Min)) and edges_gdf[col].dtype.kind not in "iuf":
            categories = np.sort(edges_gdf[col].dropna().unique())
            code_map = {v: i for i, v in enumerate(categories)}
            edges_gdf[col] = edges_gdf[col].map(code_map)
            decoders[col] = categories

    result = gh_edges_to_level(
        edges_gdf,
        level_gdf,
        id_col=id_col,
        columns=columns,
        agg=agg,
        geometry_col="geometry",
        crs=crs,
    )

    for col, categories in decoders.items():
        codes = result[col].to_numpy()
        valid = ~np.isnan(codes.astype(float)) if codes.dtype.kind == "f" else np.ones(len(codes), dtype=bool)
        decoded = np.full(len(codes), None, dtype=object)
        rounded = np.where(valid, np.nan_to_num(codes).round().astype(int), 0)
        decoded[valid] = categories[rounded[valid]]
        result[col] = decoded

    return result
