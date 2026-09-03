"""POI quality scoring: from raw attributes to a calibrated [0, 1] access score.

Consolidates three previously separate, duplicated modules: ``scoring.py`` and
``quality.py`` were byte-for-byte near-duplicates (a "POI scoring" vs.
"transit stop quality" naming fork of the same elasticity/calibration code),
and ``poi_utils.py`` internally duplicated ``score_by_values``/
``score_by_area`` under both "quality_*" and "score_*" names. Per the
package's naming convention, everything here is called ``score``/
``access_score`` -- never "quality".

Methodology:

- :func:`score_by_values` and :func:`score_by_area` produce a simple
  ``[0, 1]`` ordinal score from a categorical value or a continuous area,
  respectively, given an explicit priority ordering.
- :func:`elasticity_based_score` computes a continuous decay score using the
  economic notion of elasticity: score(v) = (v / reference) ** elasticity for
  constant elasticity, or the exponential of the integral of
  ``elasticity(x)/x`` for a value- or interval-dependent elasticity (matching
  how price elasticity of demand extends to non-constant elasticities). This
  gives a principled decay curve (e.g. for distance or headway) instead of an
  arbitrary hand-picked shape.
- :func:`build_adaptive_grids` builds a non-uniform grid per input variable
  so that adjacent grid points never differ by more than ``delta`` in a given
  (possibly expensive/non-vectorizable) score function -- used to precompute
  a lookup table instead of recomputing the function per row.
- :func:`calibrate_scoring_func` linearly rescales an arbitrary score
  function's output range to a target ``[min_score, max_score]`` given
  calibration anchor points.
"""

from itertools import product
from typing import Any, Callable, List, Optional, Sequence, TypeVar, Union
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

T = TypeVar("T")


def condense_rows(df: pd.DataFrame, row_values: Optional[list] = None, columns: Optional[list] = None) -> list:
    """Collapse multiple boolean/tag columns into a single value per row.

    Args:
        df: Input DataFrame (attribute columns to condense, e.g. one column
            per OSM tag like ``shop``, ``amenity``).
        row_values: Priority order of values; the first matching value (by
            this order) found in any of ``columns`` wins for each row. If
            ``None``, the first non-null value across ``columns`` is used.
        columns: Columns to consider; defaults to every column except
            ``geometry``.

    Returns:
        List of condensed values, one per row (``None`` where no column
        matched).
    """
    df = df.copy()
    data_columns = [c for c in df.columns if c != "geometry"]
    columns = [c for c in (columns or data_columns) if c in data_columns]

    service_type = [None] * len(df)
    if row_values is not None:
        for val in row_values:
            mask = df[columns].eq(val).any(axis=1) & pd.isna(service_type)
            for idx in df[mask].index:
                if (df.loc[idx, columns] == val).any():
                    service_type[idx] = val
    else:
        for idx in df.index:
            for col in columns:
                if pd.notna(df.at[idx, col]):
                    service_type[idx] = df.at[idx, col]
                    break
    return service_type


def score_by_values(values: Union[list, pd.Series], value_priority: list) -> list:
    """Map categorical values to a ``[0, 1]`` score by explicit priority order.

    Args:
        values: Categorical values to score (e.g. amenity subtype).
        value_priority: Values in priority order, best first; must be
            unique and contain no null.

    Returns:
        List of scores in ``[0, 1]`` (best value -> 1.0), same length as
        ``values``. This is a "higher is better" score, so values not found
        in ``value_priority`` (including null input) score ``0.0`` -- the
        worst possible score, not ``None`` -- consistent with the package's
        convention that non-computed/unreachable values on a bounded 0-1
        "higher is better" scale collapse to 0, while non-computed values on
        an unbounded "lower is better" scale (e.g. a distance) should stay
        ``None``/``null`` instead (representing an effectively infinite,
        i.e. unreached, distance).

    Raises:
        Exception: If ``value_priority`` is empty, has duplicates, or
            contains a null.
    """
    values = pd.Series(values)
    str_values = values.astype(str).where(~values.isna(), None)

    if len(value_priority) == 0:
        raise Exception("No values in value_priority")
    if len(values) == 0:
        return []
    if len(set(value_priority)) != len(value_priority):
        raise Exception("value_priority has to have unique values")
    if any(pd.isna(x) for x in value_priority):
        raise Exception("value_priority has None or NaN")

    value_priority_str = [str(x) for x in value_priority]
    unique_values = set(str_values.dropna().unique())
    not_in_values = set(value_priority_str) - unique_values
    if not_in_values:
        warnings.warn(f"Values {not_in_values} in value_priority are not in the input values")
    not_in_priority = unique_values - set(value_priority_str)
    if not_in_priority:
        warnings.warn(f"Values {not_in_priority} in input values are not in value_priority. They will score 0.0.")

    value_to_priority = {val: i + 1 for i, val in enumerate(value_priority_str)}
    n = len(value_priority_str)
    result = str_values.map(value_to_priority)
    result = (n + 1 - result) / n  # invert so first-priority value -> 1.0, last -> 1/n
    result = result.fillna(0.0)  # unmatched/null input -> worst score, not NaN (see docstring)
    return np.round(result, 3).tolist()


def score_by_area(gdf: Union[gpd.GeoDataFrame, gpd.GeoSeries], area_steps: list[float], large_is_better: bool = True) -> list:
    """Score geometries by which area threshold bin they fall into.

    Args:
        gdf: Input geometries (any CRS; reprojected to local UTM to measure
            area).
        area_steps: Area breakpoints (m²), any order (sorted internally).
        large_is_better: If ``True``, larger area -> higher score.

    Returns:
        List of scores in ``[0, 1]``, one per input geometry. This is a
        "higher is better" score: geometries that don't clear any threshold
        (never assigned a bucket) score ``0.0`` rather than ``None`` -- see
        :func:`score_by_values` for the package's None-vs-0 convention.
    """
    gdf = gdf.copy()
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    gdf["area"] = gdf.geometry.area
    area_steps = np.unique(area_steps)
    gdf["_score"] = np.nan

    for i in range(len(area_steps)):
        j = len(area_steps) - i - 1
        if large_is_better:
            gdf.loc[gdf.geometry.area > area_steps[i], "_score"] = j + 1
        else:
            gdf.loc[gdf.geometry.area > area_steps[j], "_score"] = i + 1

    gdf["_score"] = list(gdf["_score"].max() + 1 - gdf["_score"])
    gdf["_score"] = list(gdf["_score"] / gdf["_score"].max())
    gdf["_score"] = gdf["_score"].fillna(0.0)  # never cleared a threshold -> worst score, not NaN
    return list(gdf["_score"].round(3))


def polygons_to_points(poi: gpd.GeoDataFrame, street_edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert polygon/line POIs to points by intersecting with street edges.

    Points and multipoints pass through unchanged. Polygons are reduced to
    their boundary; everything else is intersected with the union of
    ``street_edges``, which pulls out the point(s) where the POI actually
    touches the walkable network (used before snapping POIs onto a graph).

    Args:
        poi: POI geometries (any mix of Point/Polygon/MultiPolygon/etc.).
        street_edges: Street edge geometries to intersect non-point POIs with.

    Returns:
        GeoDataFrame of Point geometries (exploded, empty intersections
        dropped), with a ``poi_id`` column pointing back to the original
        ``poi`` row index for non-point inputs.
    """
    poi_points = poi.copy()
    if not (poi.geometry.type == "Point").all():
        poi_points["poi_id"] = poi.index
        polygons_bool = poi.geometry.type.isin(["Polygon", "MultiPolygon"])
        poi_points.loc[polygons_bool, "geometry"] = poi_points[polygons_bool].geometry.boundary
        points_bool = poi.geometry.type.isin(["Point", "MultiPoint"])
        poi_points.loc[~points_bool, "geometry"] = poi_points[~points_bool].geometry.intersection(street_edges.union_all())
        poi_points = poi_points[~poi_points.geometry.is_empty]

    return poi_points.explode(index_parts=False).reset_index(drop=True)


def build_adaptive_grids(
    func: Callable, variables: List[Union[List[float], tuple, np.ndarray]], delta: float = 0.1, max_iters: int = 30
) -> List[np.ndarray]:
    """Build non-uniform grids so ``func`` changes by at most ``delta`` between adjacent points.

    Args:
        func: Broadcastable callable, ``func(var1, var2, ...)``.
        variables: Per-variable spec: a continuous ``[min, max]``/``(min, max)``
            pair to refine, or a pre-discretized list/array to leave as-is.
        delta: Maximum allowed change in ``func`` between adjacent grid
            points along any one continuous variable's axis.
        max_iters: Maximum refinement iterations.

    Returns:
        List of 1-D numpy arrays, one refined grid per variable.
    """
    n_vars = len(variables)
    grids, is_discrete = [], []
    for var in variables:
        if isinstance(var, (list, tuple)) and len(var) == 2 and all(isinstance(x, (int, float)) for x in var):
            grids.append(np.array([var[0], var[1]], dtype=float))
            is_discrete.append(False)
        else:
            grids.append(np.array(var))
            is_discrete.append(True)

    for _ in range(max_iters):
        changed_any = False
        for i, (grid, discrete) in enumerate(zip(grids, is_discrete)):
            if discrete:
                continue
            broadcast_vars = []
            for j, g in enumerate(grids):
                shape = [1] * n_vars
                shape[j] = len(g)
                broadcast_vars.append(np.reshape(g, shape))
            q = func(*broadcast_vars)
            dq = np.abs(np.diff(q, axis=i))
            worst_dq = dq.max(axis=tuple(k for k in range(n_vars) if k != i))
            bad = worst_dq > delta
            if not np.any(bad):
                continue
            mids = 0.5 * (grid[:-1][bad] + grid[1:][bad])
            grids[i] = np.sort(np.unique(np.concatenate([grid, mids])))
            changed_any = True
        if not changed_any:
            break
    return grids


def elasticity_from_linear_decay(decay: float, point: float) -> float:
    """Convert a linear decay rate at a reference point into an equivalent elasticity.

    Args:
        decay: Linear decay rate (fraction lost per unit of ``point``).
        point: The point at which ``decay`` applies.

    Returns:
        The elasticity value ``e`` such that a constant-elasticity curve
        matches the given linear decay at ``point``.
    """
    return -abs(decay) * point / (1 - abs(decay) * point)


def elasticity_based_score(
    value: Union[float, List[float], np.ndarray],
    reference: float,
    elasticity: Union[float, Callable[[float], float], List[Sequence[float]]],
    steps: int = 200,
) -> Union[float, np.ndarray]:
    """Compute a decay score via elasticity-based integration, vectorized.

    Args:
        value: Current value(s) of the variable (e.g. distance, headway).
        reference: Reference value where score == 1.0.
        elasticity: Constant elasticity (float, analytic solution), a
            piecewise elasticity ``[[lower_bound, e], ...]`` (analytic per
            segment), or a callable ``e(x)`` (numerically integrated).
        steps: Integration steps, used only for the callable case.

    Returns:
        Score(s) in ``(0, 1]``, decreasing as ``value`` moves away from
        ``reference``.

    Raises:
        TypeError: If ``elasticity`` is not a float, callable, or piecewise list.
    """
    values = np.atleast_1d(value).astype(float)
    q = np.ones_like(values, dtype=float)
    mask = values != reference
    if not np.any(mask):
        return q if np.ndim(value) > 0 else q[0]
    v = values[mask]

    if isinstance(elasticity, (int, float)):
        q[mask] = (v / reference) ** elasticity
        return q if np.ndim(value) > 0 else q[0]

    if isinstance(elasticity, (list, tuple)):
        processed = np.array([(-np.inf if lb is None else lb, e) for lb, e in elasticity])
        processed = processed[np.argsort(processed[:, 0])]
        lbs, es = processed[:, 0], processed[:, 1]
        result = np.empty_like(v)
        for i, val in enumerate(v):
            idx = np.searchsorted(lbs, val, side="right") - 1
            result[i] = (val / reference) ** es[idx]
        q[mask] = result
        return q if np.ndim(value) > 0 else q[0]

    if callable(elasticity):
        xs = np.linspace(reference, v[:, None], steps)
        e_vals = np.vectorize(elasticity)(xs)
        integral = np.trapezoid(e_vals / xs, xs, axis=1)
        q[mask] = np.exp(integral)
        return q if np.ndim(value) > 0 else q[0]

    raise TypeError("elasticity must be float, callable, or piecewise list")


def calibrate_scoring_func(
    score_func: Callable[..., Union[float, np.ndarray]],
    *,
    min_score: float = 0.1,
    max_score: float = 1.0,
    min_point: Optional[Sequence[T]] = None,
    max_point: Optional[Sequence[T]] = None,
    variable_steps: Optional[List[Any]] = None,
) -> Callable[..., Union[float, np.ndarray]]:
    """Linearly rescale a multi-parameter score function to ``[min_score, max_score]``.

    Args:
        score_func: Function accepting positional arguments; may return a
            scalar or vectorized ``np.ndarray``.
        min_score: Output value at the calibration minimum.
        max_score: Output value at the calibration maximum.
        min_point: Explicit argument tuple defining the minimum; if omitted,
            inferred as the minimum over ``variable_steps`` combinations.
        max_point: Explicit argument tuple defining the maximum; if omitted,
            inferred as the maximum over ``variable_steps`` combinations.
        variable_steps: Per-argument value lists used to search for the
            calibration min/max when ``min_point``/``max_point`` aren't given.

    Returns:
        A new function with ``score_func``'s signature returning the
        rescaled score.

    Raises:
        ValueError: If no calibration points can be determined, or if the
            calibration min and max coincide.
        Exception: If every evaluated score is zero while ``min_score > 0``.
    """
    combinations: List[Sequence[Any]] = []
    if (min_point is None or max_point is None) and variable_steps is not None:
        steps = [sorted(s) if isinstance(s, (list, tuple, np.ndarray)) else [s] for s in variable_steps]
        combinations.extend(product(*steps))
    if min_point is not None:
        combinations.append(min_point)
    if max_point is not None:
        combinations.append(max_point)
    if not combinations:
        raise ValueError("No points provided to compute score range.")

    scores_list = []
    for c in combinations:
        res = score_func(*c)
        scores_list.extend(res.flatten() if isinstance(res, np.ndarray) else [res])
    scores_array = np.array(scores_list, dtype=float)

    nonzero_scores = scores_array[scores_array != 0]
    if min_score > 0 and len(nonzero_scores) == 0:
        raise Exception("All scores returned by score_func are 0.")

    q_min = float(score_func(*min_point)) if min_point is not None else np.min(nonzero_scores)
    q_max = float(score_func(*max_point)) if max_point is not None else np.max(nonzero_scores)
    if q_max == q_min:
        raise ValueError("q_min and q_max are equal; cannot normalize")

    def access_score(*args: T) -> Union[float, np.ndarray]:
        x = score_func(*args)
        x_arr = np.atleast_1d(x).astype(float)
        normalized = min_score + (x_arr - q_min) * (max_score - min_score) / (q_max - q_min)
        if np.isscalar(x) or x_arr.size == 1:
            return float(normalized[0]) if x_arr.size == 1 else float(normalized)
        return normalized

    return access_score
