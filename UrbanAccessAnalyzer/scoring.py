import numpy as np
from itertools import product
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    TypeVar,
    Tuple,
    Union,
)

# Type variables
T = TypeVar("T")  # Generic input type (interval, route_type, etc.)


def build_adaptive_grids(
    func: callable,
    variables: List[Union[List[float], tuple, np.ndarray]],
    delta: float = 0.1,
    max_iters: int = 30
) -> List[np.ndarray]:
    """
    Build adaptive grids for multiple variables to guarantee that stepping along
    any continuous variable changes the function by at most `delta`.

    Each variable can be:
        - Continuous: [min, max] or (min, max)
        - Pre-discrete: list or np.ndarray of values

    Args:
        func: Callable supporting broadcasting, e.g., func(var1, var2, ...)
        variables: List of variable specifications
        delta: Max allowed change between adjacent points in any continuous variable
        max_iters: Max number of refinement iterations

    Returns:
        List of np.ndarray grids for each variable
    """
    n_vars = len(variables)

    # Initialize grids and detect discrete variables
    grids = []
    is_discrete = []
    for var in variables:
        if isinstance(var, (list, tuple)) and len(var) == 2 and all(isinstance(x, (int, float)) for x in var):
            grids.append(np.array([var[0], var[1]], dtype=float))  # continuous
            is_discrete.append(False)
        else:
            arr = np.array(var)
            grids.append(arr)
            is_discrete.append(True)

    for _ in range(max_iters):
        changed_any = False

        for i, (grid, discrete) in enumerate(zip(grids, is_discrete)):
            if discrete:
                continue  # skip refinement for pre-discrete variables

            # --- Broadcast all variables safely ---
            broadcast_vars = []
            for j, g in enumerate(grids):
                shape = [1] * n_vars
                shape[j] = len(g)
                broadcast_vars.append(np.reshape(g, shape))

            # Evaluate function on full grid
            q = func(*broadcast_vars)

            # Compute differences along the axis of current variable
            dq = np.abs(np.diff(q, axis=i))

            # Collapse all other axes to find the worst-case delta
            worst_dq = dq.max(axis=tuple(k for k in range(n_vars) if k != i))

            # Identify intervals that exceed delta
            bad = worst_dq > delta
            if not np.any(bad):
                continue

            # Insert midpoints where needed
            mids = 0.5 * (grid[:-1][bad] + grid[1:][bad])
            new_grid = np.sort(np.unique(np.concatenate([grid, mids])))
            grids[i] = new_grid
            changed_any = True

        if not changed_any:
            break

    return grids


def elasticity_from_linear_decay(decay, point):
    return -abs(decay) * point / (1 - abs(decay) * point)

def elasticity_based_score(
    value: Union[float, List[float], np.ndarray],
    reference: float,
    elasticity: Union[float, Callable[[float], float], List[Sequence[float]]],
    steps: int = 200,
) -> Union[float, np.ndarray]:
    """
    Compute score using elasticity-based integration, vectorized over values.

    Parameters
    ----------
    value : float or array-like
        The current value(s) of the variable.
    reference : float
        Reference value (e.g., baseline).
    elasticity : float, callable, or list of [lower_bound, elasticity]
        - float: constant elasticity (analytic solution)
        - callable: function of x returning elasticity (numerical integration)
        - list/tuple: piecewise elasticity [[lower_bound, e], ...] (analytic per segment)
    steps : int, default=200
        Number of steps for numerical integration (used only for callable elasticity).

    Returns
    -------
    float or np.ndarray
        Score value(s) in (0, 1], decreasing as value moves away from reference.
    """
    # Ensure array
    values = np.atleast_1d(value).astype(float)
    q = np.ones_like(values, dtype=float)

    mask = values != reference
    if not np.any(mask):
        return q if np.ndim(value) > 0 else q[0]

    v = values[mask]

    # --- Case 1: constant elasticity (float) ---
    if isinstance(elasticity, (int, float)):
        q[mask] = (v / reference) ** elasticity
        return q if np.ndim(value) > 0 else q[0]

    # --- Case 2: piecewise list/tuple ---
    elif isinstance(elasticity, (list, tuple)):
        processed = np.array([(-np.inf if lb is None else lb, e) for lb, e in elasticity])
        processed = processed[np.argsort(processed[:,0])]  # sort by lower_bound
        lbs = processed[:,0]
        es = processed[:,1]

        result = np.empty_like(v)
        for i, val in enumerate(v):
            # Find which piece each val falls into
            idx = np.searchsorted(lbs, val, side='right') - 1
            e = es[idx]
            # analytic integral: ∫(e/x) dx = e * ln(value/reference)
            result[i] = (val / reference) ** e
        q[mask] = result
        return q if np.ndim(value) > 0 else q[0]

    # --- Case 3: callable ---
    elif callable(elasticity):
        def elasticity_fn(x):
            return elasticity(x)
        xs = np.linspace(reference, v[:, None], steps)  # shape (len(v), steps)
        e_vals = np.vectorize(elasticity_fn)(xs)
        integrand = e_vals / xs
        integral = np.trapezoid(integrand, xs, axis=1)
        q[mask] = np.exp(integral)
        return q if np.ndim(value) > 0 else q[0]

    else:
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
    """
    Normalize a multi-parameter score function to a given range.
    Supports vectorized score functions that can take lists or numpy arrays.

    Parameters
    ----------
    score_func : callable
        Function accepting positional arguments (e.g., interval, route_type, speed, distance).
        Can return scalar or np.ndarray if inputs are arrays/lists.
    min_score : float, default=0.1
        Minimum normalized score.
    max_score : float, default=1.0
        Maximum normalized score.
    min_point : sequence, optional
        Explicit point to define minimum score.
    max_point : sequence, optional
        Explicit point to define maximum score.
    variable_steps : list of iterables, optional
        Steps for each argument to generate combinations if min/max points are not provided.

    Returns
    -------
    callable
        Function with same arguments as `score_func` that returns normalized score.
        Preserves vectorized behavior.
    """
    # --- Build combinations for calibration ---
    combinations: List[Sequence[Any]] = []

    if (min_point is None or max_point is None) and variable_steps is not None:
        steps = [
            sorted(step) if isinstance(step, (list, tuple, np.ndarray)) else [step]
            for step in variable_steps
        ]
        combinations.extend(product(*steps))

    if min_point is not None:
        combinations.append(min_point)
    if max_point is not None:
        combinations.append(max_point)

    if not combinations:
        raise ValueError("No points provided to compute score range.")

    # Evaluate qualities for all combinations
    qualities_list = []
    for c in combinations:
        res = score_func(*c)
        if isinstance(res, np.ndarray):
            qualities_list.extend(res.flatten())
        else:
            qualities_list.append(res)
    qualities_array = np.array(qualities_list, dtype=float)

    # Filter out zeros if needed
    nonzero_qualities = qualities_array[qualities_array != 0]
    if min_score > 0 and len(nonzero_qualities) == 0:
        raise Exception("All qualities returned by score_func are 0.")

    # Determine calibration min/max
    q_min = float(score_func(*min_point)) if min_point is not None else np.min(nonzero_qualities)
    q_max = float(score_func(*max_point)) if max_point is not None else np.max(nonzero_qualities)

    if q_max == q_min:
        raise ValueError("q_min and q_max are equal; cannot normalize")

    # --- Vectorized access function ---
    def access_score(*args: T) -> Union[float, np.ndarray]:
        x = score_func(*args)
        x_arr = np.atleast_1d(x).astype(float)
        normalized = min_score + (x_arr - q_min) * (max_score - min_score) / (q_max - q_min)
        if np.isscalar(x) or x_arr.size == 1:
            if x_arr.size == 1:
                return float(normalized[0])
            else:
                return float(normalized)

        else:
            return normalized

    return access_score