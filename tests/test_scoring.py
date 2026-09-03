import numpy as np
import pytest

from UrbanAccessAnalyzer import scoring


def test_score_by_values_best_gets_one():
    values = ["b", "a", "c"]
    result = scoring.score_by_values(values, value_priority=["a", "b", "c"])
    assert result[1] == 1.0  # "a" is first in priority -> best -> score 1.0
    assert max(result) == 1.0


def test_score_by_values_unmatched_defaults_to_zero_not_nan():
    # "d" is not in value_priority -> higher-is-better score, so unmatched -> 0.0, never NaN.
    values = ["a", "d"]
    result = scoring.score_by_values(values, value_priority=["a", "b", "c"])
    assert result[1] == 0.0
    assert not any(np.isnan(r) for r in result)


def test_score_by_area_below_smallest_threshold_defaults_to_zero():
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(0, 0, 100, 100)], crs="EPSG:32633")  # areas 1 m² and 10000 m²
    result = scoring.score_by_area(gdf, area_steps=[500, 5000], large_is_better=True)

    assert result[0] == 0.0  # 1 m² clears no threshold -> worst score, not NaN
    assert result[1] > result[0]
    assert not any(np.isnan(r) for r in result)


def test_elasticity_based_score_constant_elasticity_monotonic_decay():
    values = [50, 100, 200]
    scores = scoring.elasticity_based_score(values, reference=100, elasticity=-1.0)
    assert scores[1] == pytest.approx(1.0)  # at reference, score == 1
    assert scores[0] > scores[1] > scores[2] or scores[0] < scores[1]  # monotonic away from reference in some direction
    assert np.all(scores > 0)


def test_calibrate_scoring_func_hits_min_and_max():
    def raw(x):
        return x

    calibrated = scoring.calibrate_scoring_func(raw, min_score=0.1, max_score=1.0, min_point=(0,), max_point=(10,))
    assert calibrated(0) == pytest.approx(0.1)
    assert calibrated(10) == pytest.approx(1.0)


def test_build_adaptive_grids_respects_delta():
    def f(x):
        return x**2

    grids = scoring.build_adaptive_grids(f, variables=[[0, 10]], delta=1.0, max_iters=50)
    grid = grids[0]
    diffs = np.abs(np.diff(f(grid)))
    assert diffs.max() <= 1.0 + 1e-6
