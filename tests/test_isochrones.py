import numpy as np
import polars as pl
import pytest
import shapely

from UrbanAccessAnalyzer import graph_ops, isochrones, osm_io, routing


@pytest.fixture(scope="module")
def simplified_network(sample_pbf_path):
    nodes, edges = osm_io.load_pbf(sample_pbf_path, network_type="all")
    nodes, edges, crs = graph_ops.project(nodes, edges)
    nodes, edges = graph_ops.simplify(nodes, edges, cluster_distance=10.0)
    return nodes, edges


def test_default_distance_matrix_shape_and_zero_handling():
    poi = pl.DataFrame({"poi_score": [1.0, 0.5, 1.0, 0.5]})
    matrix, values = isochrones.default_distance_matrix(poi, [100, 200, 400], poi_score_col="poi_score")

    assert matrix.height == 2  # one row per distinct nonzero poi_score
    assert 0.0 in values
    assert values == sorted(values, reverse=True)


def test_distance_matrix_to_processing_order_from_list_ranks_best_first():
    order = isochrones.distance_matrix_to_processing_order([100, 200, 400], access_score_values=["a", "b", "c"])

    assert set(order.columns) >= {"poi_score", "distance", "access_score", "rank"}
    best = order.filter(pl.col("rank") == 0)
    assert best["access_score"][0] == "a"
    assert best["distance"][0] == 100


def test_graph_isochrone_end_to_end(simplified_network):
    nodes, edges = simplified_network
    xs = nodes["x"].to_numpy()[:3]
    ys = nodes["y"].to_numpy()[:3]
    poi = pl.DataFrame(
        {"geometry": shapely.to_wkb(shapely.points(xs, ys)), "poi_score": [1.0, 0.5, 1.0]}
    )

    matrix, _ = isochrones.default_distance_matrix(poi, [100, 200, 400], poi_score_col="poi_score")
    node_access, edge_access = isochrones.graph(
        nodes, edges, poi, matrix, poi_score_col="poi_score", min_edge_length=1.0, max_dist=100, verbose=False
    )

    assert node_access.height > 0
    assert edge_access.height > 0
    # Unreached network is now *present* and scored 0 (never null/absent), so
    # both tables cover the whole graph and neither carries a null score.
    assert node_access["access_score"].null_count() == 0
    assert edge_access["access_score"].null_count() == 0
    assert (node_access["access_score"] >= 0).all()
    assert (edge_access["access_score"] >= 0).all()
    assert (node_access["access_score"] > 0).any()
    assert (edge_access["access_score"] > 0).any()
    assert (edge_access["length_m"] > 0).all()
    # Every node of the (POI-snapped, hence >= original) graph gets a row.
    assert node_access.height >= nodes.height
    # Essentially every meter of the input network is represented (only
    # sub-`min_edge_length` slivers are dropped).
    assert edge_access["length_m"].sum() > 0.99 * edges["length_m"].sum()

    # Opting out restores the old sparse "reached only" behavior.
    sparse_nodes, sparse_edges = isochrones.graph(
        nodes, edges, poi, matrix, poi_score_col="poi_score", min_edge_length=1.0, max_dist=100,
        verbose=False, unreached_access_score=None,
    )
    assert (sparse_nodes["access_score"] > 0).all()
    assert (sparse_edges["access_score"] > 0).all()
    assert sparse_edges.height < edge_access.height


def test_graph_isochrone_respects_radius(simplified_network):
    nodes, edges = simplified_network
    source_pt = shapely.points([nodes["x"][0]], [nodes["y"][0]])
    poi = pl.DataFrame({"geometry": shapely.to_wkb(source_pt)})

    node_access, _ = isochrones.graph(
        nodes, edges, poi, [50], poi_score_col=None, min_edge_length=1.0, max_dist=100, verbose=False
    )
    # remaining_dist must never exceed the tier's own radius (50m)
    assert (node_access["remaining_dist"] <= 50).all()


def test_compute_node_access_hoisted_csr_matches_per_tier_build(simplified_network):
    """Regression test: hoisting routing.build_csr out of the per-tier loop in
    compute_node_access must not change results. Build a small multi-tier
    process_order (several distinct access_score/distance combinations) so
    the per-tier loop actually runs more than once, then compare the current
    (hoisted-CSR) implementation's output against a manual "rebuild CSR every
    tier" reimplementation of the old behavior.
    """
    nodes, edges = simplified_network
    xs = nodes["x"].to_numpy()[:4]
    ys = nodes["y"].to_numpy()[:4]
    poi = pl.DataFrame(
        {"geometry": shapely.to_wkb(shapely.points(xs, ys)), "poi_score": [1.0, 0.75, 0.5, 0.25]}
    )
    matrix, _ = isochrones.default_distance_matrix(poi, [100, 300, 600], poi_score_col="poi_score")
    process_order = isochrones.distance_matrix_to_processing_order(matrix)
    assert process_order["rank"].n_unique() > 1  # sanity: multiple tiers

    point_geoms = shapely.from_wkb(poi["geometry"].to_numpy())
    snapped_nodes, snapped_edges, snapped_ids = graph_ops.snap_points(
        nodes, edges, point_geoms, max_dist=100, min_edge_length=1.0
    )
    points = poi.with_columns(pl.Series("node_id", snapped_ids))

    # Current (hoisted CSR) implementation.
    node_access, tier_tables = isochrones.compute_node_access(
        snapped_nodes, snapped_edges, points, process_order, poi_score_col="poi_score", verbose=False
    )

    # Old behavior: rebuild the CSR fresh inside the loop for every tier.
    old_tier_tables: dict[float, pl.DataFrame] = {}
    for row in process_order.iter_rows(named=True):
        score_group = row["poi_score"] if isinstance(row["poi_score"], list) else [row["poi_score"]]
        source_ids = (
            points.filter(pl.col("poi_score").is_in(score_group))["node_id"].drop_nulls().unique().to_list()
        )
        if not source_ids:
            continue
        reach = routing.multi_source_distances(
            snapped_nodes, snapped_edges, source_ids, radius=row["distance"], directed=False
        )
        if reach.is_empty():
            continue
        reach = reach.select("node_id", (row["distance"] - pl.col("dist")).alias("remaining_dist"))
        existing = old_tier_tables.get(row["access_score"])
        old_tier_tables[row["access_score"]] = (
            reach if existing is None
            else pl.concat([existing, reach]).group_by("node_id").agg(pl.col("remaining_dist").max())
        )

    assert set(tier_tables.keys()) == set(old_tier_tables.keys())
    for score, new_table in tier_tables.items():
        old_table = old_tier_tables[score].sort("node_id")
        new_table = new_table.sort("node_id")
        assert new_table["node_id"].to_list() == old_table["node_id"].to_list()
        np.testing.assert_allclose(
            new_table["remaining_dist"].to_numpy(), old_table["remaining_dist"].to_numpy()
        )


def test_default_distance_matrix_score_bins_none_matches_current_default():
    poi = pl.DataFrame({"poi_score": [0.1, 0.33, 0.5, 0.75, 0.9, 1.0]})
    matrix_default, values_default = isochrones.default_distance_matrix(poi, [100, 300], poi_score_col="poi_score")
    matrix_explicit_none, values_explicit_none = isochrones.default_distance_matrix(
        poi, [100, 300], poi_score_col="poi_score", score_bins=None
    )
    assert matrix_default.equals(matrix_explicit_none)
    assert values_default == values_explicit_none


def test_default_distance_matrix_score_bins_reduces_tier_count_within_tolerance():
    rng = np.random.default_rng(0)
    continuous_scores = rng.uniform(0.01, 1.0, size=200)
    poi = pl.DataFrame({"poi_score": continuous_scores})

    matrix_full, _ = isochrones.default_distance_matrix(poi, [200, 400, 800], poi_score_col="poi_score")
    order_full = isochrones.distance_matrix_to_processing_order(matrix_full)
    n_tiers_full = order_full["rank"].n_unique()

    matrix_binned, _ = isochrones.default_distance_matrix(
        poi, [200, 400, 800], poi_score_col="poi_score", score_bins=10
    )
    order_binned = isochrones.distance_matrix_to_processing_order(matrix_binned)
    n_tiers_binned = order_binned["rank"].n_unique()

    # Binning should meaningfully shrink the tier count...
    assert n_tiers_binned < n_tiers_full
    # ...but the binned matrix's poi_score column must still contain every
    # original distinct raw score value (so it joins correctly against POIs).
    assert set(matrix_binned["poi_score"].to_list()) == set(matrix_full["poi_score"].to_list())

    # And per-poi_score access scores should be a "sane" approximation of the
    # unbinned values: same [0, 1] range, and rank-order preserving (higher
    # poi_score never gets a worse binned access_score than a lower
    # poi_score), even though bucketing changes the absolute scaling of the
    # rank->score formula (fewer ranks compress the whole range differently).
    joined = matrix_full.join(matrix_binned, on="poi_score", suffix="_binned").sort("poi_score")
    for dist in ["200", "400", "800"]:
        binned_vals = joined[f"{dist}_binned"].to_numpy()
        assert (binned_vals >= 0).all() and (binned_vals <= 1).all()
        # non-decreasing as poi_score increases (ties allowed within a bucket)
        assert (np.diff(binned_vals) >= -1e-9).all()


def test_exact_edge_access_schema_stable_with_long_runs_of_one_sided_rows():
    """Regression test for a polars.ComputeError seen historically:
    `pl.DataFrame(list[dict])` infers a column's dtype from (by default) only
    the first `infer_schema_length` rows. exact_edge_access emits rows with
    `v=None` for u-side pieces and `u=None` for v-side pieces; if a tier
    produces a long run (>100) of edges that only ever get a u-side piece
    (e.g. because they're all near one endpoint), the `v` column can look
    all-null to polars' inferencer and then raise a ComputeError the moment a
    real int `v` value shows up later. Build exactly that shape directly
    against tier_tables/process_order (bypassing routing) and confirm it no
    longer raises and produces a fully-typed, non-null-typed schema.
    """
    n_edges = 150
    # A long line per edge so `shapely.ops.substring` has room to cut.
    edges = pl.DataFrame(
        {
            "u": list(range(n_edges)),
            "v": list(range(n_edges, 2 * n_edges)),
            "length_m": [10.0] * n_edges,
            "geometry_wkb": [shapely.to_wkb(shapely.LineString([(i, 0), (i + 10, 0)])) for i in range(n_edges)],
        }
    )
    process_order = pl.DataFrame({"access_score": [1.0], "rank": [0]})
    # Every u endpoint reachable (remaining_dist small so only the u side gets
    # a piece), no v endpoints present at all -> a long run of v=None rows.
    tier_tables = {1.0: pl.DataFrame({"node_id": list(range(n_edges)), "remaining_dist": [2.0] * n_edges})}

    result = isochrones.exact_edge_access(
        edges, process_order, tier_tables, min_edge_length=0.0, unreached_access_score=None
    )

    assert result.height == n_edges
    assert result.schema["v"] != pl.Null
    assert result["v"].is_null().all()
    assert result["u"].is_null().sum() == 0

    # With the default `unreached_access_score=0.0`, the 8m of each 10m edge
    # that no tier reached comes back too, scored 0 rather than dropped.
    dense = isochrones.exact_edge_access(edges, process_order, tier_tables, min_edge_length=0.0)
    assert dense.height == 2 * n_edges
    assert dense.filter(pl.col("access_score") == 0.0).height == n_edges
    np.testing.assert_allclose(dense["length_m"].sum(), edges["length_m"].sum())


def test_buffers_produces_nonoverlapping_tiers():
    poi = pl.DataFrame(
        {
            "geometry": shapely.to_wkb(shapely.points([0.0, 0.0], [0.0, 0.0])),
            "poi_score": [1.0, 0.5],
        }
    )
    matrix, _ = isochrones.default_distance_matrix(poi, [100, 300], poi_score_col="poi_score")
    result = isochrones.buffers(poi, matrix, poi_score_col="poi_score")

    assert result.height > 0
    assert set(result.columns) >= {"access_score", "geometry_wkb"}
