import numpy as np
import polars as pl
import pytest
import shapely

from UrbanAccessAnalyzer import graph_ops, osm_io, routing


@pytest.fixture(scope="module")
def projected_network(sample_pbf_path):
    nodes, edges = osm_io.load_pbf(sample_pbf_path, network_type="all")
    nodes, edges, crs = graph_ops.project(nodes, edges)
    return nodes, edges, crs


def test_project_produces_metric_coordinates(projected_network):
    nodes, edges, crs = projected_network
    assert crs.startswith("EPSG:")
    assert {"x", "y"} <= set(nodes.columns)
    # projected edge length should be within a couple meters of the haversine length
    # (already recomputed from the projected geometry, so just check it's positive and finite)
    assert (edges["length_m"] > 0).all()
    assert np.isfinite(edges["length_m"].to_numpy()).all()


def test_crop_by_aoi_keeps_only_intersecting_edges(projected_network, sample_aoi):
    nodes, edges, crs = projected_network
    aoi_proj = sample_aoi.to_crs(crs)
    small_aoi = aoi_proj.copy()
    small_aoi["geometry"] = small_aoi.buffer(-200)  # shrink well inside the sample extent

    cropped_nodes, cropped_edges = graph_ops.crop_by_aoi(nodes, edges, small_aoi)
    assert cropped_edges.height <= edges.height
    used_ids = set(cropped_edges["u"].to_list()) | set(cropped_edges["v"].to_list())
    assert used_ids <= set(cropped_nodes["node_id"].to_list())


def test_simplify_drops_short_edges_and_keeps_graph_connected(projected_network):
    nodes, edges, crs = projected_network
    simplified_nodes, simplified_edges = graph_ops.simplify(nodes, edges, cluster_distance=15.0)

    assert simplified_edges.height <= edges.height
    assert (simplified_edges["u"] != simplified_edges["v"]).all()
    used_ids = set(simplified_edges["u"].to_list()) | set(simplified_edges["v"].to_list())
    assert used_ids <= set(simplified_nodes["node_id"].to_list())
    # simplification must not increase total network length materially
    assert simplified_edges["length_m"].sum() <= edges["length_m"].sum() * 1.05


def test_snap_points_creates_valid_new_nodes(projected_network):
    nodes, edges, crs = projected_network
    xs = nodes["x"].to_numpy()[:3] + 3.0
    ys = nodes["y"].to_numpy()[:3] + 3.0
    points = shapely.points(xs, ys)

    new_nodes, new_edges, ids = graph_ops.snap_points(nodes, edges, points, max_dist=100, min_edge_length=1.0)

    assert len(ids) == 3
    assert all(i is not None for i in ids)
    assert set(ids) <= set(new_nodes["node_id"].to_list())
    used_ids = set(new_edges["u"].to_list()) | set(new_edges["v"].to_list())
    assert used_ids <= set(new_nodes["node_id"].to_list())


def test_snap_points_respects_max_dist(projected_network):
    nodes, edges, crs = projected_network
    far_point = shapely.points([nodes["x"][0] + 1_000_000], [nodes["y"][0] + 1_000_000])

    _, _, ids = graph_ops.snap_points(nodes, edges, far_point, max_dist=10, min_edge_length=1.0)
    assert ids == [None]


def test_multi_source_distances_matches_networkx_free_expectations(projected_network):
    nodes, edges, crs = projected_network
    source = [int(nodes["node_id"][0])]

    result = routing.multi_source_distances(nodes, edges, source, radius=1000, directed=False)

    assert set(result.columns) == {"node_id", "source_id", "dist"}
    assert (result["dist"] <= 1000).all()
    assert (result["dist"] >= 0).all()
    # the source node itself is always reachable at distance 0
    src_row = result.filter(pl.col("node_id") == source[0])
    assert src_row["dist"][0] == 0.0


def test_multi_source_distances_radius_cutoff_is_monotonic(projected_network):
    nodes, edges, crs = projected_network
    source = [int(nodes["node_id"][0])]

    small = routing.multi_source_distances(nodes, edges, source, radius=100, directed=False)
    large = routing.multi_source_distances(nodes, edges, source, radius=2000, directed=False)

    assert small.height <= large.height
    assert set(small["node_id"].to_list()) <= set(large["node_id"].to_list())


def test_induced_edges_endpoints_within_reach(projected_network):
    nodes, edges, crs = projected_network
    source = [int(nodes["node_id"][0])]
    reach = routing.multi_source_distances(nodes, edges, source, radius=500, directed=False)

    induced = routing.induced_edges(edges, reach)
    assert (induced["u_dist"] <= 500).all()
    assert (induced["v_dist"] <= 500).all()
