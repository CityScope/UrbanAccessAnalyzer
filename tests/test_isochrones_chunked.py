"""Correctness test: chunked vs. unchunked compute_node_access must agree.

Highest-priority test for the H3-chunked isochrone pipeline (see
`TransitLOSStudies/CS_transitLOS/docs/H3_CHUNKED_PIPELINE_DESIGN.md`): a
subtle bug in the chunking/merge logic would silently produce wrong
access_score values for every city, not just the ones OOMing on the
unchunked path. This runs the small real sample OSM extract both ways and
asserts every node's access_score/rank/remaining_dist match exactly (pure
float noise tolerance only).
"""

import numpy as np
import polars as pl
import pytest
import shapely

from UrbanAccessAnalyzer import graph_ops, isochrones, osm_io


@pytest.fixture(scope="module")
def projected_network(sample_pbf_path):
    nodes, edges = osm_io.load_pbf(sample_pbf_path, network_type="all")
    nodes, edges, crs = graph_ops.project(nodes, edges)
    nodes, edges = graph_ops.simplify(nodes, edges, cluster_distance=10.0)
    return nodes, edges, crs


def test_chunked_matches_unchunked_node_access(projected_network):
    nodes, edges, crs = projected_network

    # Spread POIs across the sample extract so multiple H3 chunks each get
    # at least one source.
    n_poi = min(6, nodes.height)
    idx = np.linspace(0, nodes.height - 1, n_poi).astype(int)
    xs = nodes["x"].to_numpy()[idx]
    ys = nodes["y"].to_numpy()[idx]
    scores = [1.0, 0.75, 0.5, 0.25, 1.0, 0.5][:n_poi]
    poi = pl.DataFrame({"geometry": shapely.to_wkb(shapely.points(xs, ys)), "poi_score": scores})

    matrix, _ = isochrones.default_distance_matrix(poi, [200, 600, 1200], poi_score_col="poi_score")
    process_order = isochrones.distance_matrix_to_processing_order(matrix)

    point_geoms = shapely.from_wkb(poi["geometry"].to_numpy())
    snapped_nodes, snapped_edges, snapped_ids = graph_ops.snap_points(
        nodes, edges, point_geoms, max_dist=200, min_edge_length=1.0
    )
    points = poi.with_columns(pl.Series("node_id", snapped_ids))

    unchunked_access, _ = isochrones.compute_node_access(
        snapped_nodes, snapped_edges, points, process_order, poi_score_col="poi_score", verbose=False
    )

    # Fine H3 resolution so the small sample extract actually splits into
    # multiple chunks, with a buffer comfortably larger than the largest
    # tier distance (1200m) to avoid any boundary-truncation warning/effect.
    chunked_access, _ = isochrones.compute_node_access_chunked(
        snapped_nodes, snapped_edges, points, process_order, crs,
        poi_score_col="poi_score", verbose=False,
        chunk_h3_resolution=9, buffer_m=2000.0,
    )
    n_chunks = chunked_access["h3_chunk_cell"].n_unique()
    assert n_chunks > 1, "test setup should exercise multiple H3 chunks"

    a = unchunked_access.sort("node_id")
    b = chunked_access.sort("node_id").select(unchunked_access.columns)

    assert a["node_id"].to_list() == b["node_id"].to_list()
    assert a["rank"].to_list() == b["rank"].to_list()
    np.testing.assert_allclose(a["access_score"].to_numpy(), b["access_score"].to_numpy())
    np.testing.assert_allclose(a["remaining_dist"].to_numpy(), b["remaining_dist"].to_numpy(), atol=1e-6)


def test_chunked_warns_when_buffer_smaller_than_max_tier_distance(projected_network):
    nodes, edges, crs = projected_network
    poi = pl.DataFrame({
        "geometry": shapely.to_wkb(shapely.points(nodes["x"].to_numpy()[:2], nodes["y"].to_numpy()[:2])),
        "poi_score": [1.0, 0.5],
    })
    matrix, _ = isochrones.default_distance_matrix(poi, [2000], poi_score_col="poi_score")
    process_order = isochrones.distance_matrix_to_processing_order(matrix)
    point_geoms = shapely.from_wkb(poi["geometry"].to_numpy())
    snapped_nodes, snapped_edges, snapped_ids = graph_ops.snap_points(
        nodes, edges, point_geoms, max_dist=200, min_edge_length=1.0
    )
    points = poi.with_columns(pl.Series("node_id", snapped_ids))

    with pytest.warns(UserWarning, match="buffer_m"):
        isochrones.compute_node_access_chunked(
            snapped_nodes, snapped_edges, points, process_order, crs,
            poi_score_col="poi_score", verbose=False,
            chunk_h3_resolution=9, buffer_m=500.0,
        )
