"""Tests for `graph_ops.connected_component_labels`/`crop_by_aoi_connected`.

Synthetic node/edge tables only (no real PBF fixture needed) -- these are
pure graph-topology functions, so a small hand-built graph is enough to
exercise every branch: already-connected, bridgeable via a wider profile,
genuinely unbridgeable (falls back to the largest component), and the
per-simple-polygon (not per-multipolygon-as-a-whole) behavior.
"""

import geopandas as gpd
import polars as pl
import shapely

from UrbanAccessAnalyzer import graph_ops


def _line_wkb(x1, y1, x2, y2):
    return shapely.to_wkb(shapely.LineString([(x1, y1), (x2, y2)]))


def _nodes(ids_xy: dict) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "node_id": list(ids_xy.keys()),
            "x": [xy[0] for xy in ids_xy.values()],
            "y": [xy[1] for xy in ids_xy.values()],
        }
    )


def _edges(pairs, coords) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "u": [p[0] for p in pairs],
            "v": [p[1] for p in pairs],
            "length_m": [1.0] * len(pairs),
            "geometry_wkb": [
                _line_wkb(coords[p[0]][0], coords[p[0]][1], coords[p[1]][0], coords[p[1]][1]) for p in pairs
            ],
        }
    )


def test_connected_component_labels_separates_two_disjoint_pieces():
    coords = {1: (0, 0), 2: (1, 0), 3: (100, 100), 4: (101, 100)}
    nodes = _nodes(coords)
    edges = _edges([(1, 2), (3, 4)], coords)
    labeled = graph_ops.connected_component_labels(nodes, edges)
    lookup = dict(zip(labeled["node_id"].to_list(), labeled["_component"].to_list()))
    assert lookup[1] == lookup[2]
    assert lookup[3] == lookup[4]
    assert lookup[1] != lookup[3]


def test_connected_component_labels_single_component():
    coords = {1: (0, 0), 2: (1, 0), 3: (2, 0)}
    nodes = _nodes(coords)
    edges = _edges([(1, 2), (2, 3)], coords)
    labeled = graph_ops.connected_component_labels(nodes, edges)
    assert labeled["_component"].n_unique() == 1


def test_crop_by_aoi_connected_bridges_disconnected_pieces_via_wider_profile():
    # Two clusters (1-2) and (3-4), disconnected in the "walk" graph, but
    # linked by a road only present in the wider "all roads" graph (2-3).
    coords = {1: (0, 0), 2: (10, 0), 3: (20, 0), 4: (30, 0)}
    nodes = _nodes(coords)
    walk_edges = _edges([(1, 2), (3, 4)], coords)
    all_edges = _edges([(1, 2), (2, 3), (3, 4)], coords)  # (2, 3) is the missing link

    aoi = gpd.GeoDataFrame(geometry=[shapely.box(-5, -5, 35, 5)])

    out_nodes, out_edges = graph_ops.crop_by_aoi_connected(
        nodes, walk_edges, aoi, all_nodes=nodes, all_edges=all_edges
    )
    labeled = graph_ops.connected_component_labels(out_nodes, out_edges)
    assert labeled["_component"].n_unique() == 1, "the bridging edge should have been adopted"
    # The bridging edge (2, 3) must actually be present, not just declared connected.
    pairs = set(zip(out_edges["u"].to_list(), out_edges["v"].to_list())) | set(
        zip(out_edges["v"].to_list(), out_edges["u"].to_list())
    )
    assert (2, 3) in pairs


def test_crop_by_aoi_connected_falls_back_to_largest_component_when_unbridgeable():
    # No wider graph given at all -- (1, 2) is a real island (e.g. water-separated).
    coords = {1: (0, 0), 2: (1, 0), 3: (100, 100), 4: (101, 100), 5: (102, 100)}
    nodes = _nodes(coords)
    edges = _edges([(1, 2), (3, 4), (4, 5)], coords)
    aoi = gpd.GeoDataFrame(geometry=[shapely.box(-10, -10, 200, 200)])

    out_nodes, out_edges = graph_ops.crop_by_aoi_connected(nodes, edges, aoi)
    labeled = graph_ops.connected_component_labels(out_nodes, out_edges)
    assert labeled["_component"].n_unique() == 1
    # The larger component (3,4,5 -- 2 edges) survives, not the smaller (1,2 -- 1 edge).
    kept_ids = set(out_nodes["node_id"].to_list())
    assert kept_ids == {3, 4, 5}


def test_crop_by_aoi_connected_does_not_force_connectivity_across_separate_polygons():
    # Two real, separate simple polygons (e.g. two disjoint districts) --
    # each internally connected, but never expected to connect to the
    # OTHER polygon's graph. Passed as one MultiPolygon AOI.
    coords = {1: (0, 0), 2: (1, 0), 3: (100, 100), 4: (101, 100)}
    nodes = _nodes(coords)
    edges = _edges([(1, 2), (3, 4)], coords)
    multi_aoi = gpd.GeoDataFrame(
        geometry=[shapely.MultiPolygon([shapely.box(-5, -5, 5, 5), shapely.box(95, 95, 105, 105)])]
    )

    out_nodes, out_edges = graph_ops.crop_by_aoi_connected(nodes, edges, multi_aoi)
    # Both polygons' pieces are kept whole (each is internally connected on
    # its own) -- NOT reduced to a single "largest component" across the
    # whole AOI, which would have wrongly dropped one polygon entirely.
    assert set(out_nodes["node_id"].to_list()) == {1, 2, 3, 4}
    labeled = graph_ops.connected_component_labels(out_nodes, out_edges)
    assert labeled["_component"].n_unique() == 2
