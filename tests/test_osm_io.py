import polars as pl

from UrbanAccessAnalyzer import osm_io


def test_load_pbf_returns_polars_nodes_and_edges(sample_pbf_path):
    nodes, edges = osm_io.load_pbf(sample_pbf_path, network_type="all")

    assert isinstance(nodes, pl.DataFrame)
    assert isinstance(edges, pl.DataFrame)
    assert nodes.height > 0
    assert edges.height > 0
    assert {"node_id", "lon", "lat"} <= set(nodes.columns)
    assert {"u", "v", "length_m", "highway", "oneway", "geometry_wkb", "edge_uid"} <= set(edges.columns)
    assert (edges["length_m"] > 0).all()
    # every edge endpoint must be a known node
    assert set(edges["u"].to_list()) | set(edges["v"].to_list()) <= set(nodes["node_id"].to_list())


def test_load_pbf_network_type_filters_edges(sample_pbf_path):
    nodes_all, edges_all = osm_io.load_pbf(sample_pbf_path, network_type="all")
    nodes_walk, edges_walk = osm_io.load_pbf(sample_pbf_path, network_type="walk")

    assert edges_walk.height <= edges_all.height
    assert set(edges_walk["highway"].unique().to_list()) <= osm_io.WALK_HIGHWAYS | {None}


def test_load_pbf_bbox_prefilter_shrinks_result(sample_pbf_path, sample_aoi):
    nodes_full, edges_full = osm_io.load_pbf(sample_pbf_path, network_type="all")
    nodes_aoi, edges_aoi = osm_io.load_pbf(sample_pbf_path, aoi=sample_aoi, network_type="all")

    assert nodes_aoi.height <= nodes_full.height
    assert edges_aoi.height <= edges_full.height


def test_load_pbf_oneway_produces_single_direction(sample_pbf_path):
    _, edges = osm_io.load_pbf(sample_pbf_path, network_type="all")
    oneway_edges = edges.filter(pl.col("oneway"))
    if oneway_edges.height == 0:
        return
    # for a oneway edge_uid there should be exactly one directed row
    counts = oneway_edges.group_by("edge_uid").agg(pl.len().alias("n"))
    assert (counts["n"] == 1).all()
