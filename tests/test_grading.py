import geopandas as gpd
import polars as pl
import pytest
import shapely

from UrbanAccessAnalyzer import api, grading, osm_io


def _network(sample_pbf_path, sample_aoi):
    aoi = api.AreaOfInterest(sample_aoi)
    return api.StreetNetwork.from_pbf(sample_pbf_path, aoi=aoi, network_type="all")


def _specs():
    return [
        {"name": "a", "query": None, "geometry": "Point", "importance": 3, "max_distance": 800},
        {"name": "b", "query": None, "geometry": "Polygon", "importance": 1, "max_distance": 800},
    ]


def test_tag_query_returns_callable():
    q = osm_io.tag_query("amenity", "dentist")
    assert callable(q) and q.__name__ == "amenity_dentist"


def test_check_specs_rejects_bad_geometry():
    bad = [{"name": "x", "query": None, "geometry": "Line", "importance": 1, "max_distance": 1}]
    with pytest.raises(ValueError):
        grading.fetch_pois(bad, gpd.GeoDataFrame(geometry=[], crs=4326), 4326)


def test_grade_weighted_scores(sample_pbf_path, sample_aoi):
    network = _network(sample_pbf_path, sample_aoi)
    x, y = network.nodes.select("x", "y").row(0)
    pois = {
        "a": gpd.GeoDataFrame(geometry=[shapely.Point(x, y)], crs=network.crs),
        "b": gpd.GeoDataFrame(geometry=[shapely.Point(x, y).buffer(60)], crs=network.crs),
    }
    edges, _ = grading.grade(network, _specs(), pois, n_steps=4, max_snap_dist=500, min_edge_length=1)
    assert {"a", "b", "access_score"} <= set(edges.columns)
    assert edges["access_score"].between(0, 100).all()
    assert edges["access_score"].max() > 0
    # weighted mean: (3*a + 1*b) / 4
    assert (abs(edges["access_score"] - (3 * edges["a"] + edges["b"]) / 4) < 1e-6).all()
