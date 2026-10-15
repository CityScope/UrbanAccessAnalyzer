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


def test_query_from_file_and_poi_table(tmp_path):
    import geopandas as gpd
    import shapely

    pts = gpd.GeoDataFrame(
        {"nombre": ["A", "B", "far"]},
        geometry=[shapely.Point(-0.48, 38.70), shapely.Point(-0.47, 38.70), shapely.Point(10, 10)],
        crs=4326,
    ).to_crs(25830)
    path = tmp_path / "pois.gpkg"
    pts.to_file(path)
    bounds = gpd.GeoDataFrame(geometry=[shapely.box(-0.5, 38.6, -0.4, 38.8)], crs=4326)

    df = grading.query_from_file(str(path), name_column="nombre")(bounds)
    assert df.height == 2 and set(df["name"]) == {"A", "B"}

    specs = [{"name": "mine", "query": None, "geometry": "Point", "importance": 1, "max_distance": 500, "icon": "🍺"}]
    gdf = grading.fetch_pois([{**specs[0], "query": lambda b: df}], bounds, 4326)
    table = grading.poi_table(specs, gdf)
    assert list(table["icon"].unique()) == ["🍺"] and set(table["name"]) == {"A", "B"}


def test_general_map_shows_emoji_icons_and_polygon_color():
    import geopandas as gpd
    import shapely
    from UrbanAccessAnalyzer import plotting

    table = gpd.GeoDataFrame(
        {"type": ["shop", "park"], "name": ["S", "P"], "icon": ["🛒", "🌳"]},
        geometry=[shapely.Point(0, 0), shapely.box(0.001, 0.001, 0.002, 0.002)],
        crs=4326,
    )
    html = plotting.general_map(pois=table, poi_column="type", poi_icon_column="icon").get_root().render()
    assert "🛒" in html and "🌳" in html
    assert "#4c9f50" in html


def test_popup_js_uses_labels():
    from UrbanAccessAnalyzer import plotting

    js = plotting.popup_js({"population": "Population", "access_score": "Walkability"})
    assert "Population" in js and "toFixed(1)" in js


def test_popup_js_categories():
    from UrbanAccessAnalyzer import plotting

    js = plotting.popup_js({"mode_rank": "Mode"}, categories={"mode_rank": ["walk", "bike"]})
    assert '"walk"' in js and "Math.round" in js


def test_tiers_table():
    specs = [
        {"name": "a", "query": None, "geometry": "Point", "importance": 3, "max_distance": 1000, "icon": "🛒"},
        {"name": "b", "query": None, "geometry": "Point", "importance": 1, "max_distance": 500},
    ]
    t = grading.tiers_table(specs, n_steps=4)
    a = t[t["type"] == "a"]
    assert list(a["within (m)"]) == [250, 500, 750, 1000] and list(a["score"]) == [100, 75, 50, 25]
    assert set(a["weight %"]) == {75.0} and set(t[t["type"] == "b"]["icon"]) == {"📍"}


def test_poi_sources(tmp_path):
    import geopandas as gpd
    import shapely
    from UrbanAccessAnalyzer import osm_io

    assert grading.poi("s", "schools")["query"] is osm_io.schools
    p = grading.poi("parks", "green_areas", importance=2, icon="🌳")
    assert p["geometry"] == "Polygon" and p["importance"] == 2 and p["icon"] == "🌳"
    assert grading.poi("d", ("amenity", "dentist"))["query"].__name__ == "amenity_dentist"
    f = lambda b: None
    assert grading.poi("c", f, geometry="Polygon")["query"] is f

    path = tmp_path / "x.gpkg"
    gpd.GeoDataFrame({"n": ["a"]}, geometry=[shapely.Point(0, 0)], crs=4326).to_file(path)
    mine = grading.poi("mine", str(path), name_column="n")
    assert mine["geometry"] == "Point" and callable(mine["query"])
    for bad in (123, "no_such_file.gpkg"):
        try:
            grading.poi("x", bad)
        except (TypeError, FileNotFoundError):
            pass
        else:
            raise AssertionError(bad)


def test_overpass_filter_becomes_full_query(monkeypatch):
    import geopandas as gpd
    import polars as pl
    import shapely
    from UrbanAccessAnalyzer import osm_io

    seen = {}
    monkeypatch.setattr(osm_io, "overpass_query", lambda q, b: seen.setdefault("q", q) and pl.DataFrame())
    bounds = gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 1, 1)], crs=4326)
    grading.poi("c", 'nwr["amenity"~"hospital|clinic"]')["query"](bounds)
    assert '[out:json]' in seen["q"] and '{{bbox}}' in seen["q"] and 'hospital|clinic' in seen["q"] and "out body" in seen["q"]
    full = '[out:json];node["a"="b"]({{bbox}});out;'
    assert grading.poi("f", full)["query"].__name__ == "overpass_query_custom"
