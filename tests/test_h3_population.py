import geopandas as gpd
import pandas as pd
import shapely

from UrbanAccessAnalyzer import h3_population, plotting, utils


def _hexes():
    return gpd.GeoDataFrame(
        {"population": [100, 300, 600], "access_score": [10.0, 55.0, 100.0]},
        geometry=[shapely.box(i, 0, i + 1, 1) for i in range(3)],
        crs=4326,
    )


def test_population_by_band():
    stats = h3_population.population_by_band(_hexes(), "access_score")
    got = dict(zip(stats["access_score"], stats["population"]))
    assert got["0-20"] == 100 and got["40-60"] == 300 and got["80-100"] == 600  # 100 is inside the last band
    assert abs(stats["population %"].sum() - 100) < 0.2


def test_population_by_group():
    h = _hexes()
    stats = h3_population.population_by_group(h, pd.Series(["a", "b", "a"], name="mode"))
    assert dict(zip(stats["mode"], stats["population"])) == {"a": 700, "b": 300}


def test_buffer_gdf_merges_and_grows():
    gdf = gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 0.01, 0.01), shapely.box(0.01, 0, 0.02, 0.01)], crs=4326)
    out = utils.buffer_gdf(gdf, 500)
    assert len(out) == 1 and out.union_all().area > gdf.union_all().area
    assert utils.buffer_gdf(gdf, 0).union_all().equals(gdf.union_all())


def test_select_file_local(tmp_path):
    f = tmp_path / "x.gpkg"
    f.write_text("x")
    assert utils.select_file(str(f)) == str(f)
    try:
        utils.select_file("")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError outside Colab")


def test_select_file_opens_upload_in_colab(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("google.colab")
    fake.files = types.SimpleNamespace(upload=lambda: {"my_pois.gpkg": b""})
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", fake)
    assert utils.select_file("") == "my_pois.gpkg"


def test_save_access_map_writes_standalone_html(tmp_path):
    h = _hexes()
    out = tmp_path / "maps" / "m.html"
    plotting.save_access_map(str(out), h, h, {"population": "👥 Population", "access_score": "🚶 Score"}, "access_score")
    html = out.read_text()
    assert "L.maplibreGL" in html and "Population" in html and "tile.openstreetmap.org" in html  # fallback


def test_rank_to_label():
    out = h3_population.rank_to_label(pd.Series([1.0, 1.6, 2.4, 9.0]), ["a", "b", "c"])
    assert list(out) == ["a", "b", "b", "c"]


def test_categorical_map_has_labels_in_legend(tmp_path):
    h = _hexes().assign(mode=["🚶 walk", "🚲 bike", "🚌 bus"])
    cats = {"🚶 walk": "#2e7d32", "🚲 bike": "#f9a825", "🚌 bus": "#c62828"}
    out = tmp_path / "m.html"
    plotting.save_access_map(str(out), h, h, {"population": "People", "mode": "Mode"}, "mode", categories=cats)
    html = out.read_text()
    assert "#2e7d32" in html and "L.maplibreGL" in html


def test_points_from_gdf():
    from UrbanAccessAnalyzer import api

    g = gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 2, 2)], crs=4326)
    assert api.PointsOfInterest.from_gdf(g).df.height == 1


def test_hexagons_without_streets_are_none_then_filled_from_nearest(monkeypatch):
    import numpy as np
    import pycensus

    aoi = gpd.GeoDataFrame(geometry=[shapely.box(-0.48, 38.69, -0.47, 38.70)], crs=4326)
    # one street along the west edge, score 3; people all over the area
    segs = [shapely.LineString([(-0.4755, 38.6905 + i * 0.0003), (-0.4755, 38.6905 + (i + 1) * 0.0003)]) for i in range(30)]
    street = gpd.GeoDataFrame({"rank": [3.0] * 30}, geometry=segs, crs=4326).to_crs(32630)  # short edges, like real streets

    def fake_pop(aoi, source="worldpop", **kw):
        xs, ys = np.meshgrid(np.linspace(-0.4795, -0.4705, 15), np.linspace(38.6905, 38.6995, 15))
        pts = [shapely.Point(x, y) for x, y in zip(xs.ravel(), ys.ravel())]
        return gpd.GeoDataFrame({"population": [4.0] * len(pts)}, geometry=pts, crs=4326)

    monkeypatch.setattr(pycensus, "get_population", fake_pop)
    raw = h3_population.h3_with_population(street, aoi, ["rank"], 10, empty_cells="ignore")
    assert raw["rank"].isna().sum() > 0 and (raw["rank"] == 0).sum() == 0  # no street -> None, never 0
    filled = h3_population.h3_with_population(street, aoi, ["rank"], 10)
    inhabited = filled[filled["population"] > 0]
    assert inhabited["rank"].notna().all() and set(inhabited["rank"].round(6)) == {3.0}  # nearest street's value
    assert filled["filled"].sum() > 0 and filled.loc[~filled["filled"] & filled["rank"].notna(), "rank"].gt(0).all()
    assert filled[filled["population"] == 0]["rank"].isna().all()  # no people, no street: stays empty
    capped = h3_population.h3_with_population(street, aoi, ["rank"], 10, max_distance=60)
    assert capped["rank"].isna().sum() > filled["rank"].isna().sum()  # farther hexagons are not filled


def test_relocate_weight_method_moves_people_to_hexagons_with_streets(monkeypatch):
    import numpy as np
    import pycensus

    aoi = gpd.GeoDataFrame(geometry=[shapely.box(-0.48, 38.69, -0.47, 38.70)], crs=4326)
    segs = [shapely.LineString([(-0.4755, 38.6905 + i * 0.0003), (-0.4755, 38.6905 + (i + 1) * 0.0003)]) for i in range(30)]
    street = gpd.GeoDataFrame({"rank": [3.0] * 30}, geometry=segs, crs=4326).to_crs(32630)

    def fake_pop(aoi, source="worldpop", **kw):
        xs, ys = np.meshgrid(np.linspace(-0.4795, -0.4705, 15), np.linspace(38.6905, 38.6995, 15))
        pts = [shapely.Point(x, y) for x, y in zip(xs.ravel(), ys.ravel())]
        return gpd.GeoDataFrame({"population": [4.0] * len(pts)}, geometry=pts, crs=4326)

    monkeypatch.setattr(pycensus, "get_population", fake_pop)
    ignore = h3_population.h3_with_population(street, aoi, ["rank"], 10, empty_cells="ignore")
    moved = h3_population.h3_with_population(street, aoi, ["rank"], 10, empty_cells="relocate_weight", max_distance=100000)
    assert moved["population"].sum() == ignore["population"].sum() and moved.attrs["excluded_weight"] == 0
    assert (moved.loc[moved["rank"].isna(), "population"] == 0).all()  # everyone now lives where there is a street
    capped = h3_population.h3_with_population(street, aoi, ["rank"], 10, empty_cells="relocate_weight", max_distance=30)
    assert capped.attrs["excluded_weight"] > 0 and capped["population"].sum() < ignore["population"].sum()
