import geopandas as gpd
import pytest
import shapely

from UrbanAccessAnalyzer import aoi_selector, select_aoi

PLACE = gpd.GeoDataFrame(geometry=[shapely.box(-0.5, 38.6, -0.4, 38.7)], crs=4326)


def _feature(poly):
    return {"type": "Feature", "properties": {}, "geometry": shapely.geometry.mapping(poly)}


def test_run_all_cells_uses_the_geocoded_place():
    sel = aoi_selector.AoiSelector("Alcoy", PLACE)
    assert sel.source == "place"
    assert sel.aoi.union_all().equals(PLACE.union_all()) and sel.aoi.crs == 4326  # nothing drawn: the place


def test_drawn_area_replaces_the_place_and_several_are_merged():
    sel = aoi_selector.AoiSelector("Alcoy", PLACE)
    a, b = shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)
    sel.draw.data = [_feature(a)]
    assert sel.source == "drawn" and sel.aoi.union_all().equals(a)
    sel.draw.data = [_feature(a), _feature(b)]
    assert sel.aoi.union_all().equals(shapely.union(a, b))
    sel.draw.data = []  # deleted: back to the place
    assert sel.source == "place" and sel.aoi.union_all().equals(PLACE.union_all())


def test_not_found_place_needs_a_drawing():
    sel = aoi_selector.AoiSelector("nowhere", None)
    with pytest.raises(ValueError, match="draw"):
        sel.aoi
    sel.draw.data = [_feature(shapely.box(0, 0, 1, 1))]
    assert sel.aoi.union_all().equals(shapely.box(0, 0, 1, 1))


def test_select_aoi_geocodes_and_survives_a_failed_search(monkeypatch, capsys):
    monkeypatch.setattr(aoi_selector.utils, "get_city_geometry", lambda name, cache_dir=None: PLACE)
    assert select_aoi("Alcoy").source == "place"
    def boom(name, cache_dir=None):
        raise RuntimeError("not found")
    monkeypatch.setattr(aoi_selector.utils, "get_city_geometry", boom)
    sel = select_aoi("zzz")
    assert sel.geocoded is None and "Draw your area" in capsys.readouterr().out
    assert select_aoi("").geocoded is None


def test_view_shows_the_place_not_the_world():
    sel = aoi_selector.AoiSelector("Alcoy", PLACE)  # a ~10 km box
    assert 10 <= sel.map.zoom <= 12 and abs(sel.map.center[0] - 38.65) < 1e-6
    country = gpd.GeoDataFrame(geometry=[shapely.box(-9, 36, 3, 43)], crs=4326)
    assert 4 <= aoi_selector.AoiSelector("Spain", country).map.zoom <= 6
    assert aoi_selector.view_for_bounds((0, 0, 0.0001, 0.0001))[1] <= 18


def test_geocoding_retries_then_caches(tmp_path, monkeypatch):
    import requests

    calls = {"n": 0}

    class Resp:
        status_code = 200
        headers = {}
        def raise_for_status(self): pass
        def json(self):
            return [{"display_name": "Alcoy", "geojson": shapely.geometry.mapping(shapely.box(-0.5, 38.6, -0.4, 38.7))}]

    def flaky_get(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.exceptions.Timeout("slow")
        return Resp()

    monkeypatch.setattr(requests, "get", flaky_get)
    monkeypatch.setattr("time.sleep", lambda s: None)
    first = aoi_selector.utils.get_city_geometry("Alcoy", cache_dir=str(tmp_path))
    assert calls["n"] == 3 and len(first) == 1                      # two timeouts retried, third worked
    again = aoi_selector.utils.get_city_geometry("  ALCOY ", cache_dir=str(tmp_path))
    assert calls["n"] == 3 and again.union_all().equals(first.union_all())  # second time: from the cache, no request


def test_geocoding_gives_up_with_a_clear_error(monkeypatch):
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.Timeout("slow")))
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="did not answer after 3 attempts"):
        aoi_selector.utils.get_city_geometry("Alcoy")


def test_map_has_no_default_openstreetmap_layer():
    sel = aoi_selector.AoiSelector("Alcoy", PLACE)
    urls = [getattr(layer, "url", "") or "" for layer in sel.map.layers]
    assert not any("openstreetmap.org/" in u and "{z}" in u for u in urls)  # only OpenFreeMap (or nothing offline)
    assert "poi" in aoi_selector.BASEMAP_STYLES and aoi_selector.BASEMAP_STYLES["poi"]["opacity"] == 0
