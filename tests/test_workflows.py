import geopandas as gpd
import numpy as np
import shapely

from UrbanAccessAnalyzer import workflows


def _fake_population(monkeypatch):
    import pycensus

    def fake(aoi, source="worldpop", **kw):
        xs = np.linspace(aoi.total_bounds[0], aoi.total_bounds[2], 12)
        ys = np.linspace(aoi.total_bounds[1], aoi.total_bounds[3], 12)
        pts = [shapely.Point(x, y) for x in xs for y in ys]
        return gpd.GeoDataFrame({"population": [5.0] * len(pts)}, geometry=pts, crs=4326)

    monkeypatch.setattr(pycensus, "get_population", fake)


def test_report_writes_map_and_stats(tmp_path, monkeypatch):
    _fake_population(monkeypatch)
    aoi = gpd.GeoDataFrame(geometry=[shapely.box(-0.48, 38.69, -0.46, 38.70)], crs=4326)
    xs = np.linspace(-0.478, -0.462, 8)
    lines, scores = [], []
    for x, sc in zip(xs, np.linspace(0, 100, 8)):  # short edges like a real street network
        for i in range(30):
            lines.append(shapely.LineString([(x, 38.6905 + i * 0.0003), (x, 38.6905 + (i + 1) * 0.0003)]))
            scores.append(sc)
    edges = gpd.GeoDataFrame({"access_score": scores}, geometry=lines, crs=4326)
    labels = {"population": "People", "access_score": "Score"}
    hexes, stats, m = workflows.report(edges, aoi, str(tmp_path), ["access_score"], labels, "access_score", "m.html", 10)
    assert (tmp_path / "maps" / "m.html").exists() and (tmp_path / "stats.csv").exists()
    assert hexes["population"].sum() > 0 and abs(stats["population %"].sum() - 100) < 0.5
    html = (tmp_path / "maps" / "m.html").read_text()
    assert "L.maplibreGL" in html and '"streets"' in html and "pointer" in html.lower()


def test_folder_required_for_geodataframe():
    aoi = gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 1, 1)], crs=4326)
    try:
        workflows._setup(aoi, None)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _edges_and_aoi():
    aoi = gpd.GeoDataFrame(geometry=[shapely.box(-0.48, 38.69, -0.46, 38.70)], crs=4326)
    xs = np.linspace(-0.478, -0.462, 8)
    lines, scores = [], []
    for x, sc in zip(xs, np.linspace(0, 100, 8)):
        for i in range(30):
            lines.append(shapely.LineString([(x, 38.6905 + i * 0.0003), (x, 38.6905 + (i + 1) * 0.0003)]))
            scores.append(sc)
    edges = gpd.GeoDataFrame({"access_score": scores, "schools": scores}, geometry=lines, crs=4326)
    return edges, aoi


def test_walkability_and_green_results_steps_run_with_every_empty_cells_method(tmp_path, monkeypatch):
    _fake_population(monkeypatch)
    edges, aoi = _edges_and_aoi()
    pois = [{"name": "schools", "query": None, "geometry": "Point", "importance": 1, "max_distance": 1000, "icon": "🏫"}]
    for method in ("nearest_value", "relocate_weight", "ignore"):
        hexes, stats, m = workflows.walkability_results(
            edges, aoi, str(tmp_path / method), pois, None, 10, "worldpop", empty_cells=method, empty_max_distance=500
        )
        assert (tmp_path / method / "maps" / "walkability_map.html").exists() and len(hexes) > 0
        hexes, stats, m = workflows.green_results(
            edges, aoi, str(tmp_path / ("g" + method)), 1200, 3, None, 10, "worldpop", empty_cells=method, empty_max_distance=500
        )
        assert (tmp_path / ("g" + method) / "maps" / "green_areas_map.html").exists()


def test_school_results_step_runs(tmp_path, monkeypatch):
    _fake_population(monkeypatch)
    edges, aoi = _edges_and_aoi()
    modes = ["🚶 walk", "🚲 bike", "🚌 bus/car"]
    edges["mode"] = np.where(edges["access_score"] < 40, modes[0], np.where(edges["access_score"] < 80, modes[1], modes[2]))
    hexes, stats, m = workflows.school_results(edges, aoi, str(tmp_path), modes, ["#2e7d32", "#f9a825", "#c62828"], None, 10, "worldpop")
    assert set(stats["mode"]) <= set(modes) and (tmp_path / "maps" / "school_access_map.html").exists()
