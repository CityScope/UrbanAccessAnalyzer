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
    xs = np.linspace(-0.48, -0.46, 8)
    lines = [shapely.LineString([(x, 38.69), (x, 38.70)]) for x in xs]
    edges = gpd.GeoDataFrame({"access_score": np.linspace(0, 100, 8)}, geometry=lines, crs=4326)
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
