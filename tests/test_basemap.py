import folium

from UrbanAccessAnalyzer import plotting
from UrbanAccessAnalyzer.basemap import BASEMAP_STYLES, add_basemap


def _html(m):
    return m.get_root().render()


def test_add_basemap_uses_maplibre_style_not_raster():
    m = folium.Map(location=[0, 0], zoom_start=3, tiles=None)
    add_basemap(m, "positron")
    html = _html(m)
    assert "L.maplibreGL" in html and BASEMAP_STYLES["positron"] in html
    assert "cartocdn" not in html.lower()


def test_add_basemap_accepts_style_url():
    m = folium.Map(tiles=None)
    add_basemap(m, "https://example.org/style.json")
    assert "https://example.org/style.json" in _html(m)


def test_general_map_has_no_cartodb():
    import geopandas as gpd
    from shapely.geometry import box

    aoi = gpd.GeoDataFrame(geometry=[box(0, 0, 0.01, 0.01)], crs=4326)
    html = _html(plotting.general_map(aoi=aoi))
    assert "L.maplibreGL" in html and "carto" not in html.lower()


def test_basemap_sets_max_zoom_so_fit_bounds_works():
    m = folium.Map(tiles=None)
    add_basemap(m)
    html = _html(m)
    assert "map.options.maxZoom = 20" in html
