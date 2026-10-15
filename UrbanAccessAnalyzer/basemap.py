"""MapLibre vector-tile background maps for Folium maps.

Folium/Leaflet only draws raster tiles on its own. This module adds a
MapLibre GL vector-tile style (served by OpenFreeMap, no API key) as the
background of a Folium map, through the ``maplibre-gl-leaflet`` bridge:

    m = folium.Map(location=[42.37, -71.12], zoom_start=13, tiles=None)
    add_basemap(m, "positron")

Styles: https://openfreemap.org
"""

from __future__ import annotations

import folium
from branca.element import MacroElement
from jinja2 import Template

MAPLIBRE_JS = "https://unpkg.com/maplibre-gl@5.13.0/dist/maplibre-gl.js"
MAPLIBRE_CSS = "https://unpkg.com/maplibre-gl@5.13.0/dist/maplibre-gl.css"
VECTORGRID_JS = "https://unpkg.com/leaflet.vectorgrid@1.3.0/dist/Leaflet.VectorGrid.bundled.min.js"
MAPLIBRE_LEAFLET_JS = "https://unpkg.com/@maplibre/maplibre-gl-leaflet@0.1.3/leaflet-maplibre-gl.js"

BASEMAP_STYLES = {
    "positron": "https://tiles.openfreemap.org/styles/positron",  # light, minimal
    "bright": "https://tiles.openfreemap.org/styles/bright",
    "liberty": "https://tiles.openfreemap.org/styles/liberty",
    "dark": "https://tiles.openfreemap.org/styles/dark",
}
DEFAULT_BASEMAP = "positron"

ATTRIBUTION = (
    '&copy; <a href="https://openfreemap.org">OpenFreeMap</a> '
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)


class MapLibreBasemap(MacroElement):
    """A MapLibre GL style used as the background layer of a Folium map."""

    _template = Template(
        """
        {% macro header(this, kwargs) %}
        <link rel="stylesheet" href="{{ this.css }}"/>
        <script src="{{ this.js }}"></script>
        <script src="{{ this.leaflet_js }}"></script>
        <script src="{{ this.vectorgrid_js }}"></script>
        {% endmacro %}
        {% macro script(this, kwargs) %}
        (function() {
            var map = {{ this._parent.get_name() }};
            // Leaflet needs a max zoom for fitBounds/zoom controls; a vector style does not provide one.
            if (map.options.maxZoom === undefined) { map.options.maxZoom = 20; }
            function hasWebGL() {
                try { var c = document.createElement('canvas'); return !!(c.getContext('webgl2') || c.getContext('webgl')); }
                catch (e) { return false; }
            }
            function notice(text) {
                var c = L.control({position: 'bottomleft'});
                c.onAdd = function() {
                    var d = L.DomUtil.create('div');
                    d.style.cssText = 'background:#fff;padding:4px 8px;border-radius:6px;font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)';
                    d.textContent = text; return d;
                };
                c.addTo(map);
            }
            // Fallback without WebGL or web workers (e.g. Colab's output frame): same OpenFreeMap tiles, drawn on canvas.
            function canvasFallback() {
                if (!L.vectorGrid) { notice(hint); return; }
                fetch('https://tiles.openfreemap.org/planet').then(function(r) { return r.json(); }).then(function(tj) {
                    var st = {
                        water: {fill: true, fillColor: '#c8d6e5', fillOpacity: 1, stroke: false, weight: 0},
                        landcover: {fill: true, fillColor: '#dfe6d8', fillOpacity: 0.7, stroke: false, weight: 0},
                        park: {fill: true, fillColor: '#d5e3cf', fillOpacity: 0.8, stroke: false, weight: 0},
                        building: {fill: true, fillColor: '#e2e2e2', fillOpacity: 0.8, stroke: false, weight: 0},
                        waterway: {color: '#c8d6e5', weight: 1},
                        transportation: {color: '#ffffff', weight: 1.5, opacity: 1},
                        boundary: {color: '#b8b8c8', weight: 1, dashArray: '4 3'}
                    };
                    var rules = {};
                    Object.keys(st).forEach(function(k) { rules[k] = st[k]; });
                    var vg = L.vectorGrid["protobuf"](tj.tiles[0], {
                        vectorTileLayerStyles: new Proxy(rules, {get: function(t, k) { return t[k] || []; }}),
                        maxNativeZoom: 14, attribution: {{ this.attribution|tojson }}, pane: {{ this.pane|tojson }}
                    });
                    vg.addTo(map);
                }).catch(function() { notice(hint); });
            }
            var hint = 'Background map not available here. Open the saved .html file in your browser.';
            if (typeof L.maplibreGL === 'function' && typeof maplibregl !== 'undefined' && hasWebGL()) {
                try {
                    var gl = L.maplibreGL({
                        style: {{ this.style|tojson }},
                        attribution: {{ this.attribution|tojson }},
                        pane: {{ this.pane|tojson }}
                    }).addTo(map);
                    // A style paints a solid background even if its tiles never load (e.g. blocked workers): if no
                    // vector tile arrives in a few seconds, remove it and say so.
                    var tilesLoaded = false, dropGl = function() {
                        if (!tilesLoaded && map.hasLayer(gl)) { map.removeLayer(gl); canvasFallback(); console.warn('MapLibre tiles did not load'); }
                    };
                    var mgl = gl.getMaplibreMap();
                    mgl.on('sourcedata', function(e) { if (e.tile) { tilesLoaded = true; } });
                    mgl.on('error', function() { setTimeout(dropGl, 1500); });
                    setTimeout(dropGl, 8000);
                } catch (e) { canvasFallback(); console.warn('MapLibre background failed', e); }
            } else { canvasFallback(); }
        })();
        {% endmacro %}
        """
    )

    def __init__(self, style: str, attribution: str = ATTRIBUTION, pane: str = "tilePane"):
        super().__init__()
        self._name = "MapLibreBasemap"
        self.style = style
        self.attribution = attribution
        self.pane = pane
        self.css = MAPLIBRE_CSS
        self.js = MAPLIBRE_JS
        self.leaflet_js = MAPLIBRE_LEAFLET_JS
        self.vectorgrid_js = VECTORGRID_JS


def add_basemap(m: folium.Map, style: str = DEFAULT_BASEMAP) -> folium.Map:
    """Use a MapLibre vector-tile style as the background of a Folium map.

    Create the map with ``tiles=None`` so no raster background is added.

    Args:
        m: The Folium map.
        style: A key of :data:`BASEMAP_STYLES` (``"positron"``, ``"bright"``,
            ``"liberty"``, ``"dark"``) or the URL of any MapLibre style JSON.

    Returns:
        ``m``, for chaining.
    """
    MapLibreBasemap(BASEMAP_STYLES.get(style, style)).add_to(m)
    return m
