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
        {% endmacro %}
        {% macro script(this, kwargs) %}
        (function() {
            var map = {{ this._parent.get_name() }};
            // Leaflet needs a max zoom for fitBounds/zoom controls; a vector style does not provide one.
            if (map.options.maxZoom === undefined) { map.options.maxZoom = 20; }
            L.maplibreGL({
                style: {{ this.style|tojson }},
                attribution: {{ this.attribution|tojson }},
                pane: {{ this.pane|tojson }}
            }).addTo(map);
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
