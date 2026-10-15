"""Choose the area of interest: type a place, see its polygon, or draw your own on the map.

    selector = select_aoi("Alcoy, España")   # geocodes the place and builds the map
    selector.map                             # shows it: draw a polygon or rectangle to use your own area instead
    aoi = selector.aoi                       # the area to use from here on

If nothing is drawn (for example when all cells are simply run) the geocoded place is used.

Needs the optional ``ipyleaflet`` package (``pip install "UrbanAccessAnalyzer[plot]"``). In Google Colab the
interactive widgets are enabled automatically.
"""

from __future__ import annotations

import math
import sys
from typing import Optional

import geopandas as gpd
import shapely
from shapely.geometry import shape

from . import utils

OPENFREEMAP_TILEJSON = "https://tiles.openfreemap.org/planet"  # vector tiles of OpenStreetMap data, no key
OPENFREEMAP_ATTRIBUTION = (
    '&copy; <a href="https://openfreemap.org">OpenFreeMap</a> '
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)
# Light style for the OpenFreeMap (OpenMapTiles schema) layers
BASEMAP_STYLES = {
    "landcover": {"fill": True, "fillColor": "#d9e8cb", "fillOpacity": 0.7, "weight": 0},
    "landuse": {"fill": True, "fillColor": "#ece8e1", "fillOpacity": 0.6, "weight": 0},
    "park": {"fill": True, "fillColor": "#cde7c0", "fillOpacity": 0.8, "weight": 0},
    "water": {"fill": True, "fillColor": "#aad3df", "fillOpacity": 1, "color": "#aad3df", "weight": 0.5},
    "waterway": {"color": "#aad3df", "weight": 1},
    "building": {"fill": True, "fillColor": "#e4dfd6", "fillOpacity": 1, "color": "#d5cec3", "weight": 0.5},
    "transportation": {"color": "#b4aea3", "weight": 1},
    "boundary": {"color": "#999999", "weight": 1, "dashArray": "4,4"},
}
# Point/label layers are not drawn (an unstyled layer would get Leaflet's default blue style)
for _layer in ("aerodrome_label", "aeroway", "housenumber", "mountain_peak", "place", "poi", "transportation_name",
               "water_name"):
    BASEMAP_STYLES[_layer] = {"fill": False, "stroke": False, "opacity": 0, "fillOpacity": 0, "weight": 0}

LARGE_AREA_KM2 = 2000  # warn above this: downloads and isochrones get slow


def _openfreemap_layer():
    """OpenFreeMap vector tiles as an ipyleaflet layer (``None`` if the tile service cannot be reached)."""
    import ipyleaflet
    import requests

    try:
        tiles = requests.get(OPENFREEMAP_TILEJSON, timeout=10, headers={"User-Agent": utils.USER_AGENT}).json()["tiles"][0]
    except Exception as e:  # noqa: BLE001 - offline, blocked...
        print(f"Background map not available ({e}); the place is still drawn.")
        return None
    return ipyleaflet.VectorTileLayer(
        url=tiles, layer_styles=BASEMAP_STYLES, attribution=OPENFREEMAP_ATTRIBUTION,
        max_native_zoom=14, name="OpenFreeMap",
    )


def view_for_bounds(bounds, width: int = 800, height: int = 500, margin: float = 1.25):
    """Map center and zoom that show ``bounds`` (minx, miny, maxx, maxy in lon/lat) in a ``width`` x ``height`` px map.

    ``Map.fit_bounds`` does nothing before the widget is displayed, so the view is computed here.
    """
    minx, miny, maxx, maxy = bounds
    lat = (miny + maxy) / 2
    lon_span = max(maxx - minx, 1e-6) * margin
    lat_span = max((maxy - miny) / max(math.cos(math.radians(lat)), 0.05), 1e-6) * margin  # in lon-degree units
    zoom = min(math.log2(width * 360 / (256 * lon_span)), math.log2(height * 360 / (256 * lat_span)))
    return ((miny + maxy) / 2, (minx + maxx) / 2), int(max(1, min(18, math.floor(zoom))))


class AoiSelector:
    """A map with the geocoded place and drawing tools; :attr:`aoi` is the area chosen."""

    def __init__(self, place: str = "", geocoded: Optional[gpd.GeoDataFrame] = None, height: str = "500px"):
        import ipyleaflet

        self.place = place
        self.geocoded = geocoded
        self.map = ipyleaflet.Map(center=(20, 0), zoom=2, scroll_wheel_zoom=True, layout={"height": height})
        self.map.clear_layers()  # ipyleaflet adds an OpenStreetMap tile layer by default: not used here
        background = _openfreemap_layer()
        if background is not None:
            self.map.add_layer(background)

        if geocoded is not None and not geocoded.empty:
            self.map.add_layer(ipyleaflet.GeoJSON(
                data=geocoded.__geo_interface__, name=place or "place",
                style={"color": "#1f77b4", "weight": 3, "dashArray": "6,6", "fillOpacity": 0.1},
            ))
            self.map.center, self.map.zoom = view_for_bounds(geocoded.total_bounds)

        shape_style = {"shapeOptions": {"color": "#d62728", "fillOpacity": 0.15}}
        self.draw = ipyleaflet.DrawControl(
            polygon=shape_style, rectangle=shape_style,  # only these tools; {} disables the others
            polyline={}, circle={}, marker={}, circlemarker={},
        )
        self.map.add_control(self.draw)

    @property
    def source(self) -> str:
        """``"drawn"`` if you drew an area, otherwise ``"place"`` (the geocoded polygon)."""
        return "drawn" if self._drawn() else "place"

    def _drawn(self) -> list:
        return [
            shape(f["geometry"]) for f in (self.draw.data or [])
            if f.get("geometry", {}).get("type") in ("Polygon", "MultiPolygon")
        ]

    @property
    def aoi(self) -> gpd.GeoDataFrame:
        """The area to use: everything you drew (merged) if anything, otherwise the geocoded place (EPSG:4326)."""
        drawn = self._drawn()
        if drawn:
            gdf = gpd.GeoDataFrame(geometry=[shapely.union_all(drawn)], crs=4326)
            origin = "the area you drew"
        elif self.geocoded is not None and not self.geocoded.empty:
            gdf = gpd.GeoDataFrame(geometry=[self.geocoded.to_crs(4326).union_all()], crs=4326)
            origin = f"the place {self.place!r}"
        else:
            raise ValueError(
                f"Nothing to use: {self.place!r} was not found and no area was drawn. "
                "Draw a polygon or rectangle on the map, or write another place."
            )
        km2 = gdf.to_crs(gdf.estimate_utm_crs()).area.iloc[0] / 1e6
        note = f" Large area: it will take long." if km2 > LARGE_AREA_KM2 else ""
        print(f"Area of interest: {origin}, {km2:,.1f} km2.{note}")
        return gdf


def select_aoi(place: str = "", height: str = "500px", cache_dir: Optional[str] = None) -> AoiSelector:
    """Geocode ``place`` and open a map to confirm it or draw your own area (see the module docstring).

    Args:
        place: Place name to geocode (Nominatim). Empty or not found: draw your area on the map.
        height: Height of the map.
        cache_dir: Folder where the geocoded boundary is kept and reused (no new request when you rerun).
    """
    if "google.colab" in sys.modules:  # interactive widgets in Colab
        try:
            from google.colab import output

            output.enable_custom_widget_manager()
        except Exception:  # noqa: BLE001
            pass
    geocoded = None
    if place.strip():
        try:
            geocoded = utils.get_city_geometry(place, cache_dir=cache_dir)
        except Exception as e:  # noqa: BLE001 - not found / network
            print(f"Could not find {place!r} ({e}). Draw your area on the map.")
    return AoiSelector(place, geocoded, height)
