"""Interactive Folium/ipyleaflet mapping for AOIs, POIs, and result layers.

Accepts the package's Polars-native output shapes directly: an H3 table
(Polars DataFrame with an ``h3_cell`` column, converted via
:func:`UrbanAccessAnalyzer.h3_ops.to_gdf`) or a street-edge table (Polars
DataFrame with a WKB ``geometry`` column, e.g. from
:mod:`UrbanAccessAnalyzer.isochrones`), alongside plain GeoDataFrames -- this
is the plotting-time conversion point back to geopandas, consistent with the
rest of the package treating geopandas as an I/O/display boundary rather than
an internal representation.
"""

import geopandas as gpd
from typing import Optional
from shapely.geometry import shape

from geohierarchy.maps.general_map import TOOLTIP_KWDS, TOOLTIP_STYLE, _format_value, general_map  # noqa: F401  (one implementation, in geohierarchy)





def popup_js(labels: dict, decimals: int = 1, categories: Optional[dict] = None) -> str:
    """JS function (for ``MapLayer.popup_js``) showing a small table with readable labels.

    Args:
        labels: ``{column: label}``, in display order, e.g.
            ``{"population": "Population", "access_score": "Walkability (0-100)"}``.
        decimals: Decimals for non-integer numbers.
        categories: Optional ``{column: [label1, label2, ...]}`` to show a rounded
            numeric value ``k`` as ``label_k`` (1-based), e.g. a mode rank as a name.
    """
    import json

    return (
        "function(p) {\n"
        f"  var labels = {json.dumps(labels)};\n"
        f"  var cats = {json.dumps(categories or {})};\n"
        "  var html = '<table style=\"border-spacing:6px 2px\">';\n"
        "  Object.keys(labels).forEach(function(k) {\n"
        "    var v = p[k];\n"
        "    if (v === undefined || v === null || v !== v) return;\n"
        "    if (cats[k]) { v = cats[k][Math.min(Math.max(Math.round(v), 1), cats[k].length) - 1]; }\n"
        f"    if (typeof v === 'number') v = Number.isInteger(v) ? v.toLocaleString() : v.toFixed({int(decimals)});\n"
        "    html += '<tr><td style=\"font-weight:600\">' + labels[k] + '</td><td>' + v + '</td></tr>';\n"
        "  });\n"
        "  return html + '</table>';\n"
        "}"
    )




def ipyleaflet_drawable_map(m=None, center=[0, 0], zoom=11, height="800px"):
    """
    Creates an interactive ipyleaflet map with drawing controls.

    Returns:
        - m: ipyleaflet map
        - get_drawn_geometries(): returns GeoDataFrame

    Args:
        m: Existing ipyleaflet map; ``None`` creates one.
        center: ``[lat, lon]`` of the initial view.
        zoom: Initial zoom.
        height: CSS height of the map.
    """
    import ipyleaflet  # optional dependency: only the drawing helper needs it
    from ipyleaflet import DrawControl, Map


    if m is None:
        m = Map(
            center=center,
            zoom=zoom,
            scroll_wheel_zoom=True,
            layout={'height': height}
        )

        google_hybrid = ipyleaflet.TileLayer(
            url="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            name="Google Hybrid",
            attribution="Google",
            opacity=0.75
        )

        m.add_layer(google_hybrid)

    draw_control = DrawControl(
        rectangle={"shapeOptions": {"color": "red"}},
        polygon={"shapeOptions": {"color": "red"}},
        circle={"shapeOptions": {"color": "red"}},
        polyline={},
        marker={},
        circlemarker={}
    )

    m.add_control(draw_control)

    # Store geometries by layer id
    drawn_geometries = {}

    def handle_draw(target, action, geo_json):
        """
        Handle draw events.
        """

        layer_id = geo_json.get("id")

        if action == "created":
            geom = shape(geo_json["geometry"])
            drawn_geometries[layer_id] = geom
            print(f"Geometry created: {layer_id}")

        elif action == "deleted":
            if layer_id in drawn_geometries:
                del drawn_geometries[layer_id]
                print(f"Geometry deleted: {layer_id}")

        elif action == "edited":
            geom = shape(geo_json["geometry"])
            drawn_geometries[layer_id] = geom
            print(f"Geometry edited: {layer_id}")

    draw_control.on_draw(handle_draw)

    def get_drawn_geometries():

        if len(drawn_geometries) == 0:
            return None

        return gpd.GeoDataFrame(
            geometry=list(drawn_geometries.values()),
            crs="EPSG:4326"
        )

    return m, get_drawn_geometries