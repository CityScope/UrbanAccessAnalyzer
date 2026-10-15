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

import folium
import pandas as pd
import geopandas as gpd
import polars as pl
import shapely
from typing import Optional, List, Union
from matplotlib import colormaps as mpl_colormaps
import matplotlib.colors as mcolors
import numpy as np
from shapely.geometry import shape

from . import h3_ops
from .basemap import add_basemap

# Hover-card look for the colored layers (hexagons, streets): white rounded card.
TOOLTIP_STYLE = (
    "background-color: white; color: #222; border: none; border-radius: 10px; "
    "box-shadow: 0 2px 12px rgba(0,0,0,0.25); padding: 8px 12px; "
    "font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; font-size: 13px;"
)
TOOLTIP_KWDS = {"style": TOOLTIP_STYLE, "sticky": True}


def _format_value(v) -> str:
    """Readable value: thousands separators for integers, one decimal for floats."""
    import html as _html

    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{int(v):,}" if float(v).is_integer() else f"{v:,.1f}"
    return _html.escape(str(v))


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


def general_map(
    m: Optional[folium.Map] = None,
    aoi: Optional[gpd.GeoDataFrame] = None,
    pois: Optional[Union[gpd.GeoDataFrame, List[gpd.GeoDataFrame]]] = None,
    gdfs: Optional[
        Union[
            gpd.GeoDataFrame,
            pd.DataFrame,
            List[Union[gpd.GeoDataFrame, pd.DataFrame]],
        ]
    ] = None,
    poi_column: Optional[str] = None,
    poi_color: Optional[str] = None,
    poi_cmap: Optional[str] = None,
    poi_vmin: Optional[float] = None,
    poi_vmax: Optional[float] = None,
    poi_icon_column: Optional[str] = None,
    column: Optional[str] = None,
    color: str = "black",
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    opacity: float = 0.4,
    size_column: Optional[str] = None,
    zoom_start: int = 11,
    cluster_pois: bool = True,
    categories: Optional[dict] = None,
    legend_on: bool = True,
) -> folium.Map:

    # ==========================================================
    # DEFAULTS
    # ==========================================================
    if pois is None:
        pois = []
    if gdfs is None:
        gdfs = []

    # ==========================================================
    # CRS NORMALIZATION
    # ==========================================================
    if aoi is not None:
        aoi = aoi.to_crs(4326)

    def _normalize_gdfs(objs):
        out = []
        for g in objs:
            if isinstance(g, gpd.GeoDataFrame):
                g = g.to_crs(4326)

            elif isinstance(g, pl.DataFrame) and "h3_cell" in g.columns:
                g = h3_ops.to_gdf(g).to_crs(4326)

            elif isinstance(g, pl.DataFrame) and "geometry" in g.columns and g.schema["geometry"] == pl.Binary:
                geom = shapely.from_wkb(g["geometry"].to_numpy())
                g = gpd.GeoDataFrame(g.drop("geometry").to_pandas(), geometry=geom, crs=4326).to_crs(4326)

            else:
                raise ValueError("Unsupported GeoDataFrame input")

            g = g[g.geometry.is_valid]
            out.append(g)

        return out

    if not isinstance(gdfs, list):
        gdfs = [gdfs]
    gdfs = _normalize_gdfs(gdfs)

    if not isinstance(pois, list):
        pois = [pois]
    pois = _normalize_gdfs(pois)

    if len(gdfs) == 0 and len(pois) == 0 and aoi is None:
        raise ValueError("Nothing to map")

    # ==========================================================
    # CENTER
    # ==========================================================
    all_geoms = []

    if aoi is not None:
        all_geoms.append(aoi.geometry)

    for g in gdfs:
        all_geoms.append(g.geometry)

    for p in pois:
        all_geoms.append(p.geometry)

    centroid = pd.concat(all_geoms).union_all().centroid

    # ==========================================================
    # MAP (MapLibre vector-tile basemap)
    # ==========================================================
    if m is None:
        m = folium.Map(
            location=[centroid.y, centroid.x],
            zoom_start=zoom_start,
            tiles=None,
            control_scale=True,
        )
        add_basemap(m, "positron")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def split_geoms(gdf):
        return (
            gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])],
            gdf[gdf.geometry.type.isin(["LineString", "MultiLineString", "LinearRing"])],
            gdf[gdf.geometry.type.isin(["Point", "MultiPoint"])],
        )

    def is_thematic(gdf, column, cmap):
        return column is not None and cmap is not None and column in gdf.columns

    def compute_radius(series: pd.Series, max_radius: float = 12.0):
        """
        Scale values from 0 → p90 into 0 → max_radius
        """
        p90 = series.quantile(0.9)
        clipped = series.clip(lower=0, upper=p90)
        return max_radius * clipped / p90 if p90 > 0 else max_radius

    def tooltip_html(row):
        """Card for a POI: emoji + name as the title, its type below, then the other values."""
        import html as _html

        icon = row.get(poi_icon_column) if poi_icon_column else None
        name = row.get("name")
        name = "" if name is None or (isinstance(name, float) and np.isnan(name)) else str(name).strip()
        kind = row.get(poi_column) if poi_column else None
        kind = "" if kind is None else str(kind)
        skip = {"geometry", poi_icon_column, "name", poi_column}

        title = _html.escape(name or kind or "Point of interest")
        card = (
            "<div style=\"font-family:system-ui,-apple-system,'Segoe UI',sans-serif;min-width:170px;\">"
            f"<div style='font-size:15px;font-weight:600;line-height:1.3'>{icon + ' ' if icon else ''}{title}</div>"
        )
        if name and kind:
            card += f"<div style='color:#777;font-size:12px;margin-bottom:4px'>{_html.escape(kind)}</div>"
        rows = ""
        for k, v in row.items():
            if k in skip:
                continue
            text = _format_value(v)
            if text:
                rows += (
                    f"<tr><td style='color:#777;padding-right:10px'>{_html.escape(str(k))}</td>"
                    f"<td style='font-weight:600'>{text}</td></tr>"
                )
        if rows:
            card += f"<table style='font-size:13px;margin-top:2px'>{rows}</table>"
        return card + "</div>"

    def increase_legend_size(m):
        legend_css = """
        <style>

        /* Main legend container */
        .legend {
            font-size: 18px !important;
            background-color: white !important;
            padding: 12px !important;
            border-radius: 8px !important;
            border: 2px solid rgba(0,0,0,0.2) !important;
            box-shadow: 0 0 15px rgba(0,0,0,0.2) !important;
        }

        /* Legend title */
        .legend-title {
            font-size: 22px !important;
            font-weight: bold !important;
            background-color: white !important;
            padding-bottom: 8px !important;
        }

        /* Legend labels / bins */
        .legend-labels {
            font-size: 18px !important;
            background-color: white !important;
        }

        /* Ensure all list items also have white background */
        .legend ul,
        .legend li {
            background-color: white !important;
        }

        </style>
        """

        m.get_root().header.add_child(folium.Element(legend_css))
        return m

    # ------------------------------------------------------------------
    # vmin / vmax for gdfs
    # ------------------------------------------------------------------
    if poi_cmap is None:
        poi_cmap = cmap 


    if poi_vmin is None:
        poi_vmin = vmin 
    
    if poi_vmax is None:
        poi_vmax = vmax 
        
    if column:
        values = [g[column].dropna() for g in gdfs if column in g.columns]
        if values:
            if vmin is None:
                vmin = min(v.min() for v in values)
            if vmax is None:
                vmax = max(v.max() for v in values)

    if poi_column is None:
        poi_column = column 

    if poi_column:
        values = [p[poi_column].dropna() for p in pois if poi_column in p.columns]
        if values:
            if poi_vmin is None:
                poi_vmin = min(v.min() for v in values)
            if poi_vmax is None:
                poi_vmax = max(v.max() for v in values)

    # ------------------------------------------------------------------
    # Draw gdfs
    # ------------------------------------------------------------------
    # Categorical column (text labels): ``categories`` is {label: color}; the legend lists the labels.
    if categories:
        cmap = list(categories.values())
    scale = (
        dict(categorical=True, categories=list(categories), cmap=cmap)
        if categories
        else dict(cmap=cmap, vmin=vmin, vmax=vmax)
    )

    legend = legend_on
    for g in gdfs:
        polys, lines, points = split_geoms(g)

        # Polygons
        if not polys.empty:
            if is_thematic(polys, column, cmap):
                m = polys.explore(
                    m=m,
                    column=column,
                    **scale,
                    legend=legend,
                    legend_kwds={"position": "topright", "caption": column},
                    style_kwds={"color": "white", "weight": 0.6, "fillOpacity": opacity},
                    tooltip_kwds=TOOLTIP_KWDS,
                )
                if legend:
                    m = increase_legend_size(m)
                legend = False
            else:
                m = polys.explore(
                    m=m,
                    color=color,
                    style_kwds={"fillColor": color, "fillOpacity": opacity, "weight": 0},
                )

        # Lines
        if not lines.empty:
            if is_thematic(lines, column, cmap):
                m = lines.explore(
                    m=m,
                    column=column,
                    **scale,
                    legend=legend,
                    legend_kwds={"caption": column},
                    style_kwds={"weight": 2.5, "opacity": 0.9},
                    tooltip_kwds=TOOLTIP_KWDS,
                )
                if legend:
                    m = increase_legend_size(m)
                legend = False
            else:
                m = lines.explore(m=m, color=color, style_kwds={"weight": 2})


        # Points with size scaling
        if not points.empty:
            if size_column is not None and size_column in points.columns:
                # Compute radii
                radii = compute_radius(points[size_column])
                points = points.assign(__radius=radii)
                
                # --- Prepare size legend ---
                # Choose 5 representative values from the size column
                size_values = np.linspace(points[size_column].min(), points[size_column].max(), 5)
                radius_values = compute_radius(pd.Series(size_values))
                
                # Add legend as a separate HTML overlay
                legend_html = f"""
                    <div style="position: fixed; top: 40px; left: 40px; z-index:9999; 
                    background:white; padding:10px; border-radius:5px; 
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
                    <b>{size_column} legend</b><br>
                """
                for val, r in zip(size_values, radius_values):
                    # Small circle with text
                    legend_html += f'<i style="background: black; border-radius:50%; width:{2*r}px; height:{2*r}px; display:inline-block; margin-right:5px;"></i>{val:.1f}<br>'
                legend_html += '</div>'
                m.get_root().html.add_child(folium.Element(legend_html))
                
            else:
                points = points.assign(__radius=4)  # default radius

            # Style function for dynamic radius and no border
            style_fn = lambda feature: {
                "radius": feature["properties"]["__radius"],
                "color": None,        # no border
                "weight": 0,          # border thickness (0 = none)
                "fillOpacity": 1.0,   # full fill
                "opacity": 1.0,       # stroke opacity (irrelevant here)
            }

            if is_thematic(points, column, cmap):
                # Thematic coloring with variable size
                m = points.explore(
                    m=m,
                    column=column,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    legend=legend,
                    marker_type="circle_marker",
                    style_kwds={"style_function": style_fn},  # dynamic radius
                )
                if legend:
                    m = increase_legend_size(m)

                legend = False
            else:
                # Fixed color with variable size
                points = points.assign(__color=color)
                style_fn_fixed = lambda feature: {
                    "radius": feature["properties"]["__radius"],
                    "fillColor": feature["properties"]["__color"],
                    "color": None,
                    "weight": 0,
                    "fillOpacity": 1.0,
                    "opacity": 1.0,
                }
                m = points.explore(
                    m=m,
                    marker_type="circle_marker",
                    style_kwds={"style_function": style_fn_fixed},
                )

    # ==========================================================
    # POIS
    # ==========================================================
    legend_map = {}

    # Many pins would hide the map: group nearby POIs (they expand when zooming in).
    from folium.plugins import MarkerCluster

    poi_layer = m
    if cluster_pois and len(pois):
        poi_layer = MarkerCluster(
            name="Points of interest",
            options={"maxClusterRadius": 45, "showCoverageOnHover": False, "disableClusteringAtZoom": 16},
        ).add_to(m)

    for p in pois:

        polys, lines, points = split_geoms(p)

        # ---------------------------
        # POLYGONS + CENTROID ICON
        # ---------------------------
        if len(polys):
            # All polygons at once (one layer), then one emoji marker per polygon.
            folium.GeoJson(
                polys[["geometry"]].to_json(),
                style_function=lambda _f, c=poi_color or "#4c9f50": {
                    "color": c, "fillColor": c, "fillOpacity": 0.3, "weight": 1.5,
                },
            ).add_to(m)

        for _, row in polys.iterrows():

            geom = row.geometry
            centroid = geom.centroid

            icon_value = row.get(poi_icon_column) if poi_icon_column else None
            label_value = row.get(poi_column, "POI")

            fill = poi_color or "#4c9f50"

            bg = "white" if icon_value else fill

            html = f"""
            <div style="position:relative;width:34px;height:34px;">
                <div style="
                    position:absolute;
                    width:34px;height:34px;
                    background:{bg};
                    border:2px solid #555;box-shadow:0 2px 6px rgba(0,0,0,.35);
                    border-radius:50% 50% 50% 0;
                    transform:rotate(-45deg);
                "></div>

                <div style="
                    position:absolute;
                    top:50%;left:50%;
                    transform:translate(-50%,-50%);
                    font-size:18px;
                ">
                    {icon_value if icon_value else ""}
                </div>
            </div>
            """

            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(html=html, icon_size=(34, 34), icon_anchor=(17, 34)),
                tooltip=tooltip_html(row),
                popup=folium.Popup(tooltip_html(row), max_width=300),
            ).add_to(poi_layer)

            if icon_value:
                legend_map[str(label_value)] = icon_value

        # ---------------------------
        # LINES
        # ---------------------------
        for _, row in lines.iterrows():
            gpd.GeoDataFrame([row], geometry=[row.geometry], crs=4326).explore(
                m=m,
                color=color,
                style_kwds={"weight": 2},
            )

        # ---------------------------
        # POINTS
        # ---------------------------
        for _, row in points.iterrows():

            geom = row.geometry
            icon_value = row.get(poi_icon_column) if poi_icon_column else None
            label_value = row.get(poi_column, "POI")

            bg = "white" if icon_value else (poi_color or "blue")

            html = f"""
            <div style="position:relative;width:34px;height:34px;">
                <div style="
                    position:absolute;
                    width:34px;height:34px;
                    background:{bg};
                    border:2px solid #555;box-shadow:0 2px 6px rgba(0,0,0,.35);
                    border-radius:50% 50% 50% 0;
                    transform:rotate(-45deg);
                "></div>

                <div style="
                    position:absolute;
                    top:50%;left:50%;
                    transform:translate(-50%,-50%);
                    font-size:18px;
                ">
                    {icon_value if icon_value else ""}
                </div>
            </div>
            """

            folium.Marker(
                location=[geom.y, geom.x],
                icon=folium.DivIcon(html=html, icon_size=(34, 34), icon_anchor=(17, 34)),
                tooltip=tooltip_html(row),
                popup=folium.Popup(tooltip_html(row), max_width=300),
            ).add_to(poi_layer)

            if icon_value:
                legend_map[str(label_value)] = icon_value

    # ==========================================================
    # LEGEND (label + icon)
    # ==========================================================
    if legend_map:

        legend_html = """
        <div style="position:fixed;bottom:40px;left:40px;
        background:white;padding:10px;z-index:9999;">
        <b>POI Legend</b><br>
        """

        for label, icon in legend_map.items():
            legend_html += f"<div>{label}: {icon}</div>"

        legend_html += "</div>"

        m.get_root().html.add_child(folium.Element(legend_html))

    # ==========================================================
    # AOI (FIXED - ALWAYS DRAWN)
    # ==========================================================
    if aoi is not None:
        folium.GeoJson(
            aoi,
            name="AOI",
            style_function=lambda x: {
                "fillColor": "none",
                "color": "blue",
                "weight": 3,
                "dashArray": "5,5",
            },
        ).add_to(m)

    # Zoom to the study area (or to everything drawn if there is no AOI).
    bounds_geoms = [aoi.geometry] if aoi is not None else all_geoms
    minx, miny, maxx, maxy = pd.concat(bounds_geoms).total_bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    return m


def ipyleaflet_drawable_map(m=None, center=[0, 0], zoom=11, height="800px"):
    """
    Creates an interactive ipyleaflet map with drawing controls.

    Returns:
        - m: ipyleaflet map
        - get_drawn_geometries(): returns GeoDataFrame
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

def save_poi_map(path: str, aoi: gpd.GeoDataFrame, pois: gpd.GeoDataFrame) -> folium.Map:
    """Save a standalone HTML map (double-click to open) of POIs with their emoji icons.

    Args:
        path: Output ``.html`` file (folders are created).
        aoi: Study area.
        pois: Table with ``type``, ``name`` and ``icon`` columns, see
            :func:`UrbanAccessAnalyzer.grading.poi_table`.
    """
    import os

    m = general_map(aoi=aoi, pois=pois, poi_column="type", poi_icon_column="icon")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    m.save(path)
    return m


def _street_layer(
    m: folium.Map,
    edges: gpd.GeoDataFrame,
    score: str,
    cmap: str,
    vmin: float,
    vmax: float,
    categories: Optional[dict],
    n_classes: int = 10,
    tolerance_m: float = 3.0,
) -> None:
    """Street network as a light, non-interactive layer (no hover, no click).

    Streets are grouped by color class (a handful of features instead of one per street),
    duplicates are dropped and geometry is simplified, so the map file stays small.
    """
    import numpy as np

    gdf = edges[[score, "geometry"]].copy()
    if gdf.crs is None or gdf.crs.is_geographic:
        tol, precision = tolerance_m / 111_000, 1e-5
        gdf = gdf.to_crs(4326)
    else:
        tol, precision = tolerance_m, 0.5
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]

    if categories:
        colors = gdf[score].map(categories).fillna("#999999")
    else:
        cm = mpl_colormaps[cmap]
        frac = ((gdf[score].astype(float) - vmin) / (vmax - vmin)).clip(0, 1)
        cls = np.minimum((frac * n_classes).fillna(0).astype(int), n_classes - 1)
        palette = [mcolors.to_hex(cm((i + 0.5) / n_classes)) for i in range(n_classes)]
        colors = cls.map(lambda i: palette[i])

    features = []
    for color, part in gdf.groupby(colors.to_numpy()):
        geoms = shapely.get_parts(part.geometry.to_numpy())
        geoms = shapely.get_parts(shapely.simplify(geoms, tol))  # simpler lines
        geoms = geoms[~shapely.is_empty(geoms)]
        norm = shapely.normalize(geoms)  # same line in both directions -> one
        _, first = np.unique(shapely.to_wkb(norm), return_index=True)
        geoms = shapely.set_precision(geoms[np.sort(first)], precision)
        if gdf.crs.is_projected:  # back to lon/lat for the map
            geoms = gpd.GeoSeries(geoms, crs=gdf.crs).to_crs(4326).to_numpy()
            geoms = shapely.set_precision(geoms, 1e-5)
        features.append({
            "type": "Feature", "properties": {"c": color},
            "geometry": shapely.geometry.mapping(shapely.multilinestrings(geoms)),
        })

    folium.map.CustomPane("streets", z_index=390, pointer_events=False).add_to(m)  # below the circles, ignores the mouse
    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        pane="streets",
        style_function=lambda f: {"color": f["properties"]["c"], "weight": 1.2, "opacity": 0.55},
    ).add_to(m)



from branca.element import MacroElement as _MacroElement
from jinja2 import Template as _Template


class _Circles(_MacroElement):
    """Population circles as one compact layer: a shared color palette and one tooltip card style."""

    _template = _Template(
        """
        {% macro header(this, kwargs) %}
        <style>
          .uaa-tip { background:#fff; color:#222; border:none; border-radius:10px; padding:8px 12px;
                     box-shadow:0 2px 12px rgba(0,0,0,.25); font:13px system-ui,-apple-system,'Segoe UI',sans-serif; }
          .uaa-tip td { padding:1px 6px 1px 0; } .uaa-tip td:last-child { font-weight:600; }
        </style>
        {% endmacro %}
        {% macro script(this, kwargs) %}
        (function() {
          var d = {{ this.data|tojson }}, map = {{ this._parent.get_name() }};
          var group = L.layerGroup();
          d.pts.forEach(function(p) {
            var c = L.circleMarker([p[0], p[1]], {radius: p[2], fillColor: d.colors[p[3]], color: '#444',
                                                  weight: 0.7, fillOpacity: 0.85});
            var rows = '';
            for (var i = 0; i < d.labels.length; i++) { rows += '<tr><td>' + d.labels[i] + '</td><td>' + p[4][i] + '</td></tr>'; }
            c.bindTooltip('<table>' + rows + '</table>', {sticky: true, className: 'uaa-tip', opacity: 1});
            group.addLayer(c);
          });
          group.addTo(map);
        })();
        {% endmacro %}
        """
    )

    def __init__(self, data):
        super().__init__()
        self._name = "Circles"
        self.data = data

def _circle_layer(
    m: folium.Map,
    hexes: gpd.GeoDataFrame,
    labels: dict,
    score: str,
    cmap: str,
    vmin: float,
    vmax: float,
    categories: Optional[dict],
    max_radius: float = 7.0,
) -> None:
    """One circle per hexagon center: size = population, color = score. Adds the legends too."""
    import html as _html

    import branca.colormap as bcm

    pop_label = labels["population"]
    hexes = hexes[hexes[pop_label] > 0].copy()
    if hexes.empty:
        return
    utm = hexes.estimate_utm_crs()
    hexes["geometry"] = hexes.to_crs(utm).centroid.to_crs(4326)

    # Size: area proportional to population (radius ~ sqrt), capped at the 98th percentile.
    pop = hexes[pop_label].astype(float)
    ref = max(pop.quantile(0.98), 1.0)
    radius = 2 + (max_radius - 2) * np.sqrt((pop / ref).clip(upper=1.0))

    # Color: a small palette (categories, or 16 steps of the color map) instead of one color per circle.
    values = hexes[labels[score]]
    if categories:
        palette = list(categories.values()) + ["#999999"]
        index = values.map({k: i for i, k in enumerate(categories)}).fillna(len(categories)).astype(int)
    else:
        cm, n = mpl_colormaps[cmap], 16
        palette = [mcolors.to_hex(cm((i + 0.5) / n)) for i in range(n)] + ["#999999"]
        frac = ((values.astype(float) - vmin) / (vmax - vmin)).clip(0, 1)
        index = np.minimum((frac * n).fillna(n).astype(int), n).where(values.notna(), n)

    def fmt(v):
        return _format_value(v)

    cols = list(labels)
    cols_in = [c for c in labels.values() if c in hexes.columns]
    cen = hexes.geometry
    pts = [
        [round(g.y, 6), round(g.x, 6), round(float(r), 1), int(i), [fmt(row[c]) for c in cols_in]]
        for g, r, i, (_, row) in zip(cen, radius, index, hexes[cols_in].iterrows())
    ]
    _Circles({"labels": cols_in, "colors": palette, "pts": pts}).add_to(m)

    # Legends: color scale (or category list) and circle sizes.
    if categories:
        rows = "".join(
            f"<div><span style='display:inline-block;width:12px;height:12px;border-radius:50%;"
            f"background:{c};margin-right:6px'></span>{_html.escape(str(k))}</div>"
            for k, c in categories.items()
        )
        m.get_root().html.add_child(folium.Element(
            "<div style='position:fixed;top:12px;right:12px;z-index:9999;background:white;padding:8px 12px;"
            "border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,.25);font:13px system-ui,sans-serif'>"
            f"<b>{_html.escape(labels[score])}</b>{rows}</div>"
        ))
    else:
        colors = [mcolors.to_hex(mpl_colormaps[cmap](i)) for i in np.linspace(0, 1, 12)]
        bcm.LinearColormap(colors, vmin=vmin, vmax=vmax, caption=labels[score]).add_to(m)

    samples = [pop.quantile(0.2), pop.quantile(0.6), ref]
    sizes = [2 + (max_radius - 2) * np.sqrt(min(v / ref, 1.0)) for v in samples]
    circles = "".join(
        f"<div style='display:flex;align-items:center;gap:8px;margin-top:2px'>"
        f"<span style='display:inline-block;width:{2 * r:.0f}px;height:{2 * r:.0f}px;border-radius:50%;"
        f"background:#bbb;border:1px solid white;box-shadow:0 0 0 1px #999'></span>{v:,.0f}</div>"
        for r, v in zip(sizes, samples)
    )
    m.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;bottom:28px;right:12px;z-index:9999;background:white;padding:8px 12px;"
        "border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,.25);font:13px system-ui,sans-serif'>"
        f"<b>{_html.escape(pop_label)}</b>{circles}</div>"
    ))


def save_access_map(
    path: str,
    hexes: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    labels: dict,
    score: str,
    edges: Optional[gpd.GeoDataFrame] = None,
    pois: Optional[gpd.GeoDataFrame] = None,
    cmap: str = "RdYlGn",
    vmin: float = 0,
    vmax: float = 100,
    categories: Optional[dict] = None,
    shape: str = "circles",
) -> folium.Map:
    """Save a standalone HTML map (double-click to open) of an access score and the population.

    By default each H3 hexagon is a circle: its size shows the population and its color the score.

    Args:
        path: Output ``.html`` file (folders are created).
        hexes: Hexagons, see :func:`UrbanAccessAnalyzer.h3_population.h3_with_population`.
        aoi: Study area.
        labels: ``{column: name shown on hover}`` in display order, e.g.
            ``{"population": "👥 Population", "access_score": "🚶 Walkability (0-100)"}``.
        score: Key of ``labels`` used to color the map.
        edges: Optional street edges (with the ``score`` column), drawn as a light layer without
            hover or click, so the file stays small.
        pois: Optional POI table (``type``, ``name``, ``icon``) drawn as emoji pins.
        cmap: Color map (default red-yellow-green).
        vmin: Score value of the first color.
        vmax: Score value of the last color.
        shape: ``"circles"`` (default: size = population, color = score) or ``"hexagons"``.
        categories: For a text ``score`` column: ``{label: color}``; the legend then lists the
            labels (no numeric scale).
    """
    import os

    def view(gdf):
        cols = [c for c in labels if c in gdf.columns]
        out = gdf.to_crs(4326)[[*cols, "geometry"]].rename(columns=labels).round(1)
        if "population" in cols:
            out[labels["population"]] = out[labels["population"]].fillna(0).round().astype(int)
        return out

    kwargs = dict(pois=pois, poi_column="type", poi_icon_column="icon") if pois is not None else {}
    if shape == "hexagons":
        m = general_map(
            aoi=aoi, gdfs=[view(hexes)], column=labels[score], cmap=cmap, vmin=vmin, vmax=vmax,
            categories=categories, **kwargs,
        )
    else:
        m = general_map(aoi=aoi, **kwargs)
    if edges is not None:
        _street_layer(m, edges, score, cmap, vmin, vmax, categories)
    if shape != "hexagons":
        _circle_layer(m, view(hexes), labels, score, cmap, vmin, vmax, categories)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    m.save(path)
    return m
