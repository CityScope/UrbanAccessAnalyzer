import folium
import pandas as pd
import geopandas as gpd
from typing import Optional, List, Union
from matplotlib import colormaps as mpl_colormaps
import matplotlib.colors as mcolors
import numpy as np
import ipyleaflet
from ipyleaflet import DrawControl, Map
from shapely.geometry import shape

from . import h3_utils


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

            elif (
                isinstance(g, pd.DataFrame)
                and (("h3_cell" in g.columns) or ("h3_cell" == g.index.name))
            ):
                g = h3_utils.to_gdf(g).to_crs(4326)

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
    # MAP (TRUE B&W BASEMAP)
    # ==========================================================
    if m is None:
        m = folium.Map(
            location=[centroid.y, centroid.x],
            zoom_start=zoom_start,
            tiles="https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png",
            attr="CartoDB",
            control_scale=True,
        )

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
        html = "<table style='width:260px'>"
        for k, v in row.items():
            if k != "geometry":
                html += f"""
                <tr>
                    <th style='text-align:left;padding-right:8px'>{k}</th>
                    <td>{v}</td>
                </tr>
                """
        html += "</table>"
        return html

    # ------------------------------------------------------------------
    # vmin / vmax for gdfs
    # ------------------------------------------------------------------
    if poi_cmap is None:
        poi_cmap = cmap 

    if poi_color is None:
        poi_color = color 

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
    legend = True
    for g in gdfs:
        polys, lines, points = split_geoms(g)

        # Polygons
        if not polys.empty:
            if is_thematic(polys, column, cmap):
                m = polys.explore(
                    m=m,
                    column=column,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    legend=legend,
                    style_kwds={"color": None, "weight": 0, "fillOpacity": opacity},
                )
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
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    legend=legend,
                    style_kwds={"weight": 2},
                )
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
                legend_html = """
                    <div style="position: fixed; bottom: 40px; right: 40px; z-index:9999; 
                    background:white; padding:10px; border-radius:5px; 
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
                    <b>Population Legend</b><
                """
                legend_html += f'<b>{size_column}</b><br>'
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

    for p in pois:

        polys, lines, points = split_geoms(p)

        # ---------------------------
        # POLYGONS + CENTROID ICON
        # ---------------------------
        for _, row in polys.iterrows():

            geom = row.geometry
            centroid = geom.centroid

            icon_value = row.get(poi_icon_column) if poi_icon_column else None
            label_value = row.get(poi_column, "POI")

            fill = poi_color or "darkgreen"

            gpd.GeoDataFrame([row], geometry=[geom], crs=4326).explore(
                m=m,
                color="black",
                style_kwds={
                    "fillColor": fill,
                    "fillOpacity": 1.0,
                    "weight": 2,
                },
            )

            bg = "white" if icon_value else fill

            html = f"""
            <div style="position:relative;width:34px;height:34px;">
                <div style="
                    position:absolute;
                    width:34px;height:34px;
                    background:{bg};
                    border:2px solid black;
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
            ).add_to(m)

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
                    border:2px solid black;
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
            ).add_to(m)

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

    return m


def ipyleaflet_drawable_map(m=None, center=[0, 0], zoom=11, height="800px"):
    """
    Creates an interactive ipyleaflet map with drawing controls.

    Returns:
        - m: ipyleaflet map
        - get_drawn_geometries(): returns GeoDataFrame
    """

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