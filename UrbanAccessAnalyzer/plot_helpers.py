import folium
import pandas as pd
import geopandas as gpd
from typing import Optional, List, Union
from matplotlib import colors, colormaps as mpl_colormaps
import matplotlib.colors as colors
from folium.plugins import BeautifyIcon
import numpy as np 

from . import h3_utils

def general_map(
    m: Optional[folium.Map] = None,
    aoi: Optional[gpd.GeoDataFrame] = None,
    pois: Optional[Union[gpd.GeoDataFrame, List[gpd.GeoDataFrame]]] = None,
    gdfs: Optional[Union[gpd.GeoDataFrame, pd.DataFrame, List[Union[gpd.GeoDataFrame, pd.DataFrame]]]] = None,
    poi_column: Optional[str] = None,
    poi_color: Optional[str] = None,
    poi_cmap: Optional[str] = None,
    poi_vmin: Optional[float] = None,
    poi_vmax: Optional[float] = None,
    poi_opacity: float = 1.0,
    column: Optional[str] = None,
    color: str = "black",
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    opacity: float = 0.4,
    size_column: Optional[str] = None,
    zoom_start=11,
) -> folium.Map:
    """
    General-purpose interactive Folium map builder.

    - Supports polygons, lines, points
    - AOI clipping
    - Multiple GeoDataFrames
    - Optional thematic coloring
    - Optional point-size scaling
    """
    if pois is None:
        pois = []
    if gdfs is None:
        gdfs = []
    # ------------------------------------------------------------------
    # CRS normalization
    # ------------------------------------------------------------------
    if aoi is not None:
        aoi = aoi.to_crs(4326)

    def _normalize_gdfs(objs):
        out = []
        for g in objs:
            if isinstance(g, gpd.GeoDataFrame):
                g = g.to_crs(4326)
            elif (
                isinstance(g, pd.DataFrame) and 
                ("h3_cell" in g.columns or ("h3_cell" == g.index.name))
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

    if len(gdfs) == 0 and len(pois) == 0 and (aoi is None):
        raise ValueError("Nothing to map")
    
    # ------------------------------------------------------------------
    # Map centering
    # ------------------------------------------------------------------
    if aoi is not None:
        centroid = aoi.union_all().centroid
    elif pois:
        centroid = pd.concat([p.geometry for p in pois]).union_all().centroid
    else:
        centroid = pd.concat([g.geometry for g in gdfs]).union_all().centroid

    if m is None:
        m = folium.Map(
            location=[centroid.y, centroid.x],
            zoom_start=zoom_start,
            tiles="CartoDB positron",
        )

    if aoi is not None:
        if m is None:
            m = aoi.explore(
                color="blue",
                fill=False,
                style_kwds={"weight": 4, "dashArray": "5,5", "opacity": 1.0},
            )
        else:
            m = aoi.explore(
                m=m,
                color="blue",
                fill=False,
                style_kwds={"weight": 4, "dashArray": "5,5", "opacity": 1.0},
            )

        if len(pois) > 0:
            pois = [p[p.intersects(aoi.union_all())] for p in pois]
        if len(gdfs) > 0:
            gdfs = [g[g.intersects(aoi.union_all())] for g in gdfs]
    else:
        m = folium.Map(
            location=[centroid.y, centroid.x],
            zoom_start=12,
            tiles="CartoDB positron",
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
                legend_html = '<div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; background:white; padding:10px; border-radius:5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">'
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



    # ------------------------------------------------------------------
    # POIs
    # ------------------------------------------------------------------
    legend = True
    for p in pois:
        polys, lines, points = split_geoms(p)

        if not polys.empty:
            if is_thematic(polys, poi_column, poi_cmap):
                m = polys.explore(
                    m=m,
                    column=poi_column,
                    cmap=poi_cmap,
                    vmin=poi_vmin,
                    vmax=poi_vmax,
                    legend=legend,
                    style_kwds={"color": "black","fillOpacity": poi_opacity,"weight": 1,},
                )
                legend = False
            else:
                m = polys.explore(
                    m=m,
                    style_kwds={
                        "color": "black",
                        "fillColor": poi_color,
                        "fillOpacity": poi_opacity,
                        "weight": 1,
                    },
                )
        if not lines.empty:
            if is_thematic(lines, poi_column, poi_cmap):
                m = lines.explore(
                    m=m,
                    column=poi_column,
                    cmap=poi_cmap,
                    vmin=poi_vmin,
                    vmax=poi_vmax,
                    legend=legend,
                    style_kwds={"weight": 2},
                )
            else:
                m = lines.explore(
                    m=m,
                    color=poi_color,
                    style_kwds={"weight": 2},
                )

        if not points.empty:
            # Determine if theming is active
            thematic = is_thematic(points, poi_column, poi_cmap)

            # Handle categorical color mapping
            color_map = {}
            if thematic and points[poi_column].dtype == "object":
                categories = points[poi_column].unique()
                cmap = mpl_colormaps[poi_cmap]
                color_map = {
                    cat: colors.to_hex(cmap(i / len(categories))) for i, cat in enumerate(categories)
                }

            # Prepare colormap for numeric data if needed
            elif thematic:
                cmap = mpl_colormaps[poi_cmap]
                norm = colors.Normalize(vmin=poi_vmin, vmax=poi_vmax)

            # Compute colors safely
            def compute_color(row):
                if thematic:
                    if points[poi_column].dtype == "object":
                        color_val = color_map.get(row[poi_column], "#000000")
                    else:
                        color_val = colors.to_hex(cmap(norm(row[poi_column])))
                else:
                    color_val = poi_color
                return str(color_val) if color_val else "#000000"

            # Assign colors to _color column
            points["_color"] = points.apply(compute_color, axis=1).astype(str)

            # Convert to EPSG:4326 for Folium
            points_geojson = points.to_crs(4326).copy()

            # Keep only JSON-serializable columns + _color + geometry
            serializable_cols = [
                c for c in points_geojson.columns
                if c != points_geojson.geometry.name
                and points_geojson[c].apply(lambda x: isinstance(x, (str, int, float, type(None)))).all()
            ]
            for col in list(set(points_geojson.columns) - set(serializable_cols)):
                if col == points_geojson.geometry.name:
                    continue 
                
                try:
                    points_geojson[col] = points_geojson[col].astype(str)
                    serializable_cols.append(col)
                except:
                    None

            points_geojson = points_geojson[serializable_cols + [points_geojson.geometry.name]]

            # Add proper Markers using BeautifyIcon
            for _, row in points_geojson.iterrows():
                tooltip_text = "<br>".join(f"{c}: {row[c]}" for c in serializable_cols if c != "_color")
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    icon=BeautifyIcon(
                        icon="circle",
                        icon_shape="marker",
                        background_color=row["_color"],
                        border_color="black",
                        text_color="white",
                    ),
                    tooltip=tooltip_text,
                    legend=legend
                ).add_to(m)
                legend=False

    # ------------------------------------------------------------------
    # AOI outline & clipping
    # ------------------------------------------------------------------
    if aoi is not None:
        m = aoi.explore(
            m=m,
            color="blue",
            fill=False,
            style_kwds={"weight": 4, "dashArray": "5,5", "opacity": 1.0},
        )
        
    return m