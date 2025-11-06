import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as cx
import h3
import shapely.geometry as geom
import folium
import branca.colormap as cm


def plot_h3_df(
    df: pd.DataFrame,
    value_column: str | None = None,
    h3_column: str = "h3_cell",
    ax=None,
    cmap: str = "viridis",
    alpha: float = 0.7,
    legend: bool = True
):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import contextily as cx
    import shapely.geometry as geom
    import h3

    df = df.copy()

    # Validate
    if h3_column not in df.columns:
        raise ValueError(f"DataFrame must contain an '{h3_column}' column with H3 indices.")

    # Ensure all cells are valid strings
    df[h3_column] = df[h3_column].astype(str)

    # ✅ Use the correct validation function for h3>=4
    df = df[df[h3_column].apply(h3.is_valid_cell)]

    # Convert each H3 cell to a polygon geometry
    def cell_to_polygon(cell):
        boundary = h3.cell_to_boundary(cell)
        return geom.Polygon([(lng, lat) for lat, lng in boundary])  # note lat/lng order flip

    df["geometry"] = df[h3_column].apply(cell_to_polygon)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)

    # Prepare axis
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.set_axis_off()

    # Plot polygons
    gdf.plot(
        ax=ax,
        column=value_column,
        cmap=cmap if value_column else None,
        alpha=alpha,
        edgecolor="k",
        linewidth=0.3,
        legend=legend if value_column else False,
        legend_kwds={"loc": "upper left"} if value_column else None,
    )

    cx.add_basemap(ax, crs=gdf.crs, source=cx.providers.CartoDB.Positron)

    return ax



def h3_map(
    df: pd.DataFrame,
    value_column: str | None = None,
    h3_column: str = "h3_cell",
    cmap: str = "viridis",
    alpha: float = 0.6,
    tiles: str = "cartodbpositron",
    m=None
) -> folium.Map:
    """
    Create an interactive folium map from a DataFrame of H3 cells.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain an H3 column (default 'h3_cell').
        Optionally include a `value_column` for coloring.
    value_column : str, optional
        Column name to color cells by (e.g., category, numeric value).
    h3_column : str, optional
        Column containing H3 cell indexes (default: 'h3_cell').
    cmap : str, optional
        Colormap name for numeric values (default: 'viridis').
    alpha : float, optional
        Polygon fill opacity (default: 0.6).
    tiles : str, optional
        Folium basemap style (default: 'cartodbpositron').

    Returns
    -------
    folium.Map
        Interactive map with hexagons rendered.
    """
    df = df.copy()

    # --- Validation ---
    if h3_column not in df.columns:
        raise ValueError(f"DataFrame must contain an '{h3_column}' column with H3 indices.")

    df[h3_column] = df[h3_column].astype(str)

    # Detect proper validation function (for H3 v3/v4)
    if hasattr(h3, "is_valid_cell"):
        is_valid = h3.is_valid_cell
    elif hasattr(h3, "h3_is_valid"):
        is_valid = h3.h3_is_valid
    else:
        raise AttributeError("Cannot find a valid H3 validation function in the h3 module.")

    df = df[df[h3_column].apply(is_valid)]

    # --- Build geometries ---
    def cell_to_polygon(cell):
        boundary = h3.cell_to_boundary(cell)
        return geom.Polygon([(lng, lat) for lat, lng in boundary])

    df["geometry"] = df[h3_column].apply(cell_to_polygon)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    # --- Initialize Folium map centered on data ---
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    if m is None:
        m = folium.Map(location=center, tiles=tiles, zoom_start=12)

    # --- Handle color mapping ---
    if value_column:
        if pd.api.types.is_numeric_dtype(df[value_column]):
            # Continuous colormap
            colormap = cm.linear.__getattribute__(cmap).scale(
                df[value_column].min(), df[value_column].max()
            )
            colormap.caption = value_column
            colormap.add_to(m)
            get_color = lambda val: colormap(val)
        else:
            # Categorical colormap
            categories = [cat for cat in df[value_column].unique() if pd.notna(cat)]
            n = len(categories)
            # Use a qualitative color palette from branca or generate one
            palette = cm.Set3_09.to_step(n).colors if n <= 9 else cm.tab20.to_step(n).colors
            color_map = dict(zip(categories, palette))
            get_color = lambda val: color_map.get(val, "#999999")  # fallback for None or unknown
    else:
        get_color = lambda val: "#3186cc"

    # --- Add polygons to map ---
    for _, row in gdf.iterrows():
        color = get_color(row[value_column]) if value_column else "#3186cc"
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda feature, color=color: {
                "fillColor": color,
                "color": "black",
                "weight": 0.5,
                "fillOpacity": alpha,
            },
            tooltip=(
                f"{h3_column}: {row[h3_column]}"
                + (f"<br>{value_column}: {row[value_column]}" if value_column else "")
            ),
        ).add_to(m)

    return m