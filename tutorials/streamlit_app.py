"""
TODO: raster map does not have legends 
TODO: level of service map when clicking one level of service polygon level of service should appear
TODO: population matrix should not be shown
TODO: Map is reloading all the time
TODO: Basic statistics population with each level of service and population not meeting requirements
"""


import sys
sys.path.append('/home/miguel/Documents/Proyectos/PTLevelofService/gtfs/pyGTFSHandler')
sys.path.append('/home/miguel/Documents/Proyectos/PTLevelofService/accessibility/UrbanAccessAnalyzer')

import os
from datetime import datetime, date, timedelta, time
from typing import Union, Tuple, Optional
import streamlit as st
import requests, folium
from streamlit_folium import st_folium
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import numpy as np
import rasterio
import osmnx as ox 
import pyproj
import io, base64
import re
import branca

from pyGTFSHandler.feed import Feed
from pyGTFSHandler.downloaders.mobility_database import (
    MobilityDatabaseClient, get_geographic_suggestions_from_string
)
import pyGTFSHandler.plot_helper as plot_helper
import pyGTFSHandler.processing_helper as processing_helper
import UrbanAccessAnalyzer.utils as utils
import UrbanAccessAnalyzer.population as population 
import UrbanAccessAnalyzer.isochrones as isochrones 
import UrbanAccessAnalyzer.raster_utils as raster_utils

# === CONFIG ===
st.set_page_config(page_title="Public Transport Level Of Service", layout="wide")

START_HOUR, END_HOUR = 8, 20
BUFFER_LOS = 2000
BUFFER_POP = 3000
WORK_DIR = "/home/miguel/Documents/Proyectos/PTLevelofService/accessibility/UrbanAccessAnalyzer/no_sync"
USER_AGENT = "urban-access-analyzer/1.0"
MOBILITY_DB_REFRESH_TOKEN = 'AMf-vByYiwMAni1pw6yTpwgwwYFc8HR4y0zUKZGPT4sjJ0wUrIXOfVxF1KotRIvEgAseaaNheL8YczJiCILb6o2PUh-8zjA-qQURzEc8tELlwFiDopMoqJnkDf13AqNaGGnnzTDmYM20AWEquUxcYFAB8Q3e5rI2DcTBSQuiUdHL8bi48xmUJk3tayHpnoicoppi_evDcWYODwOJFcwnta3K7f718w7R2JRM0zDEOYw7nI7thrQa9462BENdpv8zv8mEbBssEa189k6YcV__sQAZlng2EcsCGA'

# Custom constants for raster analysis
MIN_POP_DENSITY = 300 
TRANSIT_DESERT_COLOR = '#FF69B4' # Hot Pink for unmet demand (Transit Desert)

# Grades for LOS (A1 to F, 12 levels)
LOS_GRADES = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D', 'E', 'F']

# Custom colors for Distance/Density matrices
GRADE_COLOR_MAP = {
    'A': ['#b7e4c7', '#74c69d', '#2d6a4f'],  # Light mint → medium → deep green
    'B': ['#fff3b0', '#ffd60a', '#e0a800'],  # Light buttery → bright gold → rich amber
    'C': ['#ffd6a5', '#ff924c', '#d9480f'],  # Light peach → medium orange → deep burnt orange
    'D': ['#ff6b6b'],                        # Strong red (poor)
    'E': ['#b983ff'],                        # Purple (very poor)
    'F': ["#1F00D2"],                        # Gray (fail)
    None: ['transparent']
}

# --- DENSITY MATRIX DEFINITION (Used for Calculation) ---
DENSITY_MATRIX_CALC = pd.DataFrame(
    {
        'density':[15000,7500,3500,2500,1500,300],
        3000:     ['A1' ,'A2','B1','B3','C2','D'], 
        2500:     ['A2' ,'A3','B2','C1','C3','E'],        
        2000:     ['A3' ,'B1','B3','C2','D' ,'F'],                        
        750:      ['B1' ,'B3','C2','C3','E' ,'F'],
        500:      ['B2' ,'C1','C3','D' ,'E' , None],
        300:      ['B3' ,'C2','D' ,'E' ,None,  None],
    }
)


# --- DENSITY MATRIX DEFINITION (Used for Display) ---
density_matrix_display_data = pd.DataFrame(
    {
        'Density (Pop/km²)': DENSITY_MATRIX_CALC['density'].tolist(),
        **{f'{col}m': DENSITY_MATRIX_CALC[col].tolist() for col in DENSITY_MATRIX_CALC.columns}
    }
)


# === SESSION STATE ===
defaults = {
    "confirmed_address": None, "last_input": "", "gtfs_ready": False,
    "service_quality_gdf": None, "gtfs": None,
    "aoi": None, "aoi_download": None, "aoi_pop_download": None, "geo": None, "raster_ready": False,
    "level_of_service_gdf": None, "stop_quality_gdf_cropped": None,
    "population_file": None, # ADDED: To store cached population file path
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Color Maps ---
cmap_blue = matplotlib.colormaps["Blues_r"]

# === HELPERS ===

# FIX: Removed st.toast/st.error calls from inside the cached function 
# to resolve CacheReplayClosureError. Now returns file path and error status.
@st.cache_resource(show_spinner=False)
def get_population_file_cached(_aoi_buffer, folder_path) -> Tuple[Optional[str], Optional[str]]:
    """Handles WorldPop download using caching/background execution."""
    os.makedirs(folder_path, exist_ok=True)
    try:
        population_file = population.download_worldpop_population(
            _aoi_buffer, 2025, folder=folder_path, resolution="100m"
        )
        return population_file, None # Success: file path and no error
    except Exception as e:
        return None, str(e) # Failure: None path and error message

def get_city_geometry(city_name: str) -> gpd.GeoDataFrame:
    try:
        gdf = ox.geocode_to_gdf(city_name)
        gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception as e:
        raise ConnectionError(f"Failed to fetch city geometry using OSMnx: {e}")

def get_suggestions(q):
    if not q or len(q)<3: return []
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q":q,"format":"json","limit":5},
                         headers={"User-Agent": USER_AGENT}, timeout=8)
        return r.json() if r.status_code==200 else []
    except: return []

def geocode_one(q):
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q":q,"format":"json","limit":1},
                         headers={"User-Agent": USER_AGENT}, timeout=8)
        d = r.json()
        if d: return float(d[0]["lat"]), float(d[0]["lon"]), d[0]["display_name"]
    except: return None

def to_roman(n):
    vals = [(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    out = ""
    if n > 10: out += 'X' * (n // 10); n %= 10
    for v,s in vals:
        while n>=v: out+=s; n-=v
    return out

def sanitize_filename(s): return utils.sanitize_filename(s)

def get_grade_color_and_index(grade_str):
    if not isinstance(grade_str, str): 
        return 'transparent', None, None
    grade_str = grade_str.strip()
    
    try:
        if grade_str.startswith('A'):
            idx = int(grade_str[1]) - 1
            rgb = GRADE_COLOR_MAP['A'][idx]
            roman_index = idx + 1 
        elif grade_str.startswith('B'):
            idx = int(grade_str[1]) - 1
            rgb = GRADE_COLOR_MAP['B'][idx]
            roman_index = idx + 4 
        elif grade_str.startswith('C'):
            idx = int(grade_str[1]) - 1
            rgb = GRADE_COLOR_MAP['C'][idx]
            roman_index = idx + 7 
        elif grade_str == 'D':
            rgb = GRADE_COLOR_MAP['D'][0]
            roman_index = 10 
        elif grade_str == 'E':
            rgb = GRADE_COLOR_MAP['E'][0]
            roman_index = 11 
        elif grade_str == 'F':
            rgb = GRADE_COLOR_MAP['F'][0]
            roman_index = 12 
        else:
            rgb = 'transparent'
            roman_index = None
    except IndexError:
        rgb = 'transparent'
        roman_index = None
        
    los_index = roman_index - 1 if roman_index is not None else None 
        
    return rgb, roman_index, los_index

def format_grade_cell(grade_str, as_roman=False):
    color, roman, _ = get_grade_color_and_index(grade_str)
    
    if grade_str is None or grade_str == 'None' or grade_str == 'NaN' or pd.isna(grade_str):
        content = ""
        bg_color = "white"
    else:
        content = grade_str
        if as_roman and roman is not None:
             content = to_roman(roman)
        
        bg_color = color if color != 'transparent' else 'white'
        
    return f"<span style='background-color:{bg_color}; font-weight:bold; padding: 2px 4px;'>{content}</span>"

def roman_color_blue(val):
    try:
        v = float(val)
        if v == 0:
            rgb = "black"
            content = "No service"
        else:
            norm = (v-1)/(12-1)
            rgba = cmap_blue(norm)
            rgb = f"rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})"
            content = to_roman(int(v))
            
        return f"<span style='color:{rgb}; font-weight:bold; font-size:1.1em'>{content}</span>"
    except: return val


def get_color(val):
    if val==0: return "#000000"
    cmap = matplotlib.colormaps["Blues_r"]
    norm = (val-1)/(12-1)
    rgba = cmap(norm)
    return matplotlib.colors.rgb2hex(rgba)


def plot_service_intensity(service_intensity):
    if not isinstance(service_intensity, pd.DataFrame):
        pdf = service_intensity.to_pandas()
    else:
        pdf = service_intensity.copy()
    pdf["date"] = pd.to_datetime(pdf["date"])
    plt.figure(figsize=(12,6))
    
    bar_width=0.8
    weekend_hatch = "xx"
    holiday_hatch = "//"

    provider_map = {}
    if "file_id" in pdf.columns and hasattr(st.session_state, "gtfs_providers"):
        unique_files = sorted(pdf["file_id"].unique())
        provider_list = st.session_state.gtfs_providers
        provider_map = {fid: provider_list[i] if i < len(provider_list) else f"File {fid}"
                        for i, fid in enumerate(unique_files)}
        
        color_cycle = plt.cm.tab20.colors
        color_map = {fid: color_cycle[i%len(color_cycle)] for i,fid in enumerate(unique_files)}

        grouped = pdf.groupby("date")
        for date, group in grouped:
            bottom=0
            for _, row in group.iterrows():
                val = row["service_intensity"]
                fid = row["file_id"]
                color = color_map[fid]
                hatch = holiday_hatch if row.get("holiday",False) else weekend_hatch if row.get("weekend",False) else None
                plt.bar(date,val,width=bar_width,bottom=bottom,color=color,hatch=hatch,edgecolor="black")
                bottom+=val

        # Use provider names as legend labels
        file_handles = [plt.Line2D([],[],color=color_map[fid],label=provider_map[fid],linewidth=10) for fid in unique_files]
        pattern_handles = [mpatches.Patch(facecolor="white",hatch=weekend_hatch,edgecolor="black",label="Weekend"),
                           mpatches.Patch(facecolor="white",hatch=holiday_hatch,edgecolor="black",label="Holiday")]
        plt.legend(handles=file_handles+pattern_handles,title="Legend",bbox_to_anchor=(1.05,1),loc="upper left")
    else:
        base_color="#a6c8ff"
        for _,row in pdf.iterrows():
            hatch = holiday_hatch if row.get("holiday",False) else weekend_hatch if row.get("weekend",False) else None
            plt.bar(row["date"],row["service_intensity"],width=bar_width,color=base_color,hatch=hatch,edgecolor="black")
        pattern_handles=[plt.Line2D([],[],color=base_color,label="Weekday",linewidth=10),
                         mpatches.Patch(facecolor=base_color,hatch=weekend_hatch,edgecolor="black",label="Weekend"),
                         mpatches.Patch(facecolor=base_color,hatch=holiday_hatch,edgecolor="black",label="Holiday")]
        plt.legend(handles=pattern_handles,title="Legend",bbox_to_anchor=(1.05,1),loc="upper left")

    plt.xlabel("Date")
    plt.ylabel("Service Intensity")
    plt.title("Service Intensity Over Time")
    plt.grid(True,axis="y",linestyle="--",alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()


def color_distance_matrix(df_display):
    df_numerical = processing_helper.DISTANCE_MATRIX.copy()
    styled_df = df_display.copy()
    styled_df['Service Quality'] = df_numerical['service_quality'].apply(roman_color_blue)
    numeric_cols = [c for c in styled_df.columns if c != "Service Quality"]
    for col in numeric_cols:
        styled_df[col] = styled_df[col].apply(lambda x: format_grade_cell(x, as_roman=False))
    return styled_df

def color_density_matrix(df_display):
    styled_df = df_display.copy()
    numeric_cols = [c for c in styled_df.columns if c != "Density (Pop/km²)"]
    for col in numeric_cols:
        styled_df[col] = styled_df[col].apply(lambda x: format_grade_cell(x, as_roman=False))
    return styled_df

def style_buffer_los_color(feature):
    grade = feature['properties'].get('level_of_service')
    if not isinstance(grade, str) or grade.upper() == 'NONE': grade = None
    color, _, _ = get_grade_color_and_index(grade)
    fill_color = color if color != 'transparent' else 'none'
    return {'fillColor': fill_color, 'color': 'none', 'weight': 0, 'fillOpacity': 0.5}

# =============================================================================
# Streamlit UI Flow
# =============================================================================

# === SIDEBAR: Address search ===
with st.sidebar:
    st.header("📍 Address")
    addr_input = st.text_input("Search:", value=st.session_state.last_input)
    if addr_input != st.session_state.last_input:
        st.session_state.last_input = addr_input
        st.session_state.confirmed_address = None
        st.session_state.gtfs_ready = False
        st.session_state.raster_ready = False 
        st.session_state.population_file = None # RESET POPULATION CACHE

# === STEP 0: Address Confirmation ===
if st.session_state.confirmed_address is None:
    if addr_input:
        sugs = get_suggestions(addr_input)
        for i,s in enumerate(sugs):
            if st.button(s["display_name"],key=f"s{i}"):
                res = geocode_one(s["display_name"])
                if res:
                    lat,lon,disp = res
                    st.session_state.confirmed_address = {"lat":lat,"lon":lon,"display_name":disp}
                    try:
                        geo = get_geographic_suggestions_from_string(disp)
                        aoi = get_city_geometry(disp) 
                        st.session_state.geo = geo
                        st.session_state.aoi = aoi
                        st.session_state.aoi_download = aoi.to_crs(aoi.estimate_utm_crs()).buffer(BUFFER_LOS)
                        st.session_state.aoi_pop_download = aoi.to_crs(aoi.estimate_utm_crs()).buffer(BUFFER_POP)
                    except Exception as e:
                        st.warning(f"AOI could not be loaded: {e}")
                    st.rerun() # Rerun to stabilize confirmed address display
    
# === STEP 1: GTFS Download ===
if st.session_state.confirmed_address:
    addr = st.session_state.confirmed_address
    st.subheader(addr["display_name"])
    
    city_filename = sanitize_filename(addr["display_name"])
    level_of_service_path = os.path.join(WORK_DIR, city_filename, "level_of_service")
    population_folder = os.path.join(WORK_DIR, "population")
    
    # Map of AOI
    fmap = folium.Map(location=[addr["lat"],addr["lon"]],zoom_start=13, tiles="CartoDB Positron")
    folium.Marker(location=[addr["lat"],addr["lon"]],
                  popup=addr["display_name"],tooltip="Selected address",
                  icon=folium.Icon(color="red",icon="home",prefix="fa")).add_to(fmap)
    if st.session_state.aoi is not None:
        folium.GeoJson(st.session_state.aoi.to_crs("EPSG:4326").geometry,
                       style_function=lambda x: {"fillColor":"#3388ff","color":"#3388ff","weight":2,"fillOpacity":0.2},
                       tooltip="Area of Interest").add_to(fmap)
    st_folium(fmap,width=700,height=300)
    if not st.session_state.gtfs_ready:
        if st.button("Search feeds", key="search_feeds_btn"):
            try:
                if st.session_state.aoi_download is None:
                    st.error("AOI not loaded. Select address first.")
                else:
                    with st.spinner("Searching and downloading GTFS feeds..."):
                        api = MobilityDatabaseClient(MOBILITY_DB_REFRESH_TOKEN)
                        feeds = api.search_gtfs_feeds(
                            country_code=st.session_state.geo.get('country_codes'),
                            subdivision_name=st.session_state.geo.get('subdivision_names'),
                            municipality=st.session_state.geo.get('municipalities'),
                            is_official=None
                        )
                        st.success(f"{len(feeds)} feed(s) found")
                        
                        # Store provider names in session state
                        st.session_state.gtfs_providers = sorted(list(set([f['provider'] for f in feeds])))

                        files = api.download_feeds(feeds, download_folder=os.path.join(WORK_DIR,"gtfs_files"), overwrite=False)
                        
                        start_dt = datetime.combine(date.today(),time())
                        end_dt = datetime.combine(date.today()+timedelta(days=30),time())

                        gtfs = Feed(files, aoi=st.session_state.aoi_download, stop_group_distance=100,
                                    start_date=start_dt, end_date=end_dt)
                        st.session_state.gtfs = gtfs
                        st.session_state.gtfs_ready = True
                        st.rerun() # Rerun to proceed to Step 2
            except Exception as e:
                st.error(f"Feed error: {e}")

# Always show provider list if available
if hasattr(st.session_state, "gtfs_providers") and st.session_state.gtfs_providers:
    st.subheader("Providers:")
    provider_df = pd.DataFrame(st.session_state.gtfs_providers, columns=['Provider Name'])
    st.dataframe(
        provider_df,
        hide_index=True,
        use_container_width=True,
        height=200  # show ~5 rows
    )

# === STEP 2: Service Quality & Buffers ===
if st.session_state.gtfs_ready:
    gtfs = st.session_state.gtfs
    
    st.header("1. Amount of transit service per day")
    with st.spinner("Calculating service intensity..."):
        s_int = gtfs.get_service_intensity_in_date_range(by_feed=True)
    try:
        plot_service_intensity(s_int)
    except Exception as e:
        st.error(f"Could not plot service intensity: {e}")

    st.header("2. Level of Service (A-F)")
    
    chosen_date = st.date_input("Date for Analysis", value=date.today())
    run_quality_clicked = st.button("Next", key="run_quality_btn")

    # Determine if we should show results (either clicked button or results already exist)
    show_step2_results = run_quality_clicked or st.session_state.service_quality_gdf is not None

    if show_step2_results:
        
        # --- Display Matrices (IMMEDIATE DISPLAY) ---
        col_svc, col_dist = st.columns(2)
        
        with col_svc:
            st.subheader("Service quality (Frequency vs Mode)")
            svc = processing_helper.SERVICE_MATRIX.copy()
            if 'tram' in svc.columns: svc = svc.rename(columns={'tram': 'tram/BRT'})
            if "interval" in svc.columns: svc = svc.rename(columns={"interval":"Frequency"})
            desired_order = ["Frequency", "rail", "tram/BRT", "bus"]
            existing_cols = [c for c in desired_order if c in svc.columns]
            svc = svc[existing_cols]
            numeric = [c for c in svc.columns if c != "Frequency"]
            svc["Frequency"] = svc["Frequency"].astype(str)+" min"
            for c in numeric: svc[c] = svc[c].apply(roman_color_blue)
            st.markdown(svc.to_html(index=False, escape=False), unsafe_allow_html=True)

        with col_dist:
            st.subheader("Level of service (Quality vs Distance)")
            svc_dist = processing_helper.DISTANCE_MATRIX.copy()
            svc_dist = svc_dist.rename(columns={"service_quality": "Service Quality"})
            styled_dist_matrix = color_distance_matrix(svc_dist)
            st.markdown(styled_dist_matrix.to_html(index=False, escape=False), unsafe_allow_html=True)
        # ------------------------------------------------------------------------

        
        if run_quality_clicked or st.session_state.level_of_service_gdf is None:
            # Only run processing if the button was clicked OR if data is missing/stale
            
            # 1. Run Calculation
            if run_quality_clicked:
                outdir = level_of_service_path
                os.makedirs(outdir, exist_ok=True)
                
                with st.spinner("Calculating stop service quality..."):
                    f = processing_helper.get_service_quality(outdir, gtfs,
                                dates=datetime.combine(chosen_date,time()), times=[START_HOUR,END_HOUR])
                    gdf = gpd.read_file(f)
                    st.session_state.service_quality_gdf = gdf
                    st.session_state.raster_ready = False 
                
                with st.spinner("Generating Level of Service Buffers..."):
                    level_of_service_gdf = isochrones.buffers(
                        gdf, level_of_services=LOS_GRADES,
                        distance_matrix=processing_helper.DISTANCE_MATRIX, 
                        service_quality_col=f"service_quality_{START_HOUR}h_{END_HOUR}h",
                    )
                st.session_state.level_of_service_gdf = level_of_service_gdf
                st.success("Quality analysis complete.")
                st.rerun() # Rerun to trigger population download and stable map display

        # --- Display Map and Trigger Background Download ---
        
        if st.session_state.level_of_service_gdf is not None:

            # Load GDFs from session state
            gdf = st.session_state.service_quality_gdf.copy()
            
            st.subheader("Level of service of the offered public transport")
            
            aoi = st.session_state.aoi.to_crs(gdf.crs)
            gdf_cropped = gpd.clip(gdf, aoi)
            st.session_state.stop_quality_gdf_cropped = gdf_cropped

            if gdf_cropped.crs is None: 
                gdf_cropped = gdf_cropped.set_crs("EPSG:4326")
            else: 
                gdf_cropped = gdf_cropped.to_crs("EPSG:4326")

            col_map_rename = {
                f"service_quality_{START_HOUR}h_{END_HOUR}h": "service_quality",
                f"interval_{START_HOUR}h_{END_HOUR}h": "frequency",
                f"route_names_{START_HOUR}h_{END_HOUR}h": "route_names",
                #f"route_type_{START_HOUR}h_{END_HOUR}h": "route_type"
            }
            gdf_cropped = gdf_cropped.rename(columns=col_map_rename)
            gdf_cropped["service_quality"] = gdf_cropped["service_quality"].fillna(0).astype(int)
            gdf_cropped["roman"] = gdf_cropped["service_quality"].apply(lambda x: to_roman(x) if x > 0 else "No service")

            # --- Helper function to clean stop info ---
            def clean_stop_value(val):
                """Convert Series/list/string to clean string for popup"""
                if isinstance(val, (pd.Series, np.ndarray)):
                    val = list(val)
                if isinstance(val, list):
                    cleaned = [re.sub(r'_file_\d+', '', str(x)) for x in val if x is not None and str(x) != 'nan']
                    return ", ".join(cleaned)
                if isinstance(val, str):
                    return re.sub(r'_file_\d+', '', val)
                return ""

            m = folium.Map(location=[addr["lat"], addr["lon"]], zoom_start=13, tiles="CartoDB Positron")
            
            # Add LOS Buffer Layer 
            los_gdf = st.session_state.level_of_service_gdf.to_crs(4326)
            aoi_4326 = st.session_state.aoi.to_crs(4326)
            los_gdf_cropped = gpd.clip(los_gdf, aoi_4326, keep_geom_type=True)
            
            folium.GeoJson(
                los_gdf_cropped.__geo_interface__,
                name="LOS Coverage Buffers",
                style_function=style_buffer_los_color,
                tooltip=folium.features.GeoJsonTooltip(
                    fields=["level_of_service"],
                    aliases=["Level of Service:"],
                    localize=True
                ),
                popup=folium.features.GeoJsonPopup(
                    fields=["level_of_service"],
                    aliases=["Level of Service:"],
                    localize=True
                )
            ).add_to(m)

            # Add AOI boundary
            folium.GeoJson(
                st.session_state.aoi.to_crs("EPSG:4326").geometry,
                style_function=lambda x: {"fillColor": "none", "color": "darkblue", "weight": 3, "dashArray": "5, 5"},
                tooltip="Area of Interest Boundary"
            ).add_to(m)

            # Add stops with clean popups
            for _, row in gdf_cropped.iterrows():
                route_names_clean = clean_stop_value(row.get('route_names', ''))
                route_type_clean = clean_stop_value(row.get('route_type', ''))

                popup_html = f"""
                <b>Stop:</b> {row.get('stop_name','')}<br>
                <b>Quality:</b> {row.get('service_quality','')}<br>
                <b>Frequency:</b> {row.get('frequency','')}<br>
                <b>Routes:</b> {route_names_clean}<br>
                <b>Type:</b> {route_type_clean}
                """

                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=6,
                    color="black",
                    weight=1,
                    fill=True,
                    fill_color=get_color(row["service_quality"]),
                    fill_opacity=1,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=popup_html
                ).add_to(m)
            
            # Add home marker
            folium.Marker([addr["lat"], addr["lon"]], icon=folium.Icon(color='red', icon='home')).add_to(m)

            st_folium(m, width=1200, height=600)

        
# === STEP 3: Accessibility Analysis ===
if st.session_state.level_of_service_gdf is not None:
    
    st.markdown("---")
    st.header("3. Population accessibility")

    population_file = st.session_state.population_file

    # if population_file is None:
    #     st.info("Population data download required for raster analysis is in progress or failed. Please check Step 2.")
        
    run_raster_clicked = st.button("Next", key="run_raster_btn")
    
    show_step3_results = run_raster_clicked or st.session_state.raster_ready

    if show_step3_results:
        
        # --- Display Density Matrix (IMMEDIATE DISPLAY) ---
        # --- Data ---
        los_data = {
            "Required Level of Service": ["A", "B", "C", "D–F"],
            "Population Density (pop/km²)": [3500, 2500, 1500, 300]
        }

        density_matrix_display_data = pd.DataFrame(los_data)

        # --- Styling Function ---
        def color_density_matrix(df):
            def color_by_service(service):
                colors = {
                    "A": "#2ecc71",   # green
                    "B": "#f1c40f",   # yellow
                    "C": "#e67e22",   # orange
                    "D–F": "#e74c3c"  # red
                }
                return f"background-color: {colors.get(service, 'white')}; color: black; text-align: center;"
            
            return df.style.applymap(color_by_service, subset=["Required Level of Service"]) \
                        .set_table_styles([
                            {"selector": "th",
                                "props": [("background-color", "#004c6d"),
                                        ("color", "white"),
                                        ("text-align", "center")]},
                            {"selector": "td",
                                "props": [("text-align", "center"),
                                        ("border", "1px solid #ccc"),
                                        ("padding", "6px")]}
                        ]) \
                        .hide(axis="index")

        # --- Streamlit Layout ---
        col_density, _ = st.columns([1, 2])
        with col_density:
            st.subheader("Level of service requirements")
            styled_density_matrix = color_density_matrix(density_matrix_display_data)
            st.markdown(styled_density_matrix.to_html(escape=False), unsafe_allow_html=True)
        # -----------------------------------------------------------------------------
        
        if run_raster_clicked and not st.session_state.raster_ready:
            # --- Run Calculation ---
             # MODIFICATION 2: START POPULATION DOWNLOAD AND HANDLE FEEDBACK OUTSIDE CACHE
            if st.session_state.population_file is None:
                st.subheader("Population Data Retrieval")
                with st.status("Checking cache or starting WorldPop 100m download...", expanded=True) as status:
                    pop_file, error = get_population_file_cached(st.session_state.aoi_pop_download, population_folder)
                    
                    if pop_file:
                        st.session_state.population_file = pop_file
                        status.update(label=f"✅ WorldPop data ready: {os.path.basename(pop_file)}", state="complete", expanded=False)
                    else:
                        status.update(label=f"❌ Failed to get WorldPop data.", state="error", expanded=True)
                        st.error(f"Population download failed: {error}")
                        st.session_state.population_file = None
        
            # END MODIFICATION 2
            population_file = st.session_state.population_file
            if population_file is None:
                st.warning("Cannot proceed: Population data is not yet available.")
            else:
                with st.spinner("Calculating accessibility rasters (Offer, Demand, Difference)..."):
                    population.level_of_service_raster(
                        save_path=level_of_service_path,
                        population=population_file,
                        offer=st.session_state.level_of_service_gdf,
                        density_matrix=DENSITY_MATRIX_CALC,
                        level_of_services=LOS_GRADES,
                        min_population=MIN_POP_DENSITY, # Use MIN_POP_DENSITY
                        level_of_service_column='level_of_service',
                        aoi=st.session_state.aoi
                    )
                    st.session_state.raster_ready = True
                    st.success("Rasters generated.")
                    st.rerun() # Rerun to display the raster map

        # --- Display Map (if raster generation is complete) ---
        if st.session_state.raster_ready:

            st.markdown("---")
            st.subheader("Accessibility Raster Visualization")

            aoi_geom = st.session_state.aoi
            m_raster = folium.Map(location=[addr["lat"], addr["lon"]], zoom_start=13, tiles="CartoDB Positron")

            los_colors_list = [get_grade_color_and_index(grade)[0] for grade in LOS_GRADES]

            # --- Raster files ---
            files_to_plot = {
                "population_density_0.tif": {"name": "Population Density", "show": False},
                "offer.tif": {"name": "Provided service", "show": True},
                "demand.tif": {"name": "Demanded service", "show": False},
                "difference.tif": {"name": "Transit Desert", "show": True},
            }

            pop_data_cache = None

            for filename, config in files_to_plot.items():
                filepath = os.path.join(level_of_service_path, filename)
                if not os.path.exists(filepath):
                    continue

                try:
                    data, transform, crs = raster_utils.read_raster(filepath, aoi=aoi_geom, projected=True)
                    if data.size == 0:
                        continue

                    data_float = data.astype(np.float64, copy=True)
                    data_masked = np.ma.masked_array(data_float, mask=~np.isfinite(data_float))

                    height, width = data.shape
                    left, bottom, right, top = rasterio.transform.array_bounds(height, width, transform)

                    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    minx, miny = transformer.transform(left, bottom)
                    maxx, maxy = transformer.transform(right, top)
                    wgs_bounds = [[miny, minx], [maxy, maxx]]

                    # --- Generate image overlay for each raster ---
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.set_axis_off()

                    if filename == "population_density_0.tif":
                        pop_data_cache = data_masked.copy()
                        cmap = plt.get_cmap("viridis")
                        data_to_plot = np.ma.masked_where(data_masked <= 0, data_masked)
                        ax.imshow(data_to_plot, cmap=cmap, origin='upper')

                    elif filename == "difference.tif":
                        if pop_data_cache is None:
                            pop_filepath = os.path.join(level_of_service_path, "population_density_0.tif")
                            pop_data_raw, _, _ = raster_utils.read_raster(pop_filepath, aoi=aoi_geom, projected=True)
                            pop_data_cache = np.ma.masked_array(pop_data_raw.astype(np.float64), mask=~np.isfinite(pop_data_raw))
                        pop_mask = (pop_data_cache.data >= MIN_POP_DENSITY) & (~pop_data_cache.mask)
                        desert_mask = (data_masked.data < 0) & (~data_masked.mask) & pop_mask
                        data_viz = np.zeros_like(data_masked)
                        data_viz[desert_mask] = 1
                        ax.imshow(np.ones_like(data_viz), cmap=colors.ListedColormap(['#00000000']), origin='upper')
                        ax.contourf(
                            np.arange(data_viz.shape[1]),
                            np.arange(data_viz.shape[0]),
                            desert_mask,
                            levels=[0.5, 1],
                            hatches=['////'],
                            colors='none',
                            edgecolor=TRANSIT_DESERT_COLOR,
                            linewidth=0.0
                        )

                    else:  # offer / demand
                        data_to_plot = np.ma.masked_where((data_masked < 0) | (data_masked > 11), data_masked)
                        cmap = colors.ListedColormap(los_colors_list)
                        ax.imshow(data_to_plot, cmap=cmap, origin='upper')

                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
                    plt.close(fig)
                    buf.seek(0)
                    image_data = buf.read()

                    folium.raster_layers.ImageOverlay(
                        image=f"data:image/png;base64,{base64.b64encode(image_data).decode()}",
                        bounds=wgs_bounds,
                        name=config["name"],
                        opacity=0.8,
                        overlay=True,
                        control=True,
                        show=config["show"]
                    ).add_to(m_raster)

                except Exception as e:
                    st.warning(f"Error plotting raster {filename}: {e}")

            # --- Stops ---
            stops_group = folium.FeatureGroup(name="Service Quality Stops", show=False).add_to(m_raster)
            stops_to_plot = st.session_state.stop_quality_gdf_cropped.copy()

            # Rename columns
            col_map_rename = {
                f"service_quality_{START_HOUR}h_{END_HOUR}h":"service_quality",
                f"interval_{START_HOUR}h_{END_HOUR}h":"frequency",
                f"route_names_{START_HOUR}h_{END_HOUR}h":"route_names",
                #f"route_type_{START_HOUR}h_{END_HOUR}h":"route_type"
            }
            stops_to_plot = stops_to_plot.rename(columns=col_map_rename)
            stops_to_plot["service_quality"] = stops_to_plot["service_quality"].fillna(0).astype(int)

            # --- Clean stop columns ---
            def clean_stop_value(val):
                """Convert Series/list/string to clean string for popup"""
                if isinstance(val, (pd.Series, np.ndarray)):
                    val = list(val)
                if isinstance(val, list):
                    cleaned = [re.sub(r'_file_\d+', '', str(x)) for x in val if x is not None and str(x) != 'nan']
                    return ", ".join(cleaned)
                if isinstance(val, str):
                    return re.sub(r'_file_\d+', '', val)
                return ""

            stops_to_plot['route_type'] = stops_to_plot['route_type'].apply(clean_stop_value)
            stops_to_plot['route_names'] = stops_to_plot['route_names'].apply(clean_stop_value)

            for _, row in stops_to_plot.iterrows():
                popup_html = f"""
                <b>Stop:</b> {row.get('stop_name','')}<br>
                <b>Quality:</b> {row.get('service_quality','')}<br>
                <b>Frequency:</b> {row.get('frequency','')}<br>
                <b>Routes:</b> {row['route_names']}<br>
                <b>Type:</b> {row['route_type']}
                """
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5, color="black", weight=1, fill=True,
                    fill_color=get_color(row["service_quality"]), fill_opacity=1,
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(stops_group)

            # --- Legends using branca ---
            def add_legend(title, colors, labels):
                html = f"<div style='background:white; padding:10px; border:2px solid grey; font-size:12px; color:black;'><b>{title}</b><br>"
                for c, l in zip(colors, labels):
                    html += f"<i style='background:{c}; padding:2px 5px; margin:1px; display:inline-block;'></i> {l}<br>"
                html += "</div>"
                legend = branca.element.MacroElement()
                legend._template = branca.element.Template(html)
                m_raster.get_root().add_child(legend)

            # LOS legend
            add_legend("LOS Index", los_colors_list, LOS_GRADES)

            # Transit Desert legend
            add_legend("Transit Desert", [TRANSIT_DESERT_COLOR], ["Transit Desert (hatched)"])

            # --- Layer Control ---
            folium.LayerControl(collapsed=False).add_to(m_raster)

            st_folium(m_raster, width=1200, height=600)

            # --- Population per Service Category Table ---
            st.markdown("---")
            st.subheader("📊 Accessibility Summary")

            try:
                # --- File paths ---
                pop_filepath = os.path.join(level_of_service_path, "population.tif")
                diff_filepath = os.path.join(level_of_service_path, "difference.tif")
                offer_filepath = os.path.join(level_of_service_path, "offer.tif")
                aoi_geom = st.session_state.aoi

                # --- Read rasters ---
                pop_data, _, _ = raster_utils.read_raster(pop_filepath, aoi=aoi_geom, projected=False)
                diff_data, _, _ = raster_utils.read_raster(diff_filepath, aoi=aoi_geom, projected=False)
                offer_data, _, _ = raster_utils.read_raster(offer_filepath, aoi=aoi_geom, projected=False)

                # --- Mask invalid / nan values ---
                pop_data = np.where(np.isfinite(pop_data), pop_data, np.nan)
                diff_data = np.where(np.isfinite(diff_data), diff_data, np.nan)
                offer_data = np.where(np.isfinite(offer_data), offer_data, np.nan)

                # --- Compute population with insufficient coverage ---
                pop_total = np.nansum(pop_data)
                pop_insufficient = np.nansum(pop_data[(diff_data < 0) & (pop_data > 0)])
                pct_insufficient = (pop_insufficient / pop_total * 100) if pop_total > 0 else 0

                st.markdown(
                    f"### Population with insufficient coverage: "
                    f"<span style='color:#e74c3c; font-weight:bold;'>{pct_insufficient:.2f}%</span>",
                    unsafe_allow_html=True
                )

                # --- LOS mapping ---
                los_mapping = {
                    0: "A1", 1: "A2", 2: "A3",
                    3: "B1", 4: "B2", 5: "B3",
                    6: "C1", 7: "C2", 8: "C3",
                    9: "D", 10: "E", 11: "F"
                }

                # --- Aggregate population per LOS ---
                valid_mask = pop_data > 0
                offer_int = np.where(np.isnan(offer_data), 12, offer_data).astype(int)

                pop_by_offer = {}
                for val, label in los_mapping.items():
                    pop_by_offer[label] = np.nansum(pop_data[(offer_int == val) & valid_mask])

                # “No Service”: population with NaN offer values
                pop_by_offer["No Service"] = np.nansum(pop_data[np.isnan(offer_data) & valid_mask])

                # --- Create DataFrame with percentage of total population ---
                pop_offer_df = pd.DataFrame({
                    "Level of Service": list(pop_by_offer.keys()),
                    "Population": list(pop_by_offer.values())
                })
                pop_offer_df["Population"] = pop_offer_df["Population"].round(0).astype(int)
                pop_offer_df["% of Total"] = (pop_offer_df["Population"] / pop_total * 100).round(2)

                # --- Display table without index ---
                st.markdown("### Population per Service Category")
                st.dataframe(
                    pop_offer_df.style.format({
                        "Population": "{:,}",
                        "% of Total": "{:.2f}%"
                    }).hide(axis="index"),
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Error processing rasters: {e}")
            except Exception as e:
                st.error(f"Failed to calculate population coverage summary: {e}")