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
BUFFER_METERS = 3500
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
    'A': ['#ccffcc', '#66ff66', '#00cc00'], # Light to Dark Green
    'B': ['#ffffcc', '#ffff66', '#cccc00'], # Light to Dark Yellow
    'C': ['#ffcccc', '#ff6666', '#cc0000'], # Light to Dark Red
    'D': ['#ffb3e4'], # Light Pink
    'E': ['#d9b3ff'], # Light Purple
    'F': ['#b3d9ff'], # Light Blue
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
    "aoi": None, "aoi_download": None, "geo": None, "raster_ready": False,
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
    if not isinstance(service_intensity, pd.DataFrame): pdf = service_intensity.to_pandas()
    else: pdf = service_intensity.copy()
    pdf["date"] = pd.to_datetime(pdf["date"])
    plt.figure(figsize=(12,6))
    
    bar_width=0.8
    weekend_hatch = "xx"
    holiday_hatch = "//"

    if "file_id" in pdf.columns:
        unique_files = sorted(pdf["file_id"].unique())
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
        file_handles = [plt.Line2D([],[],color=color_map[fid],label=f"File {fid}",linewidth=10) for fid in unique_files]
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
                        st.session_state.aoi_download = aoi.to_crs(aoi.estimate_utm_crs()).buffer(BUFFER_METERS)
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

    # Search feeds button logic
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
                        
                        # MODIFICATION 1: Nicer formatted, scrollable providers list
                        provider_names = sorted(list(set([f['provider'] for f in feeds])))
                        provider_df = pd.DataFrame(provider_names, columns=['Provider Name'])
                        
                        st.subheader("Providers:")
                        st.dataframe(
                            provider_df,
                            hide_index=True,
                            use_container_width=True,
                            height=200 # Show approximately 5 rows
                        )
                        # END MODIFICATION 1

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

# === STEP 2: Service Quality & Buffers ===
if st.session_state.gtfs_ready:
    gtfs = st.session_state.gtfs
    
    st.header("1. Service Intensity Overview")
    with st.spinner("Calculating service intensity..."):
        s_int = gtfs.get_service_intensity_in_date_range(by_feed=True)
    try:
        plot_service_intensity(s_int)
    except Exception as e:
        st.error(f"Could not plot service intensity: {e}")

    st.header("2. Detailed Service Quality & Coverage")
    
    chosen_date = st.date_input("Date for Analysis", value=date.today())
    run_quality_clicked = st.button("Analyze Quality & Buffers", key="run_quality_btn")

    # Determine if we should show results (either clicked button or results already exist)
    show_step2_results = run_quality_clicked or st.session_state.service_quality_gdf is not None

    if show_step2_results:
        
        # --- Display Matrices (IMMEDIATE DISPLAY) ---
        col_svc, col_dist = st.columns(2)
        
        with col_svc:
            st.subheader("Service Levels (Frequency vs Time)")
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
            st.subheader("Distance Matrix (Quality vs Distance)")
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
            
            # MODIFICATION 2: START POPULATION DOWNLOAD AND HANDLE FEEDBACK OUTSIDE CACHE
            if st.session_state.population_file is None:
                st.subheader("Population Data Retrieval")
                with st.status("Checking cache or starting WorldPop 100m download...", expanded=True) as status:
                    pop_file, error = get_population_file_cached(st.session_state.aoi_download, population_folder)
                    
                    if pop_file:
                        st.session_state.population_file = pop_file
                        status.update(label=f"✅ WorldPop data ready: {os.path.basename(pop_file)}", state="complete", expanded=False)
                    else:
                        status.update(label=f"❌ Failed to get WorldPop data.", state="error", expanded=True)
                        st.error(f"Population download failed: {error}")
                        st.session_state.population_file = None
            # END MODIFICATION 2

            # Load GDFs from session state
            gdf = st.session_state.service_quality_gdf.copy()
            
            st.subheader("Coverage Map (Stop Service Quality and Buffer Coverage)")
            
            aoi = st.session_state.aoi.to_crs(gdf.crs)
            gdf_cropped = gpd.clip(gdf, aoi)
            st.session_state.stop_quality_gdf_cropped = gdf_cropped

            if gdf_cropped.crs is None: gdf_cropped = gdf_cropped.set_crs("EPSG:4326")
            else: gdf_cropped = gdf_cropped.to_crs("EPSG:4326")

            col_map_rename = {
                f"service_quality_{START_HOUR}h_{END_HOUR}h":"service_quality",
                f"interval_{START_HOUR}h_{END_HOUR}h":"frequency",
                f"route_names_{START_HOUR}h_{END_HOUR}h":"route_names",
                f"route_type_{START_HOUR}h_{END_HOUR}h":"route_type"
            }
            gdf_cropped = gdf_cropped.rename(columns=col_map_rename)
            gdf_cropped["service_quality"] = gdf_cropped["service_quality"].fillna(0).astype(int)
            gdf_cropped["roman"] = gdf_cropped["service_quality"].apply(lambda x: to_roman(x) if x>0 else "No service")

            m = folium.Map(location=[addr["lat"],addr["lon"]],zoom_start=13, tiles="CartoDB Positron")
            
            # Add LOS Buffer Layer 
            los_gdf = st.session_state.level_of_service_gdf.to_crs(4326)
            aoi_4326 = st.session_state.aoi.to_crs(4326)
            los_gdf_cropped = gpd.clip(los_gdf, aoi_4326, keep_geom_type=True)
            
            folium.GeoJson(
                los_gdf_cropped.__geo_interface__,
                name="LOS Coverage Buffers",
                style_function=style_buffer_los_color,
                tooltip=False, popup=False
            ).add_to(m)

            # Add AOI boundary, stops, and legend
            folium.GeoJson(st.session_state.aoi.to_crs("EPSG:4326").geometry,
                           style_function=lambda x: {"fillColor": "none", "color": "darkblue", "weight": 3,"dashArray": "5, 5"},
                           tooltip="Area of Interest Boundary").add_to(m)

            for _,row in gdf_cropped.iterrows():
                popup_html = f"<b>Stop:</b> {row.get('stop_name','')}<br>..." # Simplified
                folium.CircleMarker(location=[row.geometry.y,row.geometry.x],
                                    radius=6, color="black",weight=1, fill=True,
                                    fill_color=get_color(row["service_quality"]),
                                    fill_opacity=1, popup=popup_html, tooltip=popup_html).add_to(m)
            
            folium.Marker([addr["lat"],addr["lon"]],icon=folium.Icon(color='red',icon='home')).add_to(m)

            legend_html = """
            <div style="position: fixed; bottom: 50px; left: 50px; width: 160px; height: 250px; 
                        border:2px solid grey; z-index:9999; font-size:14px; background:white; padding:10px;">
            <b>Quality</b><br>
            <i style='color:#000000; font-weight:bold'> No service </i><br>
            """
            for i in range(1,13):
                hexc = matplotlib.colors.rgb2hex(cmap_blue((i-1)/(12-1)))
                legend_html += f"<i style='color:{hexc}; font-weight:bold'> {to_roman(i)} </i><br>"
            legend_html += "</div>"
            m.get_root().html.add_child(folium.Element(legend_html))

            st_folium(m,width=1200,height=600)

        
# === STEP 3: Accessibility Analysis ===
if st.session_state.level_of_service_gdf is not None:
    
    st.markdown("---")
    st.header("3. Accessibility Analysis: Demand & Offer")

    # MODIFICATION 3: Use the file path from session state, populated in Step 2.
    population_file = st.session_state.population_file

    if population_file is None:
        st.info("Population data download required for raster analysis is in progress or failed. Please check Step 2.")
        
    run_raster_clicked = st.button("Generate Demand & Offer Rasters", key="run_raster_btn")
    
    show_step3_results = run_raster_clicked or st.session_state.raster_ready

    if show_step3_results:
        
        # --- Display Density Matrix (IMMEDIATE DISPLAY) ---
        col_density, _ = st.columns([1, 2])
        with col_density:
            st.subheader("Population Demand Matrix")
            styled_density_matrix = color_density_matrix(density_matrix_display_data)
            st.markdown(styled_density_matrix.to_html(index=False, escape=False), unsafe_allow_html=True)
        # -----------------------------------------------------------------------------
        
        if run_raster_clicked and not st.session_state.raster_ready:
            # --- Run Calculation ---
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
            m_raster = folium.Map(location=[addr["lat"],addr["lon"]],zoom_start=13, tiles="CartoDB Positron")
            
            los_colors_list = [get_grade_color_and_index(grade)[0] for grade in LOS_GRADES]
            # Ensure transparency for unclassified (implicit) areas by using transparent for 'transparent' codes
            los_cmap_array = [c if c != 'transparent' else '#00000000' for c in los_colors_list] 
            los_cmap = colors.ListedColormap(los_cmap_array)
            
            files_to_plot = {
                "population_density_0.tif": {"name": "1. Population Density (Demand Base)", "cmap": "PopViridis", "range": [1, 15000], "show": True},
                "offer.tif": {"name": "2. Service Offer (LOS Index)", "cmap": los_cmap, "range": [-0.5, 11.5], "show": True},
                "demand.tif": {"name": "3. Service Demand (LOS Index)", "cmap": los_cmap, "range": [-0.5, 11.5], "show": False},
                "difference.tif": {"name": "4. Transit Desert (Offer < Demand)", "cmap": "RdYlGn", "range": [-12, 12], "show": False},
            }
            
            pop_data_cache = None # Cache for density data

            for filename, config in files_to_plot.items():
                filepath = os.path.join(level_of_service_path, filename)
                if not os.path.exists(filepath): continue
                
                try:
                    # Read raster data
                    data, transform, crs = raster_utils.read_raster(filepath, aoi=aoi_geom, projected=True) 

                    if data.size == 0: continue
                    
                    # === CRITICAL FIX: Aggressive float conversion and NaN isolation ===
                    
                    # 1. Force array to float64, copying the data.
                    data_float = data.astype(np.float64, copy=True)
                    
                    # 2. Explicitly mask all non-finite values (NaN, Inf)
                    non_finite_mask = ~np.isfinite(data_float)
                    data_masked = np.ma.masked_array(data_float, mask=non_finite_mask)
                    
                    # =================================================================
                    
                    height, width = data.shape
                    left, bottom, right, top = rasterio.transform.array_bounds(height, width, transform)
                    
                    cmap = config["cmap"]
                    vmin, vmax = config["range"]
                    interpolation = 'nearest'
                    data_to_plot = data_masked.copy()
                    
                    
                    if filename == "population_density_0.tif":
                        # Cache population data
                        pop_data_cache = data_to_plot.copy() 
                        
                        # Custom PopViridis: transparent below MIN_POP_DENSITY
                        cmap_colors = [(0, '#00000000')] # Start fully transparent
                        norm_min_pop = MIN_POP_DENSITY / 15000.0
                        
                        cmap_colors.append((norm_min_pop * 0.9999, '#00000000')) 
                        
                        cmap_colors.append((norm_min_pop, colors.rgb2hex(plt.get_cmap('viridis')(0.1))))
                        cmap_colors.append((1, colors.rgb2hex(plt.get_cmap('viridis')(0.95))))
                        
                        cmap = colors.LinearSegmentedColormap.from_list("PopViridisCustom", cmap_colors)
                        
                        # Mask out values below or equal to 0 
                        data_to_plot = np.ma.masked_where(data_masked <= 0, data_masked)
                        vmin = 1
                        vmax = 15000
                        
                    elif filename == "difference.tif":
                        # Defensive Population Reading
                        if pop_data_cache is None:
                            pop_filepath = os.path.join(level_of_service_path, "population_density_0.tif")
                            pop_data_raw, _, _ = raster_utils.read_raster(pop_filepath, aoi=aoi_geom, projected=True)
                            pop_data_float = pop_data_raw.astype(np.float64, copy=True) 
                            # Re-apply robust masking to the cache if needed
                            pop_data_cache = np.ma.masked_array(pop_data_float, mask=~np.isfinite(pop_data_float))
                        
                        if pop_data_cache.shape != data_to_plot.shape:
                            st.error("Population raster shape mismatch for difference calculation.")
                            continue
                        
                        # Extract underlying numpy arrays for mask computation
                        pop_data_unmasked = pop_data_cache.data
                        diff_data_unmasked = data_to_plot.data
                        
                        # Inherited masks (to ensure we don't calculate on masked data)
                        diff_mask_inherited = data_to_plot.mask
                        pop_mask_inherited = pop_data_cache.mask

                        # Check 1: Pop density must be sufficient (and not masked)
                        pop_sufficient = (pop_data_unmasked >= MIN_POP_DENSITY) & (~pop_mask_inherited)
                        
                        # Check 2: Difference must be negative (Offer < Demand) (and not masked)
                        diff_negative = (diff_data_unmasked < 0) & (~diff_mask_inherited)
                        
                        # Combine: Mask where we DO NOT want to plot
                        plot_mask = ~(pop_sufficient & diff_negative) 
                        
                        # Create visualization array (1 where transit desert, NaN otherwise)
                        data_viz = np.full(data_to_plot.shape, np.nan, dtype=np.float64)
                        data_viz[~plot_mask] = 1 
                        
                        # Mask NaNs (which are the transparent areas)
                        data_to_plot = np.ma.masked_invalid(data_viz)
                        
                        # Custom Colormap: [Transparent, Pink]
                        cmap = colors.ListedColormap(['#00000000', TRANSIT_DESERT_COLOR])
                        vmin = 0
                        vmax = 1
                        
                    elif filename in ["offer.tif", "demand.tif"]:
                        # LOS Rasters 
                        data_to_plot = np.ma.masked_where((data_masked < 0) | (data_masked > 11), data_masked)
                        vmin = -0.5
                        vmax = 11.5
                        
                    
                    # --- Plotting ---
                    fig, ax = plt.subplots(figsize=(10, 10))
                    im = ax.imshow(data_to_plot, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', interpolation=interpolation)

                    ax.set_axis_off()
                    buf = io.BytesIO()
                    # We save without borders/padding and with a transparent background
                    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
                    plt.close(fig)
                    buf.seek(0)
                    
                    # ... (Folium ImageOverlay definition)
                    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    minx, miny = transformer.transform(left, bottom)
                    maxx, maxy = transformer.transform(right, top)
                    wgs_bounds = [[miny, minx], [maxy, maxx]]

                    folium.raster_layers.ImageOverlay(
                        image=f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}",
                        bounds=wgs_bounds,
                        name=config["name"],
                        opacity=0.8,
                        overlay=True, 
                        control=True,
                        show=config["show"]
                    ).add_to(m_raster)
                
                except Exception as e:
                    st.warning(f"Error plotting raster {filename}: Plotting failure encountered during float/masking conversion: {e}")
                    # Retain the original error type if it still appears
                    if "ufunc 'isnan' not supported" in str(e):
                         st.error(f"FATAL TYPE ERROR: ufunc 'isnan' failed on file {filename}. Check source file dtype.")
                    
            stops_group = folium.FeatureGroup(name="6. Service Quality Stops (Points)", show=False).add_to(m_raster)
            
            stops_to_plot = st.session_state.stop_quality_gdf_cropped.copy()
            
            col_map_rename = {
                f"service_quality_{START_HOUR}h_{END_HOUR}h":"service_quality",
                f"interval_{START_HOUR}h_{END_HOUR}h":"frequency",
                f"route_names_{START_HOUR}h_{END_HOUR}h":"route_names",
                f"route_type_{START_HOUR}h_{END_HOUR}h":"route_type"
            }
            stops_to_plot = stops_to_plot.rename(columns=col_map_rename)
            stops_to_plot["service_quality"] = stops_to_plot["service_quality"].fillna(0).astype(int)
            
            # The loop now uses the correctly renamed DataFrame: stops_to_plot
            for _,row in stops_to_plot.iterrows(): 
                popup_html = f"<b>Stop:</b> {row.get('stop_name','')}<br>..." # Simplified
                folium.CircleMarker(location=[row.geometry.y,row.geometry.x],
                                    radius=5, color="black",weight=1, fill=True,
                                    fill_color=get_color(row["service_quality"]), fill_opacity=1,
                                    popup=popup_html, tooltip=popup_html).add_to(stops_group)
                
            # --- Custom Legends for Rasters ---
            
            # LOS Index Legend (Applicable to Offer and Demand)
            los_legend_html = """
            <div style="position: fixed; bottom: 50px; right: 10px; width: 180px; height: 300px; 
                        border:2px solid grey; z-index:9999; font-size:12px; background:white; padding:5px;">
            <b>LOS Index (Offer/Demand)</b><br>
            """
            for i, grade in enumerate(LOS_GRADES):
                color, _, _ = get_grade_color_and_index(grade)
                los_legend_html += f'<i style="background:{color}; border: 1px solid black; padding: 2px 5px; margin: 1px; display: inline-block;"></i> {grade}<br>'
            
            los_legend_html += "</div>"
            
            # Transit Desert (Difference) Legend
            diff_legend_html = f"""
            <div style="position: fixed; bottom: 50px; right: 200px; width: 180px; height: 80px; 
                        border:2px solid grey; z-index:9999; font-size:12px; background:white; padding:5px;">
            <b>Transit Desert (Unmet Need)</b><br>
            <i style="background:transparent; border: 1px dashed black; padding: 2px 5px; margin: 1px; display: inline-block;"></i> Offer $\\ge$ Demand (or Low Pop)<br>
            <i style="background:{TRANSIT_DESERT_COLOR}; border: 1px solid black; padding: 2px 5px; margin: 1px; display: inline-block;"></i> Transit Desert (Unmet Need - Pink Area)<br>
            (Only displayed where Pop $\\ge$ {MIN_POP_DENSITY} P/km²)
            </div>
            """

            # Stop Quality Legend (Blue Roman)
            legend_html_stop_quality = """
            <div style="position: fixed; bottom: 50px; left: 50px; width: 160px; height: 250px; 
                        border:2px solid grey; z-index:9999; font-size:14px; background:white; padding:10px;">
            <b>Stop Quality Index</b><br>
            <i style='color:#000000; font-weight:bold'> No service </i><br>
            """
            for i in range(1,13):
                hexc = matplotlib.colors.rgb2hex(cmap_blue((i-1)/(12-1)))
                legend_html_stop_quality += f"<i style='color:{hexc}; font-weight:bold'> {to_roman(i)} </i><br>"
            legend_html_stop_quality += "</div>"
            
            
            m_raster.get_root().html.add_child(folium.Element(los_legend_html))
            m_raster.get_root().html.add_child(folium.Element(diff_legend_html))
            m_raster.get_root().html.add_child(folium.Element(legend_html_stop_quality))

            # Add Layer Control
            folium.LayerControl(collapsed=False).add_to(m_raster)
            
            st_folium(m_raster, width=1200, height=600)