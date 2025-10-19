import sys
sys.path.append('/home/miguel/Documents/Proyectos/PTLevelofService/gtfs/pyGTFSHandler')
sys.path.append('/home/miguel/Documents/Proyectos/PTLevelofService/accessibility/UrbanAccessAnalyzer')

import os
from datetime import datetime, date, timedelta, time
import streamlit as st
import requests, folium
from streamlit_folium import st_folium
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyGTFSHandler.feed import Feed
from pyGTFSHandler.downloaders.mobility_database import (
    MobilityDatabaseClient, get_geographic_suggestions_from_string, get_city_geometry
)
import pyGTFSHandler.plot_helper as plot_helper
import pyGTFSHandler.processing_helper as processing_helper
import UrbanAccessAnalyzer.utils as utils


# === CONFIG ===
st.set_page_config(page_title="Public Transport Level Of Service", layout="wide")

START_HOUR, END_HOUR = 8, 20
BUFFER_METERS = 3500
WORK_DIR = "/home/miguel/Documents/Proyectos/PTLevelofService/accessibility/UrbanAccessAnalyzer/no_sync"
USER_AGENT = "urban-access-analyzer/1.0"
MOBILITY_DB_REFRESH_TOKEN = 'AMf-vByYiwMAni1pw6yTpwgwwYFc8HR4y0zUKZGPT4sjJ0wUrIXOfVxF1KotRIvEgAseaaNheL8YczJiCILb6o2PUh-8zjA-qQURzEc8tELlwFiDopMoqJnkDf13AqNaGGnnzTDmYM20AWEquUxcYFAB8Q3e5rI2DcTBSQuiUdHL8bi48xmUJk3tayHpnoicoppi_evDcWYODwOJFcwnta3K7f718w7R2JRM0zDEOYw7nI7thrQa9462BENdpv8zv8mEbBssEa189k6YcV__sQAZlng2EcsCGA'

# === SESSION STATE ===
defaults = {
    "confirmed_address": None, "last_input": "", "gtfs_ready": False,
    "service_quality_gdf": None, "gtfs": None,
    "aoi": None, "aoi_download": None, "geo": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# === HELPERS ===
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
    for v,s in vals:
        while n>=v: out+=s; n-=v
    return out

def sanitize_filename(s): return utils.sanitize_filename(s)

def plot_service_intensity(service_intensity):
    if not isinstance(service_intensity, pd.DataFrame):
        pdf = service_intensity.to_pandas()
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

# === SIDEBAR: Address search ===
with st.sidebar:
    st.header("📍 Address")
    addr_input = st.text_input("Search:", value=st.session_state.last_input)
    if addr_input != st.session_state.last_input:
        st.session_state.last_input = addr_input
        st.session_state.confirmed_address = None

# === CONFIRM ADDRESS AND AOI ===
if addr_input and not st.session_state.confirmed_address:
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

# === AFTER ADDRESS IS CONFIRMED ===
if st.session_state.confirmed_address:
    addr = st.session_state.confirmed_address
    st.subheader(addr["display_name"])

    # First map: address + AOI
    fmap = folium.Map(location=[addr["lat"],addr["lon"]],zoom_start=13)
    folium.Marker(location=[addr["lat"],addr["lon"]],
                  popup=addr["display_name"],tooltip="Selected address",
                  icon=folium.Icon(color="red",icon="home",prefix="fa")).add_to(fmap)
    if st.session_state.aoi is not None:
        folium.GeoJson(st.session_state.aoi.to_crs("EPSG:4326").geometry,
                       style_function=lambda x: {"fillColor":"#3388ff","color":"#3388ff","weight":2,"fillOpacity":0.2},
                       tooltip="Area of Interest").add_to(fmap)
    st_folium(fmap,width=700,height=300)

    # --- Search feeds button ---
    if st.button("Search feeds"):
        try:
            if st.session_state.aoi_download is None:
                st.error("AOI not loaded. Select address first.")
            else:
                api = MobilityDatabaseClient(MOBILITY_DB_REFRESH_TOKEN)
                feeds = api.search_gtfs_feeds(
                    country_code=st.session_state.geo.get('country_codes'),
                    subdivision_name=st.session_state.geo.get('subdivision_names'),
                    municipality=st.session_state.geo.get('municipalities'),
                    is_official=None
                )
                st.success(f"{len(feeds)} feed(s) found")
                st.write("Providers:", [f['provider'] for f in feeds])
                files = api.download_feeds(feeds, download_folder=os.path.join(WORK_DIR,"gtfs_files"), overwrite=False)
                gtfs = Feed(files, aoi=st.session_state.aoi_download, stop_group_distance=100,
                            start_date=datetime.combine(date.today(),time()),
                            end_date=datetime.combine(date.today()+timedelta(days=30),time()))
                st.session_state.gtfs = gtfs
                st.session_state.gtfs_ready = True
        except Exception as e:
            st.error(f"Feed error: {e}")

# === MAIN ANALYSIS ===
if st.session_state.gtfs_ready:
    gtfs = st.session_state.gtfs
    with st.spinner("Calculating service intensity..."):
        s_int = gtfs.get_service_intensity_in_date_range(by_feed=True)
    try:
        plot_service_intensity(s_int)
    except Exception as e:
        st.error(f"Could not plot service intensity: {e}")

    # Prepare table and second map
    chosen_date = st.date_input("Date", value=date.today())
    run = st.button("Analyze")
    if run or st.session_state.service_quality_gdf is not None:
        if run:
            outdir = os.path.join(WORK_DIR, sanitize_filename(addr["display_name"]), "level_of_service")
            os.makedirs(outdir, exist_ok=True)
            f = processing_helper.get_service_quality(outdir, gtfs,
                        dates=datetime.combine(chosen_date,time()), times=[START_HOUR,END_HOUR])
            st.session_state.service_quality_gdf = gpd.read_file(f)
        gdf = st.session_state.service_quality_gdf.copy()

        # --- Prepare table ---
        svc = processing_helper.SERVICE_MATRIX.copy()
        if "interval" in svc.columns: svc = svc.rename(columns={"interval":"Frequency"})
        numeric = [c for c in svc.columns if c != "Frequency"]
        cmap = matplotlib.colormaps["Blues_r"]
        def roman_color(val):
            try:
                v = float(val)
                norm = (v-1)/(12-1)
                rgba = cmap(norm)
                rgb = f"rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})"
                return f"<span style='color:{rgb}; font-weight:bold; font-size:1.1em'>{to_roman(int(v))}</span>"
            except: return val
        svc["Frequency"] = svc["Frequency"].astype(str)+" min"
        for c in numeric: svc[c] = svc[c].apply(roman_color)
        st.markdown("""
            <style>
            .dataframe td, .dataframe th { font-size: 1.1em !important; white-space: nowrap; padding: 4px 8px !important; }
            div[data-testid="stDataFrame"] { height:auto !important; }
            </style>
        """, unsafe_allow_html=True)
        left,right = st.columns([0.8,1.2])
        with left:
            st.subheader("Service levels")
            st.markdown(svc.to_html(index=False, escape=False), unsafe_allow_html=True)

        # --- Second map ---
        with right:
            st.subheader("Coverage")
            if gdf.crs is None: gdf = gdf.set_crs("EPSG:4326")
            else: gdf = gdf.to_crs("EPSG:4326")

            # Columns mapping
            start_int, end_int = START_HOUR, END_HOUR
            col_map = {
                f"service_quality_{start_int}h_{end_int}h":"service_quality",
                f"interval_{start_int}h_{end_int}h":"frequency",
                f"route_names_{start_int}h_{end_int}h":"route_names",
                f"route_type_{start_int}h_{end_int}h":"route_type"
            }
            gdf = gdf.rename(columns=col_map)

            gdf["service_quality"] = gdf["service_quality"].fillna(0).astype(int)
            gdf["roman"] = gdf["service_quality"].apply(lambda x: to_roman(x) if x>0 else "No service")

            def get_color(val):
                if val==0: return "#000000"
                norm = (val-1)/(12-1)
                rgba = cmap(norm)
                return matplotlib.colors.rgb2hex(rgba)

            m = folium.Map(location=[addr["lat"],addr["lon"]],zoom_start=13)
            for _,row in gdf.iterrows():
                popup_html = f"""
                <b>Stop:</b> {row.get('stop_name','')}<br>
                <b>Parent:</b> {row.get('parent_station','')}<br>
                <b>Frequency:</b> {row.get('frequency','')}<br>
                <b>Service:</b> {row.get('roman','')}<br>
                <b>Route names:</b> {row.get('route_names','')}<br>
                <b>Route type:</b> {row.get('route_type','')}
                """
                folium.CircleMarker(location=[row.geometry.y,row.geometry.x],
                                    radius=6,
                                    color="black",weight=1,
                                    fill=True,fill_color=get_color(row["service_quality"]),
                                    fill_opacity=1,
                                    popup=popup_html,
                                    tooltip=popup_html).add_to(m)
            folium.Marker([addr["lat"],addr["lon"]],icon=folium.Icon(color='red',icon='home')).add_to(m)

            # Legend
            legend_html = """
            <div style="position: fixed; bottom: 50px; left: 50px; width: 160px; height: 250px; 
                        border:2px solid grey; z-index:9999; font-size:14px; background:white; padding:10px;">
            <b>Quality</b><br>
            <i style='color:#000000; font-weight:bold'> No service </i><br>
            """
            for i in range(1,13):
                hexc = matplotlib.colors.rgb2hex(cmap((i-1)/(12-1)))
                legend_html += f"<i style='color:{hexc}; font-weight:bold'> {to_roman(i)} </i><br>"
            legend_html += "</div>"
            m.get_root().html.add_child(folium.Element(legend_html))

            st_folium(m,width=1200,height=600)
