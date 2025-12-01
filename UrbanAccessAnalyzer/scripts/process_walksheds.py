import os 
import json
import argparse

import geopandas as gpd 
import pyogrio
import shapely 
from shapely import wkt
import osmnx as ox 

import UrbanAccessAnalyzer.isochrones as isochrones
import UrbanAccessAnalyzer.graph_processing as graph_processing
import UrbanAccessAnalyzer.osm as osm
import UrbanAccessAnalyzer.h3_utils as h3_utils
import UrbanAccessAnalyzer.poi_utils as poi_utils

parser = argparse.ArgumentParser(description="Accesibility analysis.")

parser.add_argument('--poi_file', type=str, required=True, help='Path to file containing points or polygons of interest')
parser.add_argument('--poi_quality_column', type=str, default=None, help='Column or columns in poi_file dataframe with info about poi quality. Pois are grouped based on this column.')
parser.add_argument('--output_path', type=str, required=True, help='Path to save outputs.')
parser.add_argument('--street_path', type=str, default=None, help='Path to download and search for street files.')
parser.add_argument('--population_path', type=str, default=None, help='Path to download and search for population files.')
parser.add_argument('--aoi', type=str, default=None, help='Area of interest')
parser.add_argument('--min_edge_length', type=int, default=30, help='Edges less long than this are collapsed into one node.')
parser.add_argument('--h3_resolution', type=int, default=11, help='Resolution for h3 cells.')
parser.add_argument('--distance_steps', type=str, default=json.dumps([250,500,750,1000,1250,1500,1750,2000]), help='Resolution for h3 cells.')
parser.add_argument(
    '--h3',
    action='store_true',    
    help='Save results as h3 grid too. Default is False.'
)
parser.add_argument(
    '--overwrite',
    action='store_true',     # sets overwrite = True when used
    help='Overwrite all existing files. Default is False.'
)


# Parse all arguments
args = parser.parse_args()

poi_file = args.poi_file
output_path = args.output_path
street_path = args.street_path
if street_path is None:
    street_path = output_path

population_path = args.population_path
if population_path is None:
    population_path = output_path 

poi_quality_column = args.poi_quality_column
if poi_quality_column is not None:
    try:
        # Try JSON parsing (for cases like '["a","b"]')
        parsed = json.loads(poi_quality_column)

        # Only accept JSON lists. Otherwise treat as raw string
        if isinstance(parsed, list):
            poi_quality_column = parsed
        else:
            poi_quality_column = poi_quality_column   # keep original string

    except json.JSONDecodeError:
        # Not JSON → keep it as a normal string
        poi_quality_column = poi_quality_column

if isinstance(poi_quality_column,str):
    poi_quality_column = [poi_quality_column]

min_edge_length = args.min_edge_length
h3_resolution = args.h3_resolution

distance_steps = json.loads(args.distance_steps)

do_h3 = args.h3
do_population = args.population
overwrite = args.overwrite


# aoi 
if args.aoi is None:
    poi = gpd.read_file(poi_file)
    poi = poi.to_crs(poi.estimate_utm_crs()) # Convert to utm
    aoi = gpd.GeoDataFrame(geometry=[poi.union_all().envelope],crs=poi.crs) # Ensure there is only one polygon
    aoi = aoi.to_crs(aoi.estimate_utm_crs()) # Convert to utm
    aoi_download = aoi.buffer(max(distance_steps))
else:
    aoi = gpd.GeoDataFrame(
        geometry=[wkt.loads(args.aoi)],
        crs="EPSG:4326"
    )
    aoi = gpd.GeoDataFrame(geometry=[aoi.union_all()],crs="EPSG:4326") # Ensure there is only one polygon
    aoi = aoi.to_crs(aoi.estimate_utm_crs()) # Convert to utm
    aoi_download = aoi.buffer(max(distance_steps))
    info = pyogrio.read_info(poi_file) 
    poi_crs = info["crs"]
    poi = gpd.read_file(poi_file,bbox=tuple(aoi_download.to_crs(poi_crs).total_bounds))
    poi = poi.to_crs(poi.estimate_utm_crs()) # Convert to utm

aoi_download = aoi_download.intersection(shapely.buffer(poi.union_all(),max(distance_steps)))

poi['_service_quality'] = (
    poi[poi_quality_column]
    .astype(str)
    .agg('_-_'.join, axis=1)
)

osm_xml_file = os.path.normpath(output_path+f"/streets.osm")
streets_graph_path = os.path.normpath(output_path+f"/streets.graphml")
level_of_service_streets_path = os.path.normpath(output_path+f"/level_of_service_streets.gpkg")
h3_results_path = os.path.normpath(output_path+f"/h3.csv")


# Select what type of street network you want to load
network_filter = osm.osmium_network_filter("walk+bike+primary")
# Download the region pbf file crop it by aoi and convert to osm format
osm.geofabrik_to_osm(
    osm_xml_file,
    input_file=street_path,
    aoi=aoi_download,
    osmium_filter_args=network_filter,
    overwrite=False
)


# Load
G = ox.graph_from_xml(osm_xml_file)
# Project geometry coordinates to UTM system to allow euclidean meassurements in meters (sorry americans)
G = ox.project_graph(G,to_crs=aoi.estimate_utm_crs())
# Save the graph in graphml format to avoid the slow loading process
ox.save_graphml(G,streets_graph_path)

G = graph_processing.simplify_graph(G,min_edge_length=min_edge_length,min_edge_separation=min_edge_length*2,undirected=True)
# Save the result in graphml format
ox.save_graphml(G,streets_graph_path)

street_edges = ox.graph_to_gdfs(G,nodes=False)
street_edges = street_edges.to_crs(aoi.crs)

if (poi.geometry.type == 'Point').all():
    poi_points_gdf = poi.copy()
    is_point = True
else:
    poi_points_gdf = poi_utils.polygons_to_points(poi,street_edges)
    is_point = False

G, osmids = graph_processing.add_points_to_graph(
    poi_points_gdf,
    G,
    max_dist=100+min_edge_length, # Maximum distance from point to graph edge to project the point
    min_edge_length=min_edge_length # Minimum edge length after adding the new nodes
)
poi_points_gdf['osmid'] = osmids # Add the ids of the nodes in the graph to points
poi_points_gdf = poi_points_gdf.dropna(subset=['osmid'])

ls_columns = []
for quality_str in poi_points_gdf['_service_quality'].unique():
    ls_columns.append(quality_str)
    poi_selection = poi_points_gdf[poi_points_gdf['_service_quality'] == quality_str]
    level_of_service_graph = isochrones.graph(
        level_of_service_graph,
        poi_points_gdf,
        distance_steps,
        service_quality_col = None, # If all points have the same quality this could be None
        level_of_services = distance_steps, # could be None and it will set to the sorted unique values of the matrix
        min_edge_length = min_edge_length, # Do not add new nodes if there will be an edge with less than this length
    )
    # Save edges as gpkg
    level_of_service_nodes, level_of_service_edges = ox.graph_to_gdfs(level_of_service_graph)
    level_of_service_nodes = level_of_service_nodes.rename(columns={'level_of_service':f'ls_{quality_str}'})
    level_of_service_edges = level_of_service_edges.rename(columns={'level_of_service':f'ls_{quality_str}'})
    if is_point == False:
        level_of_service_nodes.loc[
            level_of_service_nodes.intersects(poi_selection.to_crs(level_of_service_nodes.crs).union_all()),f'ls_{quality_str}'
        ] = min(distance_steps)
        level_of_service_edges.loc[
            level_of_service_edges.intersects(poi_selection.to_crs(level_of_service_edges.crs).union_all()),f'ls_{quality_str}'
        ] = min(distance_steps)

    level_of_service_graph = ox.graph_from_gdfs(level_of_service_nodes, level_of_service_edges)


level_of_service_nodes, level_of_service_edges = ox.graph_to_gdfs(level_of_service_graph)
level_of_service_edges.to_file(level_of_service_streets_path)

if do_h3:
    aoi['id'] = 0
    ls_h3_df = h3_utils.from_gdf(aoi,value_column='id',value_order = [0], resolution=h3_resolution)
    ls_h3_df = ls_h3_df[['h3_cell']].drop_duplicates().reset_index(drop=True)

    for column in ls_columns:
        new_ls_h3_df = h3_utils.from_gdf(
            level_of_service_edges,
            resolution=h3_resolution,
            value_column=column,
            value_order=distance_steps,
            buffer=10
        )
        ls_h3_df = ls_h3_df.merge(new_ls_h3_df,on='h3_cell',how='left')


    ls_h3_df.to_csv(h3_results_path)