import geopandas as gpd
import pandas as pd

__all__ = ['overpass_api_query','download_graph','green_areas']

def overpass_api_query(query:str,bounds:gpd.GeoDataFrame|gpd.GeoSeries):
    import requests
    from osm2geojson import json2geojson

    bbox = bounds.to_crs(4326).total_bounds
    bbox = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    query = query.replace("{{bbox}}",bbox)
    query = query.replace("{bbox}",bbox)
    query = query.replace("bbox",bbox)
    query = query.replace("[out:xml]","[out:json]")
    overpass_url = 'https://overpass-api.de/api/interpreter'
    response = requests.get(overpass_url, params={'data': query})
    geojson_response = json2geojson(response.json())
    gdf = gpd.GeoDataFrame.from_features(geojson_response,crs=4326).reset_index(drop=True)
    new_gdf = gdf['tags'].apply(pd.Series)
    if 'type' in new_gdf.columns:
        new_gdf = new_gdf.rename(columns={'type':'geometry_type'})
    
    gdf = pd.concat([gdf.drop(columns=['tags']), new_gdf], axis=1).reset_index(drop=True)
    gdf = gdf.loc[:, ~gdf.columns.duplicated()]
    return gdf.to_crs(bounds.crs)

def download_graph(bounds:gpd.GeoSeries,network_type:str='walk',custom_filter=None):
    import osmnx as ox
    if bounds.crs.is_projected == True:
        crs = bounds.crs 
    else:
        crs = bounds.estimate_utm_crs()
    #network_type (string {"all", "all_public", "bike", "drive", "drive_service", "walk"}) – what type of street network to get if custom_filter is None
    G=ox.graph.graph_from_polygon(bounds.to_crs(4326).union_all(), network_type=network_type, simplify=True, retain_all=False, truncate_by_edge=True,custom_filter=custom_filter)
    G=ox.projection.project_graph(G,to_crs=crs)

    return G

def green_areas(bounds,pedestrian_graph=None):
    from . import service_quality
    query = """
        [out:json][timeout:25];
        (
        node[leisure = "garden"]({{bbox}});
        node[leisure = "park"]({{bbox}});
        node[landuse = "greenfield"]({{bbox}});
        node[landuse = "grass"]({{bbox}});
        node[landuse = "forest"]({{bbox}});
        way[leisure = "garden"]({{bbox}});
        way[leisure = "park"]({{bbox}});
        way[landuse = "greenfield"]({{bbox}});
        way[landuse = "grass"]({{bbox}});
        way[landuse = "forest"]({{bbox}});
        relation[leisure = "garden"]({{bbox}});
        relation[leisure = "park"]({{bbox}});
        relation[landuse = "greenfield"]({{bbox}});
        relation[landuse = "grass"]({{bbox}});
        relation[landuse = "forest"]({{bbox}});
        );
        out body;
        >;
        out skel qt;
    """
    green_areas_gdf = overpass_api_query(query,bounds)
    green_areas_gdf = service_quality.service_type(green_areas_gdf,row_values=['park','forest','garden','grass','greenfield'])
    green_areas_gdf = service_quality.merge_and_simplify_geometry(green_areas_gdf,buffer=5,min_width=10,simplify=1,min_area=200)

    if pedestrian_graph is not None:
        import osmnx as ox
        import shapely
        edges_union = ox.graph_to_gdfs(pedestrian_graph,nodes=False).to_crs(green_areas_gdf.crs).union_all()
        geoms = list(green_areas_gdf.geometry)
        shapely.prepare(geoms)
        green_areas_gdf = green_areas_gdf[shapely.intersects(geoms,edges_union)]

    return green_areas_gdf.to_crs(bounds.crs)