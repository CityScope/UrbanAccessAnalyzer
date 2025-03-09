import geopandas as gpd
import osmnx as ox
import polars as pl 
import pandas as pd 
import numpy as np
import networkx as nx
import warnings

def nodes_to_points(nodes,G): 
    from shapely.geometry import Point
    # Get point geometries for the given nodes
    point_geometries = [(node, Point((G.nodes[node]['x'], G.nodes[node]['y']))) for node in nodes if node in G.nodes]
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(point_geometries, columns=['osmid', 'geometry'])
    # Get the CRS from the graph
    crs = G.graph['crs']
    # Set the coordinate reference system (CRS) from the graph
    gdf.set_crs(crs, inplace=True)
    return gdf

def nearest_edges(geometries:gpd.GeoDataFrame|gpd.GeoSeries,G,max_dist:float=None):
    edges = ox.graph_to_gdfs(G,nodes=False)
    geom = geometries.geometry.to_crs(edges.crs)
    indices = edges.sindex.nearest(geom,max_distance=max_dist)
    # Create a DataFrame to store nearest node results
    indices = pd.DataFrame({
        'geom_idx': list(geometries.iloc[indices[0]].index),  # Indices of the geometries
        'edge_idx': list(edges.iloc[indices[1]].index)   # Corresponding indices of nearest nodes
    })
    
    # Drop duplicates to ensure a 1-to-1 mapping between geometries and nodes
    indices = indices.drop_duplicates('geom_idx').reset_index(drop=True)

    # Return lists of corresponding indices of geometries and nodes
    return list(indices['geom_idx']), list(indices['edge_idx'])
    #return list(geometries.iloc[indices[0,:]].index), list(edges.iloc[indices[1,:]].index)

#def nearest_nodes(geometries:gpd.GeoDataFrame|gpd.GeoSeries,G,max_dist:float=None):
#    nodes = ox.graph_to_gdfs(G,edges=False)
#    geom = geometries.geometry.to_crs(nodes.crs)
#    indices = nodes.sindex.nearest(geom,max_distance=max_dist)
#    indices = pd.DataFrame({'geom':list(indices[0,:]),'nodes':list(indices[1,:])})
#    indices = indices.drop_duplicates('geom').reset_index(drop=True)
#    return list(geometries.iloc[list(indices['geom'])].index), list(nodes.iloc[list(indices['nodes'])].index)

def nearest_nodes(geometries: gpd.GeoDataFrame | gpd.GeoSeries, G, max_dist: float = None):
    nodes = ox.graph_to_gdfs(G, edges=False)
    geom = geometries.geometry.to_crs(nodes.crs)

    # Find nearest nodes
    indices = nodes.sindex.nearest(geom, max_distance=max_dist, return_all=False)
    # Create a DataFrame to store nearest node results
    indices = pd.DataFrame({
        'geom_idx': list(geometries.iloc[indices[0]].index),  # Indices of the geometries
        'node_idx': list(nodes.iloc[indices[1]].index)   # Corresponding indices of nearest nodes
    })
    
    # Drop duplicates to ensure a 1-to-1 mapping between geometries and nodes
    indices = indices.drop_duplicates('geom_idx').reset_index(drop=True)

    # Return lists of corresponding indices of geometries and nodes
    return list(indices['geom_idx']), list(indices['node_idx'])

def add_points_to_graph_gdfs(points: gpd.GeoDataFrame | gpd.GeoSeries, nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame,
                             min_node_id: int = None, max_dist: float = None, min_dist: float = 0, indices: list = None, tolerance: float = 0.01):
    """ 
    Returns the nodes_gdf and edges_gdf and a list with the new node_ids
    """
    from . import utils
    import shapely
    import warnings
    import pandas as pd
    import numpy as np
    
    # Make copies of inputs to avoid modifying the original ones
    points = points.copy()
    nodes = nodes.copy()
    edges = edges.copy()

    if len(points) == 0:
        return nodes, edges, []

    if type(points) == gpd.GeoSeries:
        points = gpd.GeoDataFrame({},geometry=points.geometry,crs=points.crs)

    # Reproject points to match the CRS of edges
    points = points.to_crs(edges.crs)

    # Check if indices are provided and match the length of points
    if isinstance(indices, list) and len(indices) != len(points):
        raise Exception(f"Length of indices is {len(indices)} but you gave {len(points)} points.")

    if isinstance(indices, list):
        points['indices'] = indices
        # Ensure no duplicate node ids exist in the nodes GeoDataFrame
        if any(points['indices'].isin(nodes.index)):
            raise Exception(f"Node id {points.loc[points['indices'].isin(nodes.index),'indices']} already exists.")

    points = points.explode().reset_index(drop=True)
    

    # Create a copy of the edges GeoDataFrame for manipulation
    nearest_edges_df = edges.copy()
    
    # Assign unique edge IDs to edges
    unique_edge_ids = list(range(len(nearest_edges_df)))
    nearest_edges_df['edge_id'] = unique_edge_ids


    nearest_indices = nearest_edges_df.sindex.nearest(points.geometry.centroid, max_distance=max_dist)
    nearest_edges_helper = nearest_edges_df.iloc[nearest_indices[1, :]].copy()

    points = points.iloc[nearest_indices[0, :]]
    nearest_edges_helper['projected_dist'] = nearest_edges_helper.project(points.geometry.centroid, align=False)
    points['geometry'] = list(nearest_edges_helper.interpolate(nearest_edges_helper['projected_dist']))
    points = points.set_geometry('geometry')

    # Find the nearest edges to each point using spatial indexing
    nearest_indices = nearest_edges_df.sindex.nearest(points.geometry.centroid.buffer(tolerance, resolution=3), max_distance=tolerance)
    nearest_edges_df = nearest_edges_df.iloc[nearest_indices[1, :]]

    # If indices are provided, assign new node ids
    if isinstance(indices, list):
        nearest_edges_df['new_node_id'] = list(points.iloc[nearest_indices[0, :]]['indices'])

    # Project points onto nearest edges and calculate projected distance
    nearest_edges_df['projected_dist'] = nearest_edges_df.project(points.geometry.centroid.iloc[nearest_indices[0, :]], align=False)

    # Sort edges based on edge_id and projected distance
    nearest_edges_df = nearest_edges_df.sort_values(['edge_id', 'projected_dist']).reset_index()

    # Remove points too close to edge limits (less than min_dist)
    nearest_edges_df = nearest_edges_df.loc[nearest_edges_df['projected_dist'] > (min_dist + tolerance)]
    nearest_edges_df = nearest_edges_df.loc[(nearest_edges_df.geometry.length - nearest_edges_df['projected_dist']) > (min_dist + tolerance)]

    if len(nearest_edges_df) == 0:
        return nodes, edges, []

    # Remove duplicated points added to the same edge
    nearest_edges_df = nearest_edges_df.drop_duplicates(['edge_id', 'projected_dist'])
    nearest_edges_df = nearest_edges_df.sort_values(['edge_id', 'projected_dist']).reset_index(drop=True)

    # Ensure no points are added to the same edge within min_dist
    idx_help = np.array(nearest_edges_df.index) - 1
    idx_help[0] = 0 

    nearest_edges_df['projected_dist_next'] = list(np.abs(np.array(nearest_edges_df['projected_dist']) - np.array(nearest_edges_df.loc[idx_help, 'projected_dist'])))
    nearest_edges_df.loc[0, 'projected_dist_next'] = 999
    nearest_edges_df['edge_id_next'] = list(nearest_edges_df.loc[idx_help, 'edge_id'])
    nearest_edges_df = nearest_edges_df.loc[(nearest_edges_df['edge_id'] != nearest_edges_df['edge_id_next']) | (nearest_edges_df['projected_dist_next'] > (min_dist + tolerance))]
    nearest_edges_df = nearest_edges_df.drop(columns=['projected_dist_next', 'edge_id_next'])

    if len(nearest_edges_df) == 0:
        return nodes, edges, []

    # Interpolate and snap the projected points to the edge
    nearest_edges_df['projected_point'] = nearest_edges_df.interpolate(nearest_edges_df['projected_dist'])
    nearest_edges_df['projected_point'] = shapely.snap(nearest_edges_df['projected_point'], nearest_edges_df['projected_point'], tolerance=(min_dist + tolerance))
    
    # Create a unique identifier for each projected point to avoid duplicates
    nearest_edges_df['point_str'] = nearest_edges_df['projected_point'].to_wkt().astype(str)
    nearest_edges_df = nearest_edges_df.drop_duplicates(['point_str', 'edge_id'])

    indices_df = nearest_edges_df.drop_duplicates(['point_str'])

    # Handle node id assignment based on provided or default min_node_id
    if indices is None:
        if min_node_id is None:
            warnings.warn("min_node_id keyword was not provided. New node ids could be wrong.")
            min_node_id = min(nodes.index)
        
        if min_node_id > min(nodes.index):
            warnings.warn(f"min_node_id keyword {min_node_id} is larger than the minimum node id. New node ids could be wrong.")
            min_node_id = min(nodes.index)

        min_node_id = min_node_id - 1
        if min_node_id > 0:
            min_node_id = -1

        indices_df = pd.DataFrame({'point_str': list(indices_df['point_str']),
                                   'new_node_id': range((min_node_id - len(indices_df)) + 1, min_node_id + 1)})
    else:
        indices_df = pd.DataFrame({'point_str': list(indices_df['point_str']),
                                   'new_node_id': list(indices_df['new_node_id'])})
        nearest_edges_df = nearest_edges_df.drop(columns=['new_node_id'])

    nearest_edges_df = nearest_edges_df.merge(indices_df, on='point_str')
    nearest_edges_df = nearest_edges_df.drop(columns=['point_str'])
    nearest_edges_df = nearest_edges_df.sort_values(['edge_id', 'projected_dist']).reset_index(drop=True)

    # Create a group of edges and update the geometries
    edges_group = nearest_edges_df[['edge_id', 'projected_point']]
    edges_group = edges_group.set_geometry('projected_point', crs=nearest_edges_df.crs)
    edges_group = edges_group.dissolve(['edge_id'], sort=False).sort_values('edge_id').reset_index()
    edges_group['edge_geometry'] = list(nearest_edges_df.drop_duplicates(['edge_id']).sort_values('edge_id').geometry)

    # Split geometries and assign the updated geometries to edges
    for i in edges_group.index:
        edges_group.loc[i, 'edge_geometry'] = utils.complex_split(edges_group.loc[i, 'edge_geometry'], edges_group.loc[i, 'projected_point'], tolerance=tolerance)

    edges_group = edges_group.set_geometry('edge_geometry', crs=nearest_edges_df.crs)
    edges_group = edges_group.explode('edge_geometry')

    new_edges = nearest_edges_df.drop_duplicates('edge_id')
    new_edges = new_edges.merge(edges_group[['edge_id', 'edge_geometry']], on='edge_id')
    new_edges = new_edges.rename(columns={'geometry': 'temp', 'edge_geometry': 'geometry'})
    new_edges = new_edges.drop(columns=['temp'])
    new_edges['length'] = new_edges.geometry.length
    new_edges = new_edges.sort_values(['edge_id', 'projected_dist'])

    # Identify unique edge indices
    u_inds = np.unique(new_edges['edge_id'].searchsorted(new_edges['edge_id'], side='left'))
    v_inds = np.unique(new_edges['edge_id'].searchsorted(new_edges['edge_id'], side='right') - 1)

    # Assign new node ids to edges
    new_edges.loc[new_edges.index.isin(u_inds) == False, 'u'] = list(nearest_edges_df.sort_values(['edge_id', 'projected_dist'])['new_node_id'])
    new_edges.loc[new_edges.index.isin(v_inds) == False, 'v'] = list(nearest_edges_df.sort_values(['edge_id', 'projected_dist'])['new_node_id'])

    # Update edge indices
    new_edges = new_edges.set_index(['u', 'v', 'key'])
    new_edges = new_edges.drop(columns=['edge_id', 'projected_point', 'projected_dist', 'new_node_id'])

    # Exclude edges to be deleted
    delete_edge_ids = pd.MultiIndex.from_frame(nearest_edges_df[['u', 'v', 'key']])
    all_edges = edges.loc[edges.index.isin(delete_edge_ids) == False]
    all_edges = pd.concat([all_edges, new_edges])

    # Add new nodes
    new_nodes = nodes.loc[list(nearest_edges_df['u'])]
    new_nodes['osmid'] = list(nearest_edges_df['new_node_id'])
    new_nodes.geometry = list(nearest_edges_df['projected_point'])
    new_nodes = new_nodes.drop_duplicates(['osmid'])
    new_nodes['x'] = new_nodes.geometry.get_coordinates()['x']
    new_nodes['y'] = new_nodes.geometry.get_coordinates()['y']
    new_nodes = new_nodes.set_index('osmid')

    # Combine original and new nodes
    all_nodes = pd.concat([nodes, new_nodes])
    all_nodes['x'] = all_nodes.geometry.get_coordinates()['x']
    all_nodes['y'] = all_nodes.geometry.get_coordinates()['y']

    return all_nodes, all_edges, list(new_nodes.index)

def add_points_to_graph_gdfs_old(points:gpd.GeoDataFrame|gpd.GeoSeries,nodes:gpd.GeoDataFrame,edges:gpd.GeoDataFrame,
                            min_node_id:int=None,max_dist:float=None,min_dist:float=0,indices:list=None,tolerance:float=0.001):
    """ 
    Returns the nodes_gdf and edges_gdf and a list with the new node_ids
    """
    from . import utils
    import shapely
    if len(points) == 0:
        return nodes, edges, []
    
    unique_edge_ids = list(range(len(edges)))
    edges['edge_id'] = unique_edge_ids

    points = points.to_crs(edges.crs)
    if (type(indices) == list) and (len(indices) != len(points)):
        raise Exception(f"Length of indices is {len(indices)} but you gave {len(points)} points.")

    if type(indices) == list:
        points['indices'] = indices 
        if any(points['indices'].isin(nodes.index)):
            raise Exception(f"Node id {points.loc[points['indices'].isin(nodes.index),'indices']} already exists.")

    points = points.explode().reset_index(drop=True)

    nearest_edges_df = edges.copy()

    nearest_indices = nearest_edges_df.sindex.nearest(points.geometry.centroid.buffer(tolerance,resolution=3),max_distance=max_dist)
    nearest_edges_df = nearest_edges_df.iloc[nearest_indices[1,:]]

    if type(indices) == list:
        nearest_edges_df['new_node_id'] = list(points.iloc[nearest_indices[0,:]]['indices'])

    nearest_edges_df['projected_dist'] = nearest_edges_df.project(points.geometry.centroid.iloc[nearest_indices[0,:]],align=False)

    nearest_edges_df = nearest_edges_df.sort_values(['edge_id','projected_dist']).reset_index()

    # Delete points if they are less than min_dist from edge limits

    nearest_edges_df = nearest_edges_df.loc[nearest_edges_df['projected_dist'] > (min_dist + tolerance)]
    nearest_edges_df = nearest_edges_df.loc[(nearest_edges_df.geometry.length - nearest_edges_df['projected_dist']) > (min_dist + tolerance)]

    if len(nearest_edges_df) == 0:
        return nodes, edges, []

    # Delete repeated points to be added on the same edge

    nearest_edges_df = nearest_edges_df.drop_duplicates(['edge_id','projected_dist'])
    nearest_edges_df = nearest_edges_df.sort_values(['edge_id','projected_dist']).reset_index(drop=True)

    # Do not add point if another point is going to be added to the same edge and it is less than min_dist away.

    idx_help = np.array(nearest_edges_df.index) - 1
    idx_help[0] = 0 

    nearest_edges_df['projected_dist_next'] = list(np.abs(np.array(nearest_edges_df['projected_dist']) - np.array(nearest_edges_df.loc[idx_help,'projected_dist'])))
    nearest_edges_df.loc[0,'projected_dist_next'] = 999 
    nearest_edges_df['edge_id_next'] = list(nearest_edges_df.loc[idx_help,'edge_id'])
    nearest_edges_df = nearest_edges_df.loc[(nearest_edges_df['edge_id'] != nearest_edges_df['edge_id_next']) | (nearest_edges_df['projected_dist_next'] > (min_dist + tolerance))]
    nearest_edges_df = nearest_edges_df.drop(columns=['projected_dist_next','edge_id_next'])

    if len(nearest_edges_df) == 0:
        return nodes, edges, []

    nearest_edges_df['projected_point'] = nearest_edges_df.interpolate(nearest_edges_df['projected_dist'])
    nearest_edges_df['projected_point'] = shapely.snap(nearest_edges_df['projected_point'],nearest_edges_df['projected_point'],tolerance=(min_dist + tolerance))
    
    nearest_edges_df['point_str'] = nearest_edges_df['projected_point'].to_wkt().astype(str)
    nearest_edges_df = nearest_edges_df.drop_duplicates(['point_str','edge_id'])
    
    indices_df = nearest_edges_df.drop_duplicates(['point_str'])

    if type(indices) == type(None):
        if type(min_node_id) == type(None):
            warnings.warn("min_node_id keyword was not provided. New node ids could be wrong.")
            min_node_id = min(nodes.index)
        
        if min_node_id > min(nodes.index):
            warnings.warn(f"min_node_id keyword {min_node_id} is larger than the minimum node id. New node ids could be wrong.")
            min_node_id = min(nodes.index)

        min_node_id = min_node_id - 1
        if min_node_id > 0:
            min_node_id = -1

        indices_df = pd.DataFrame({'point_str' : list(indices_df['point_str']),
        'new_node_id' : range((min_node_id - len(indices_df)) + 1,min_node_id+1)
        })
    else:
        indices_df = pd.DataFrame({'point_str' : list(indices_df['point_str']),
        'new_node_id' : list(indices_df['new_node_id'])
        })
        nearest_edges_df = nearest_edges_df.drop(columns=['new_node_id'])

    nearest_edges_df = nearest_edges_df.merge(indices_df,on='point_str')
    nearest_edges_df = nearest_edges_df.drop(columns=['point_str'])
    nearest_edges_df = nearest_edges_df.sort_values(['edge_id','projected_dist']).reset_index(drop=True)

    edges_group = nearest_edges_df[['edge_id','projected_point']]
    edges_group = edges_group.set_geometry('projected_point',crs=nearest_edges_df.crs)
    edges_group = edges_group.dissolve(['edge_id'],sort=False).sort_values('edge_id').reset_index()
    edges_group['edge_geometry'] = list(nearest_edges_df.drop_duplicates(['edge_id']).sort_values('edge_id').geometry)

    for i in edges_group.index:
        edges_group.loc[i,'edge_geometry'] = utils.complex_split(edges_group.loc[i,'edge_geometry'],edges_group.loc[i,'projected_point'],tolerance=tolerance)

    edges_group = edges_group.set_geometry('edge_geometry',crs=nearest_edges_df.crs)
    edges_group = edges_group.explode('edge_geometry')
    #edges_group = edges_group.loc[edges_group.geometry.length > tolerance]

    new_edges = nearest_edges_df.drop_duplicates('edge_id')
    new_edges = new_edges.merge(edges_group[['edge_id','edge_geometry']],on='edge_id')
    new_edges = new_edges.rename(columns={'geometry':'temp','edge_geometry':'geometry'})
    new_edges = new_edges.drop(columns=['temp'])
    new_edges['length'] = new_edges.geometry.length
    new_edges = new_edges.sort_values(['edge_id','projected_dist'])

    u_inds = np.unique(new_edges['edge_id'].searchsorted(new_edges['edge_id'],side='left'))
    v_inds = np.unique(new_edges['edge_id'].searchsorted(new_edges['edge_id'],side='right') - 1)

    new_edges.loc[new_edges.index.isin(u_inds) == False,'u'] = list(nearest_edges_df.sort_values(['edge_id','projected_dist'])['new_node_id'])
    new_edges.loc[new_edges.index.isin(v_inds) == False,'v'] = list(nearest_edges_df.sort_values(['edge_id','projected_dist'])['new_node_id'])

    #new_edges.index=pd.MultiIndex.from_frame(new_edges[['u','v','key']])
    new_edges = new_edges.set_index(['u','v','key'])
    new_edges=new_edges.drop(columns=['edge_id','projected_point','projected_dist','new_node_id'])

    delete_edge_ids = pd.MultiIndex.from_frame(nearest_edges_df[['u','v','key']])
    all_edges = edges.loc[edges.index.isin(delete_edge_ids) == False]
    all_edges = pd.concat([all_edges,new_edges])

    new_nodes = nodes.loc[list(nearest_edges_df['u'])]

    new_nodes['osmid'] = list(nearest_edges_df['new_node_id'])
    new_nodes.geometry = list(nearest_edges_df['projected_point'])
    new_nodes = new_nodes.drop_duplicates(['osmid'])
    new_nodes['x'] = new_nodes.geometry.get_coordinates()['x']
    new_nodes['y'] = new_nodes.geometry.get_coordinates()['y']
    new_nodes = new_nodes.set_index('osmid')

    all_nodes = pd.concat([nodes,new_nodes])
    all_nodes['x'] = all_nodes.geometry.get_coordinates()['x']
    all_nodes['y'] = all_nodes.geometry.get_coordinates()['y']

    return all_nodes,all_edges, list(new_nodes.index)

def add_points_to_graph(points:gpd.GeoDataFrame|gpd.GeoSeries,G,max_dist:float=None,min_dist:float=0,indices:list=None,tolerance:float=0.01):
    """ 
    Returns the graph and a list with the new node_ids
    """
    if len(points) == 0:
        return G, []
    
    nodes, edges = ox.graph_to_gdfs(G)
    all_nodes, all_edges, new_node_ids = add_points_to_graph_gdfs(points,nodes,edges,
                                            min_node_id=min(G.nodes),max_dist=max_dist,min_dist=min_dist,indices=indices,tolerance=tolerance)
    if len(new_node_ids) == 0:
        return G, []
    
    return ox.graph_from_gdfs(all_nodes,all_edges,graph_attrs=G.graph), new_node_ids

def add_intersection_to_graph(geometries:gpd.GeoSeries|gpd.GeoDataFrame,G,max_dist:float=None,min_dist:float=0,indices:list=None,tolerance=1.0e-4,_edge_union=None):
    """ 
    Returns the graph and a list with the new node_ids
    """
    import shapely
    if type(_edge_union) == type(None):
        edges = ox.graph_to_gdfs(G,nodes=False)
        crs = edges.crs
        _edge_union = edges.union_all()
    else:
        if (type(_edge_union) == gpd.GeoSeries) or (type(_edge_union) == gpd.GeoDataFrame):
            crs = _edge_union.crs 
            _edge_union = _edge_union.union_all()
        else:
            crs = graph_crs(G)

    geometries = geometries.geometry.to_crs(crs) 

    is_polygon = (geometries.type == 'Polygon') | (geometries.type == 'MultiPolygon')
    geometries.loc[is_polygon] = geometries.loc[is_polygon].boundary

    geometries = list(geometries)
    inter = shapely.intersection(geometries,_edge_union)
    inter = gpd.GeoSeries(inter,crs=crs)

    if type(indices) == list:
        if len(indices) != len(geometries):
            raise Exception(f"Length of geometries is {len(geometries)} but length of indices is {len(indices)}.")
        
        indices = list(np.array(indices)[np.array(inter.is_empty == False)])

    inter = inter.loc[inter.is_empty == False]

    G_copy, nodes = add_points_to_graph(points=inter,G=G,max_dist=max_dist,min_dist=min_dist,indices=indices,tolerance=tolerance)
    return G_copy, nodes

def geometries_to_nodes(geometries:gpd.GeoSeries|gpd.GeoDataFrame,G,min_dist:float=0,max_dist:float=None,add_points:bool = True, edges:gpd.GeoDataFrame=None):
    """
    Returns gpd.GeoDataFrame with geometries and node_ids repeating the geometry if there are multiple matching node_ids.
        and the graph
    """
    import shapely
    G_copy = G.copy()
    new_geometries = geometries.copy().reset_index(drop=True)
    new_geometries['geometry_id'] = new_geometries.index
    #new_geometries['osmid'] = np.nan
    points = new_geometries.geometry.loc[(new_geometries.geometry.type == 'Point') | (new_geometries.geometry.type == 'MultiPoint')]
    if len(points) > 0:
        if add_points:
            G_copy,_ = add_points_to_graph(points,G_copy,min_dist=min_dist,max_dist=max_dist,tolerance=1.0e-4)

        ids,n = nearest_nodes(points,G_copy,max_dist=max_dist)
        if len(ids) > 0:
            new_geometries.loc[ids,'osmid'] = n

    polygons_and_lines = new_geometries.geometry.loc[(new_geometries.geometry.type != 'Point') & (new_geometries.geometry.type != 'MultiPoint')]
    if len(polygons_and_lines) > 0:
        if type(edges) == type(None):
            edges = ox.graph_to_gdfs(G_copy,nodes=False)

        polygons_and_lines = polygons_and_lines.to_crs(edges.crs)
        is_polygon = (polygons_and_lines.type == 'Polygon') | (polygons_and_lines.type == 'MultiPolygon')
        polygons_and_lines.loc[is_polygon] = polygons_and_lines.loc[is_polygon].boundary
        inter = shapely.intersection(list(polygons_and_lines),edges.union_all())
        inter = gpd.GeoDataFrame(geometry=inter,crs=edges.crs)
        inter['ids'] = polygons_and_lines.index
        inter = inter.explode('geometry').reset_index(drop=True)
        inter = inter.loc[inter.geometry.isna() == False]
        inter = inter.loc[inter.geometry.is_empty == False].reset_index(drop=True)
        if add_points:
            G_copy,_ = add_points_to_graph(points=inter,G=G_copy,max_dist=0.001,min_dist=min_dist,tolerance=1.0e-4)
        
        if min_dist <= 0:
            min_dist = 0.001

        ids,n = nearest_nodes(inter,G_copy,max_dist = (min_dist + 1.0e-4))
        if len(ids) > 0:
            ids = inter.loc[ids,'ids']
            new_geoms = new_geometries.loc[ids]
            new_geoms['osmid'] = n
            new_geoms = new_geoms.drop_duplicates()
            new_geometries = pd.concat([new_geometries,new_geoms])

    if 'osmid' not in new_geometries.columns:
        return gpd.GeoDataFrame([],geometry=[],crs=new_geometries.crs), G

    new_geometries = new_geometries.loc[new_geometries['osmid'].isna()==False]
    new_geometries['osmid'] = new_geometries['osmid'].astype(int)
    return new_geometries.drop_duplicates(), G_copy

def graph_crs(G):
    return ox.graph_to_gdfs(G,edges=False).crs

def add_node_elevations_open_api(G):
    orig_template = ox.settings.elevation_url_template
    ox.settings.elevation_url_template = 'https://api.open-elevation.com/api/v1/lookup?locations={locations}'
    crs = graph_crs(G)
    G = ox.projection.project_graph(G,to_latlong=True)
    G = ox.add_node_elevations_google(G,batch_size=250)
    ox.settings.elevation_url_template = orig_template
    G = ox.projection.project_graph(G,to_crs=crs)
    return G

def cluster_nodes(G,distance:float = 50,highway_priority:list = ['footway','residential']):
    from sklearn.cluster import AgglomerativeClustering
    import shapely
    from . import utils

    nodes,edges = ox.convert.graph_to_gdfs(G)
    crs = edges.crs
    if 'maxspeed' not in edges.columns:
        edges['maxspeed'] = None 

    edges = edges.reset_index()[['u','v','key','highway','maxspeed','length','geometry']]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        edges['geometry'] = edges['geometry'].to_wkt()
        edges['geometry'] = edges['geometry'].astype(str)

    edges['highway'] = edges['highway'].astype(str)
    edges['maxspeed'] = edges['maxspeed'].astype(str)
    edges = pl.from_pandas(edges)
    nodes_geometry = nodes.geometry
    nodes = nodes.drop(columns=['geometry'])
    nodes = pl.from_pandas(nodes,include_index=True)
    
    cluster_func = AgglomerativeClustering(n_clusters=None,distance_threshold=distance, metric='euclidean',linkage='complete')

    node_groups = gpd.GeoSeries(shapely.get_parts(nodes_geometry.buffer(distance/2,resolution=4).union_all()),crs=crs)
    node_groups = utils.intersects_all_with_all(nodes_geometry,node_groups).astype(int) 
    n_node_groups = node_groups.shape[1]
    node_groups *= np.arange(n_node_groups)
    node_groups = np.max(node_groups,axis=1)
    nodes = nodes.with_columns(
        node_group=node_groups,
        cluster_id=0
    )
    
    max_cluster_id = -1
    for i in range(n_node_groups):
        mask = (nodes.select('node_group')==i).to_series()
        nodes_subdf = nodes.filter(mask).select('x','y')
        if len(nodes_subdf) == 0:
            continue
        
        if len(nodes_subdf) == 1:
            clusters = [0]
        else:
            clusters = cluster_func.fit(nodes_subdf)#list(zip(x,y)))
            clusters = clusters.labels_

        clusters = np.array(clusters) + max_cluster_id + 1
        max_cluster_id = int(np.max(clusters))
        clusters_long = np.zeros(len(nodes),dtype=int)
        clusters_long[mask.to_numpy()] = clusters
        nodes = nodes.with_columns(
            cluster_id = pl.when(mask).then(clusters_long).otherwise(pl.col('cluster_id'))
        )

    nodes = nodes.group_by('cluster_id').agg(
        pl.col('osmid').first(),
        pl.col('osmid').alias('all_osmids'),
        pl.col('x').mean(),
        pl.col('y').mean()
    )
    edges = edges.join(nodes[['osmid','all_osmids','x','y']].explode('all_osmids'),left_on='u',right_on='all_osmids')
    edges = edges.drop('u').rename({'osmid':'u','x':'x_u','y':'y_u'})
    edges = edges.join(nodes[['osmid','all_osmids','x','y']].explode('all_osmids'),left_on='v',right_on='all_osmids')
    edges = edges.drop('v').rename({'osmid':'v','x':'x_v','y':'y_v'})

    edges = edges.filter(pl.col('u') != pl.col('v'))
    edges=edges.group_by('u','v').agg(
        pl.col('highway').str.concat(","),
        pl.col('length','maxspeed','geometry').filter( #### revisar maxspeed
            pl.col('length')==pl.col('length').min()
        ).first(),
        pl.col('x_u','y_u','x_v','y_v').first()
    ).with_columns(key=0).sort('u','v','key')

    for i in highway_priority: 
        edges = edges.with_columns(
            highway = pl.when(pl.col('highway').str.contains(i)).then(pl.lit(i)).otherwise(pl.col('highway'))
        )

    edges=edges.with_columns(
    geometry = pl.col('geometry').str.replace("LINESTRING (","LINESTRING (" + pl.col('x_u').cast(str) + " " + pl.col('y_u').cast(str) + ",",literal=True)
    )
    edges=edges.with_columns(
        geometry = pl.col('geometry').str.replace(")","," + pl.col('x_v').cast(str) + " " + pl.col('y_v').cast(str) + ")",literal=True)
    )
    edges=edges.drop('x_u','y_u','x_v','y_v','length')

    nodes_gdf = nodes.to_pandas()
    nodes_gdf = gpd.GeoDataFrame(nodes_gdf[['osmid','y','x']], geometry=gpd.points_from_xy(nodes_gdf["x"],nodes_gdf["y"]),crs=crs)
    nodes_gdf = nodes_gdf.set_index('osmid')#.index = nodes_gdf['osmid']
    #nodes_gdf = nodes_gdf.drop(columns='osmid')

    edges_gdf = edges.to_pandas()
    edges_gdf = edges_gdf.set_index(['u','v','key'])
    #edges_gdf.index=pd.MultiIndex.from_frame(edges_gdf[['u','v','key']])
    #edges_gdf=edges_gdf.drop(columns=['u','v','key'])
    edges_gdf = gpd.GeoDataFrame(edges_gdf,geometry=gpd.GeoSeries.from_wkt(edges_gdf['geometry']),crs=crs)
    edges_gdf['length'] = edges_gdf.geometry.length

    nodes_gdf['x'] = nodes_gdf.geometry.get_coordinates()['x']
    nodes_gdf['y'] = nodes_gdf.geometry.get_coordinates()['y']

    G_copy = ox.graph_from_gdfs(gdf_nodes=nodes_gdf,gdf_edges=edges_gdf,graph_attrs=G.graph)
    return G_copy

def multi_ego_graph(G, n, radius:float=1, center:bool=True, undirected:bool=False, distance:str=None):
    """Returns induced subgraph of neighbors centered at node n within
    a given radius.

    Parameters
    ----------
    G : graph
      A NetworkX Graph or DiGraph

    n : node
      A single node or multiple

    radius : number, optional
      Include all neighbors of distance<=radius from n.

    center : bool, optional
      If False, do not include center node in graph

    undirected : bool, optional
      If True use both in- and out-neighbors of directed graphs.

    distance : key, optional
      Use specified edge data key as distance.  For example, setting
      distance='weight' will use the edge weight to measure the
      distance from the node n.

    Notes
    -----
    For directed graphs D this produces the "out" neighborhood
    or successors.  If you want the neighborhood of predecessors
    first reverse the graph with D.reverse().  If you want both
    directions use the keyword argument undirected=True.

    Node, edge, and graph attributes are copied to the returned subgraph.
    """
    if undirected:
        if type(distance) is str:
            sp, _ = nx.multi_source_dijkstra(
                G.to_undirected(), n, cutoff=radius, weight=distance
            )
        else:
            sp = dict(
                nx.multi_source_dijkstra_path_length(
                    G.to_undirected(), n, cutoff=radius
                )
            )
    else:
        if type(distance) is str:
            sp, _ = nx.multi_source_dijkstra(G, n, cutoff=radius, weight=distance)
        else:
            sp = dict(nx.multi_source_dijkstra_path_length(G, n, cutoff=radius))

    H = G.subgraph(sp).copy()
    nx.set_node_attributes(H,sp,'dist_to_center')
    if not center:
        H.remove_node(n)

    return H

def isochrone_gdfs(G,n,radius:float,undirected:bool=False,distance:str='length',center:bool=True):
    H = multi_ego_graph(G,n,radius=radius,undirected=undirected,distance=distance,center=center)
    if len(H.edges) == 0:
        nodes, edges = ox.graph_to_gdfs(G)
        iso_edges = gpd.GeoDataFrame(columns=edges.columns, crs=edges.crs, geometry=[])
        cols = list(nodes.columns) 
        if 'dist_to_center' not in cols:
            cols.append('dist_to_center')

        iso_nodes = gpd.GeoDataFrame(columns=cols, crs=nodes.crs, geometry=[])
    else:
        iso_nodes, iso_edges = ox.graph_to_gdfs(H)

    return iso_nodes, iso_edges

def isochrone_exact_boundary_gdfs(nodes:gpd.GeoDataFrame,edges:gpd.GeoDataFrame,iso_nodes:gpd.GeoDataFrame,iso_edges:gpd.GeoDataFrame = None,
                                    radius:float = -1,undirected:bool=False,min_node_id:int=None,min_dist:float=0,accessibility=None):
    import shapely 
    """dist_to_center is lost in iso_nodes"""

    if type(min_node_id) == type(None):
        min_node_id = min(nodes.index)

    if type(iso_edges) == type(None):
        if undirected:
            edges_help = edges.reset_index()
            iso_edges = edges.loc[edges_help['u'].isin(iso_nodes.index) & edges_help['v'].isin(iso_nodes.index)]
        else:
            raise Exception("For a directed graph the iso_edges keyword must be set.")

    iso_nodes['border_dist'] = radius - iso_nodes['dist_to_center']

    edges_border = edges[edges['accessibility'] > accessibility].loc[list(edges[edges['accessibility'] > accessibility].reset_index()['u'].isin(iso_nodes.index))]
    edges_border['border_dist_u'] = edges_border.index.get_level_values('u').map(iso_nodes['border_dist'])
    edges_border['border_dist_v'] = edges_border.index.get_level_values('v').map(iso_nodes['border_dist'])
    edges_border['border_dist'] = edges_border[['border_dist_u', 'border_dist_v']].max(axis=1, skipna=True)
    edges_border = edges_border.loc[edges_border['border_dist'] > (min_dist+0.1)]
    edges_border = edges_border.loc[edges_border['border_dist'] < edges_border.geometry.length]
    edges_border['edge_id'] = list(edges_border.reset_index().apply(lambda x: str(np.sort([x['u'],x['v']])),axis=1))
    sum_per_edge = edges_border.groupby('edge_id')['border_dist'].sum()
    edges_border['total_border_len'] = edges_border['edge_id'].map(sum_per_edge)
    edges_border = edges_border.loc[edges_border.geometry.length >= edges_border['total_border_len']]
    border_points = edges_border.interpolate(edges_border['border_dist'])
    nodes_border = nodes.loc[nodes.index.isin(
            list(edges_border.reset_index()['u']) + list(edges_border.reset_index()['v'])
        )]
    edges_border = pd.concat([
            edges_border,
            edges.loc[list(edges.reset_index()['u'].isin(edges_border.reset_index()['v']) & edges.reset_index()['v'].isin(edges_border.reset_index()['u']))] 
            ]).drop_duplicates()

    new_nodes_border, new_edges_border, _ = add_points_to_graph_gdfs(border_points,nodes_border,edges_border,
                                                        min_node_id=min(nodes.index),max_dist=0.1,min_dist=min_dist,tolerance=0.1)
    
    nearest_indices = new_nodes_border.sindex.nearest(border_points.geometry, max_distance=min_dist+0.1)
    node_ids = list(new_nodes_border.iloc[nearest_indices[1, :]].index)
    
    edges = pd.concat([
        edges.loc[edges.index.isin(edges_border.index) == False],
        new_edges_border
    ])

    nodes = pd.concat([
        nodes.loc[nodes.index.isin(new_nodes_border.index) == False],
        new_nodes_border
    ])

    edges_help = new_edges_border.reset_index()
    edges_border_out = new_edges_border.loc[list(
                    edges_help['u'].isin(iso_nodes.index) & edges_help['v'].isin(node_ids)
                )]

    edges_border = pd.concat([edges_border_out,
            new_edges_border.loc[list(
                            edges_help['v'].isin(edges_border_out.reset_index()['u']) & edges_help['u'].isin(edges_border_out.reset_index()['v'])
                        )]
        ])
    edges_border = edges_border[~edges_border.index.duplicated(keep='first')]

    if undirected:
        edges_border_out = edges_border

    if len(iso_edges) > 0: 
        iso_edges = edges.loc[list((edges.reset_index()['u'].isin(list(iso_edges.reset_index()['u'])) & edges.reset_index()['v'].isin(list(iso_edges.reset_index()['v']))) | 
            (edges.reset_index()['v'].isin(list(iso_edges.reset_index()['u'])) & edges.reset_index()['u'].isin(list(iso_edges.reset_index()['v']))))]

        iso_edges = pd.concat([
            iso_edges,
            edges.loc[edges.index.isin(edges_border.index)] 
            ]).drop_duplicates()
    else:
        iso_edges = edges.loc[edges.index.isin(edges_border.index)]

    iso_nodes = nodes.loc[nodes.index.isin(iso_edges.reset_index()['u']) | nodes.index.isin(iso_edges.reset_index()['v'])].drop_duplicates()
    
    nodes = nodes.drop(columns=['total_border_len','border_dist','border_dist_u','border_dist_v','edge_id'],errors='ignore')
    edges = edges.drop(columns=['total_border_len','border_dist','border_dist_u','border_dist_v','edge_id'],errors='ignore')
    iso_nodes = iso_nodes.drop(columns=['total_border_len','border_dist','border_dist_u','border_dist_v','edge_id'],errors='ignore')
    iso_edges = iso_edges.drop(columns=['total_border_len','border_dist','border_dist_u','border_dist_v','edge_id'],errors='ignore')

    return nodes, edges, iso_nodes, iso_edges

def isochrone_exact_boundary_gdfs_border(nodes:gpd.GeoDataFrame,edges:gpd.GeoDataFrame,iso_nodes:gpd.GeoDataFrame,iso_edges:gpd.GeoDataFrame = None,
                                    radius:float = -1,undirected:bool=False,min_node_id:int=None,min_dist:float=0,accessibility=None):
    import shapely 
    """dist_to_center is lost in iso_nodes"""

    if type(min_node_id) == type(None):
        min_node_id = min(nodes.index)

    if type(iso_edges) == type(None):
        if undirected:
            edges_help = edges.reset_index()
            iso_edges = edges.loc[edges_help['u'].isin(iso_nodes.index) & edges_help['v'].isin(iso_nodes.index)]
        else:
            raise Exception("For a directed graph the iso_edges keyword must be set.")

    edges_help = edges.reset_index()
    edges_border_out = edges.loc[list(
                    edges_help['u'].isin(iso_nodes.index
                             ) & (
                    edges_help['v'].isin(iso_nodes.index) == False)
                )]
    
    if accessibility:
        edges_border_out = edges_border_out.loc[edges_border_out['accessibility'] > accessibility]

    if (len(edges_border_out) == 0):
        return nodes, edges, iso_nodes, iso_edges

    edges_border = pd.concat([edges_border_out,
            edges.loc[list(edges_help['v'].isin(edges_border_out.reset_index()['u']) & edges_help['u'].isin(edges_border_out.reset_index()['v']))]
        ])

    if accessibility:
        edges_border = edges_border.loc[edges_border['accessibility'] > accessibility]

    edges_border = edges_border[~edges_border.index.duplicated(keep='first')]
    
    if undirected:
        nodes_border = iso_nodes.loc[iso_nodes.index.isin(
            list(edges_border.reset_index()['u']) + list(edges_border.reset_index()['v'])
        )]
        edges_border_out = edges_border
    else:
        nodes_border = iso_nodes.loc[iso_nodes.index.isin(
            list(edges_border_out.reset_index()['u'])
        )]

    edges_border_out['edge_id'] = list(edges_border_out.reset_index().apply(lambda x: str(np.sort([x['u'],x['v']])),axis=1))
    nodes_border = nodes_border.loc[(radius - nodes_border['dist_to_center']) > 0]
    nodes_border['border_dist'] = radius - nodes_border['dist_to_center']
    edges_border_out = edges_border_out[list(edges_border_out.reset_index()['u'].isin(nodes_border.index))]
    edges_border_out['projected_dist'] = edges_border_out.index.get_level_values('u').map(nodes_border['border_dist'])
    sum_per_edge = edges_border_out.groupby('edge_id')['projected_dist'].sum()
    edges_border_out['proj_len_total'] = edges_border_out['edge_id'].map(sum_per_edge)
    edges_border_out = edges_border_out[edges_border_out.geometry.length >= edges_border_out['proj_len_total']]
    #edges_border_out = edges_border_out.loc[edges_border_out.groupby("edge_id")["projected_dist"].idxmax()]
    border_points = edges_border_out.interpolate(edges_border_out['projected_dist'])
    nodes_edges_border = nodes.loc[nodes.index.isin(
            list(edges_border.reset_index()['u']) + list(edges_border.reset_index()['v'])
        )]
    new_nodes_edges_border, new_edges_border, _ = add_points_to_graph_gdfs(border_points,nodes_edges_border,edges_border,
                                                        min_node_id=min(nodes.index),max_dist=0.1,min_dist=min_dist,tolerance=0.1)
    
    nearest_indices = new_nodes_edges_border.sindex.nearest(border_points.geometry, max_distance=min_dist+0.1)
    node_ids = list(new_nodes_edges_border.iloc[nearest_indices[1, :]].index)
    
    edges = pd.concat([
        edges.loc[edges.index.isin(edges_border.index) == False],
        new_edges_border
    ])

    nodes = pd.concat([
        nodes.loc[nodes.index.isin(nodes_edges_border.index) == False],
        new_nodes_edges_border
    ])

    edges_help = new_edges_border.reset_index()
    edges_border_out = new_edges_border.loc[list(
                    edges_help['u'].isin(iso_nodes.index) & edges_help['v'].isin(node_ids)
                )]

    edges_border = pd.concat([edges_border_out,
            new_edges_border.loc[list(
                            edges_help['v'].isin(edges_border_out.reset_index()['u']) & edges_help['u'].isin(edges_border_out.reset_index()['v'])
                        )]
        ])
    edges_border = edges_border[~edges_border.index.duplicated(keep='first')]

    if undirected:
        edges_border_out = edges_border

    if len(iso_edges) > 0: 
        iso_edges = edges.loc[list((edges.reset_index()['u'].isin(list(iso_edges.reset_index()['u'])) & edges.reset_index()['v'].isin(list(iso_edges.reset_index()['v']))) | 
            (edges.reset_index()['v'].isin(list(iso_edges.reset_index()['u'])) & edges.reset_index()['u'].isin(list(iso_edges.reset_index()['v']))))]

        iso_edges = pd.concat([
            iso_edges,
            edges.loc[edges.index.isin(edges_border.index)] 
            ]).drop_duplicates()
    else:
        iso_edges = edges.loc[edges.index.isin(edges_border.index)]

    iso_nodes = nodes.loc[nodes.index.isin(iso_edges.reset_index()['u']) | nodes.index.isin(iso_edges.reset_index()['v'])].drop_duplicates()

    nodes = nodes.drop(columns=['total_border_len','border_dist','border_dist_u','border_dist_v','edge_id'],errors='ignore')
    edges = edges.drop(columns=['total_border_len','border_dist','border_dist_u','border_dist_v','edge_id'],errors='ignore')
    iso_nodes = iso_nodes.drop(columns=['total_border_len','border_dist','border_dist_u','border_dist_v','edge_id'],errors='ignore')
    iso_edges = iso_edges.drop(columns=['total_border_len','border_dist','border_dist_u','border_dist_v','edge_id'],errors='ignore')
    
    return nodes, edges, iso_nodes, iso_edges
    # dist_to_center is lost in iso_nodes

def isochrone_graph(G,n=None,radius:float=None,undirected:bool=False,distance:str='length',min_dist:float=5):
    iso_nodes, iso_edges = isochrone_gdfs(G=G,n=n,radius=radius,undirected=undirected,distance=distance)
    nodes, edges = ox.graph_to_gdfs(G)
    _,_,iso_nodes,iso_edges = isochrone_exact_boundary_gdfs(iso_nodes=iso_nodes,iso_edges=iso_edges,nodes=nodes,edges=edges, 
                                                             radius=radius,undirected=undirected,min_dist=min_dist)
    return ox.graph_from_gdfs(iso_nodes,iso_edges)


def isochrone_to_polygon(G,max_segment_len:float=25,buffer:float=5,alpha:float=0.8,remove_small_objects:float=10,remove_small_holes:float=50):
    from . import utils
    geom = ox.graph_to_gdfs(G,nodes=False)
    poly = utils.alpha_shape(geom,max_segment_len=max_segment_len,buffer=buffer,alpha=alpha)
    poly = utils.remove_small_objects_and_holes(poly,remove_small_objects=remove_small_objects,remove_small_holes=remove_small_holes)
    return poly

def save_graph(path,G):
    ox.io.save_graphml(G,path)