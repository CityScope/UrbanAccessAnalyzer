import geopandas as gpd
import pandas as pd
import numpy as np
import warnings

def build_matrix(n_classes:int,n_distance_steps:int,matrix_columns:list=[],max_accessibility:int = 999):
    accessibility_matrix = []
    for i in range(n_classes): 
        L = [j for j in range(i,i+n_distance_steps)]
        if i == 0:
            L[0] = 1 

        accessibility_matrix.append(L)

    accessibility_matrix = np.array(accessibility_matrix, ndmin=2)
    for i in range(len(matrix_columns)):
        if type(matrix_columns[i]) == type(None):
            continue 
        elif type(matrix_columns[i]) != list:
            raise Exception("Please provide matrix_columns in the following format [[1,1,2,3],None,[2,3,3,4]].")
        elif len(matrix_columns[i]) != n_classes:
            raise Exception(f"Length of matrix_column {i} is {len(matrix_columns[i])} but it should be n_classes {n_classes}.")
        
        accessibility_matrix[:,i] = np.array(matrix_columns[i])[np.newaxis,:]#np.transpose(np.array(matrix_columns[i])[np.newaxis,:])

    accessibility_matrix[accessibility_matrix > max_accessibility] = max_accessibility 

    return accessibility_matrix

def areas(service_gdf:gpd.GeoDataFrame,accessibility_matrix=None,dist_steps:list=[300,500,750,1000,1250],max_accessibility:int = None,undirected:bool=False,_print:bool=True):
    new_service_gdf = service_gdf.copy().to_crs(service_gdf.geometry.estimate_utm_crs())
    new_service_gdf['service_quality'] = new_service_gdf['service_quality'].round().astype(int)
    new_service_gdf.loc[new_service_gdf['service_quality'] < 1,'service_quality'] = 1
    classes = np.array(range(1,int(max(new_service_gdf['service_quality']))+1))

    if type(accessibility_matrix) == type(None):
        accessibility_matrix = []
        for i in range(len(classes)): 
            L = [j for j in range(i,i+len(dist_steps))]
            if i == 0:
                L[0] = 1 

            accessibility_matrix.append(L)

        accessibility_matrix = np.matrix(accessibility_matrix)
    elif len(list(range(accessibility_matrix.shape[0]))) >= len(classes):
        None
    else:
        raise Exception(f"The service_gdf has the maximum class {max(classes)} but your accessibility_matrix has the maximum class {accessibility_matrix.shape[0]}.")
    
    if accessibility_matrix.shape[1] < len(dist_steps): 
        raise Exception(f"Your accessibility_matrix has {accessibility_matrix.shape[1]} but you provided {len(dist_steps)} distance steps which is more.")
    
    #accessibility_matrix = np.matrix(accessibility_matrix)
    accessibility_matrix = accessibility_matrix[0:len(classes),0:len(dist_steps)]

    if max_accessibility:
        accessibility_matrix.loc[accessibility_matrix > max_accessibility] = max_accessibility 

    np_dist_steps = np.array(dist_steps)

    LSpolygons = []
    for i in list(np.unique(np.asarray(accessibility_matrix))):
        if _print:
            print(f"accessibility quality class {i} completed")
        
        for j in range(len(classes)): 
            d = np_dist_steps[np.squeeze(np.asarray(accessibility_matrix[j,:] == i))]
            if len(d) == 0:
                continue 

            n = new_service_gdf.loc[new_service_gdf['service_quality'] == classes[j]].geometry
            if len(n) == 0:
                continue 
            
            LSpolygons.append(gpd.GeoDataFrame({'accessibility':[i]},geometry=[n.buffer(max(d)).union_all()],crs=new_service_gdf.crs))
    
    LSpolygons = pd.concat(LSpolygons,ignore_index=True)
    LSpolygons = LSpolygons.reset_index(drop=True)
    return LSpolygons.dissolve(by='accessibility').reset_index().sort_values(by='accessibility',ascending=False).reset_index(drop=True)     

def graph(service_gdf:gpd.GeoDataFrame,G,accessibility_matrix=None,dist_steps:list=[300,500,750,1000,1250],max_accessibility:int = None,undirected:bool=False,_print:bool=True):
    from . import graph_processing
    import osmnx as ox

    new_service_gdf = service_gdf.copy()
    G_copy = G.copy()
    if undirected:
        G_copy = ox.convert.to_undirected(G_copy)

    _nodes, _edges = ox.graph_to_gdfs(G_copy)
    _nodes['accessibility'] = 999 
    _edges['accessibility'] = 999 
    G_copy = ox.graph_from_gdfs(_nodes,_edges,graph_attrs=G_copy.graph)
    if undirected:
        G_copy = ox.convert.to_undirected(G_copy)

    _edge_union = gpd.GeoSeries(_edges.union_all(),crs=_edges.crs)

    new_service_gdf['service_quality'] = new_service_gdf['service_quality'].astype(int)
    classes = range(1,max(new_service_gdf['service_quality'])+1)

    if type(accessibility_matrix) == type(None):
        accessibility_matrix = []
        for i in range(len(classes)): 
            L = [j for j in range(i,i+len(dist_steps))]
            if i == 0:
                L[0] = 1 

            accessibility_matrix.append(L)

        accessibility_matrix = np.array(accessibility_matrix)
    elif len(list(range(accessibility_matrix.shape[0]))) >= len(classes):
        None
    else:
        raise Exception(f"The service_gdf has the maximum class {max(classes)} but your accessibility_matrix has the maximum class {accessibility_matrix.shape[0]}.")
    
    if accessibility_matrix.shape[1] < len(dist_steps): 
        raise Exception(f"Your LSmstrix has {accessibility_matrix.shape[1]} but you provided {len(dist_steps)} distance steps which is more.")
    
    #accessibility_matrix = np.matrix(accessibility_matrix)
    accessibility_matrix = accessibility_matrix[0:len(classes),0:len(dist_steps)]

    if max_accessibility:
        accessibility_matrix[accessibility_matrix > max_accessibility] = max_accessibility 

    np_dist_steps = np.array(dist_steps)

    node_id_and_class = [] 
    for i in classes: 
        service_gdf_i = new_service_gdf.loc[new_service_gdf['service_quality'] == i]
        service_gdf_i = service_gdf_i.reset_index(drop=True)
        if len(service_gdf_i) == 0:
            node_id_and_class.append([])
            continue 

        service_gdf_i, G_copy = graph_processing.geometries_to_nodes(service_gdf_i,G_copy,min_dist=min(dist_steps)/10,max_dist=min(dist_steps)/2,edges=_edge_union)
        service_gdf_i = service_gdf_i.loc[service_gdf_i['osmid'].isna() == False]
        service_gdf_i = service_gdf_i.reset_index(drop=True)
        if len(service_gdf_i) == 0:
            node_id_and_class.append([])
            continue 

        node_id_and_class.append(list(np.unique(service_gdf_i['osmid'])))
        if undirected:
            G_copy = ox.convert.to_undirected(G_copy)

    for i in list(np.unique(np.asarray(accessibility_matrix))):
        if _print:
            print(f"accessibility quality class {i} completed")
        
        for j in range(len(classes)): 
            d = np_dist_steps[np.squeeze(np.asarray(accessibility_matrix[j,:] == i))]
            if len(d) == 0:
                continue 

            n = node_id_and_class[j] 
            if len(n) == 0:
                continue 

            all_nodes, all_edges = ox.graph_to_gdfs(G_copy)
            iso_nodes, iso_edges = graph_processing.isochrone_gdfs(G_copy,n,radius=max(d),distance='length',undirected=undirected,center=True)
            orig_nodes = all_nodes.loc[n,:]
            orig_nodes['dist_to_center'] = 0.
            if len(iso_nodes) > 0:
                iso_nodes = pd.concat([iso_nodes,orig_nodes])
                iso_nodes = iso_nodes[~iso_nodes.index.duplicated(keep='first')]
            else:
                iso_nodes = orig_nodes
        
            if len(iso_nodes) == 0:
                continue

            iso_edges = iso_edges.loc[iso_edges['accessibility'] > i]

            if len(iso_nodes) > 0:
                all_nodes,all_edges,iso_nodes,iso_edges = graph_processing.isochrone_exact_boundary_gdfs(nodes=all_nodes,edges=all_edges,iso_nodes=iso_nodes,iso_edges=iso_edges,
                                                                                            radius=max(d),undirected=undirected,min_dist=min(dist_steps)/10,accessibility=i)
                
                all_nodes.loc[all_nodes.index.isin(iso_nodes.index) & (all_nodes['accessibility'] > i),'accessibility'] = i

            if len(iso_edges) > 0:
                all_edges.loc[all_edges.index.isin(iso_edges.index) & (all_edges['accessibility'] > i),'accessibility'] = i 

            #all_edges.loc[all_edges['accessibility'] == np.str_('nan'),'accessibility'] = 999
            #all_nodes.loc[all_nodes['accessibility'] == np.str_('nan'),'accessibility'] = 999

            if (len(iso_edges) > 0) or (len(iso_nodes) > 0):
                G_copy = ox.graph_from_gdfs(all_nodes,all_edges,graph_attrs=G_copy.graph)

            if undirected:
                G_copy = ox.convert.to_undirected(G_copy)
        
    return G_copy 


def graph_to_areas(G,max_segment_len:float=25,buffer:float=5,alpha:float=0.8,remove_small_objects:float=10,remove_small_holes:float=50):
    import osmnx as ox
    from . import utils
    geom = ox.graph_to_gdfs(G,nodes=False)
    geom['accessibility'] = geom['accessibility'].round().astype(int)
    geom.loc[geom['accessibility'] < 1,'accessibility'] = 1
    LSid = np.unique(geom['accessibility']) 
    result = []
    for i in LSid:
        geom_i = geom.loc[geom['accessibility'] == i]
        poly_i = utils.alpha_shape(geom_i,max_segment_len=max_segment_len,buffer=buffer,alpha=alpha)
        poly_i = utils.remove_small_objects_and_holes(poly_i,remove_small_objects=remove_small_objects,remove_small_holes=remove_small_holes)
        result.append(poly_i)

    result = gpd.GeoDataFrame({'accessibility':LSid},geometry=result,crs=geom.crs)
    return result




    