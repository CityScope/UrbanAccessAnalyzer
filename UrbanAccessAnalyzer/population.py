import geopandas as gpd 
import pandas as pd

def read_estat(path,bounds):
    from . import utils
    # Loading the population grid from eurostat
    rio, meta = utils.raster_crop(path,bounds)
    pop_df = utils.raster_to_gdf(rio,meta,bounds)
    pop_df["population"] = pop_df["value"].astype(int)
    return pop_df.to_crs(bounds.crs)

def read_ine(data,geo,bounds=None):
    from . import utils
    ine_geo = utils.gdf_from_file(geo,bounds=bounds)
    ine_data = pd.read_csv(data,sep=";",dtype={"Total":int},thousands=r'.')
    ine_data = ine_data.dropna().reset_index(drop=True)
    ine_data["CUSEC"] = [i[0:10] for i in ine_data["Secciones"]]
    ine_data_total = ine_data[(ine_data['Edad (grupos quinquenales)'] == 'Total') & (ine_data['Sexo'] == 'Ambos sexos')]
    ine_geo = ine_geo.merge(ine_data_total[['CUSEC','Total']], on='CUSEC', how='left')
    ine_geo["population"] = ine_geo["Total"].astype(int)
    ine_geo = ine_geo.drop(columns=["Total"]).reset_index(drop=True)
    if type(bounds) != type(None):
        ine_geo = ine_geo[bounds.to_crs(ine_geo.crs).union_all().contains(ine_geo.centroid)].reset_index(drop=True)
        ine_geo = ine_geo.to_crs(bounds.crs)

    ine_geo = ine_geo[ine_geo["population"] > 0].reset_index(drop=True)

    return ine_geo

def mix_ine_and_estat(ine,estat):
    import copy
    import pandas as pd
    import geopandas as gpd
    ine = ine.to_crs(ine.geometry.estimate_utm_crs())
    estat = estat.to_crs(ine.crs)
    res = copy.deepcopy(ine)
    res['partition'] = 0
    for i in range(len(ine)):
        if ine.geometry[i].area > 1000**2 or ine.geometry[i].boundary.length > 1000*4: 
            max_pop = ine['population'][i]
            inter = gpd.GeoDataFrame(geometry=estat.intersection(ine.geometry[i]).geometry)
            inter['pop'] = list(estat['population'] * inter.area / (1000 ** 2))
            inter.loc[inter['pop'] > max_pop, 'pop'] = max_pop
            inter = inter[inter['pop'] > 0].reset_index(drop=True)

            if len(inter) > 0:
                res.loc[i,"population"] = int(inter['pop'][0])
                res.loc[i,"geometry"] = inter.geometry[0]
                if len(inter) > 1:
                    for j in range(1,len(inter)):
                        val = copy.deepcopy(ine.loc[i:i]).reset_index(drop=True)
                        val.loc[0,'partition'] = j 
                        val.loc[0,'population'] = int(copy.deepcopy(inter['pop'][j]))
                        val.loc[0,"geometry"] = copy.deepcopy(inter.geometry[j])
                        res = pd.concat([res,val.loc[0:0]]).reset_index(drop=True) 

    return res

def add_pop_density(pop_df:gpd.GeoDataFrame,buffer:float=0):
    from . import utils
    import numpy as np
    crs = pop_df.crs

    new_pop_df = pop_df.copy()

    new_pop_df = new_pop_df.loc[new_pop_df.geometry.is_valid].reset_index(drop=True)
    if crs.is_projected == False:
        new_pop_df = new_pop_df.to_crs(new_pop_df.geometry.estimate_utm_crs())
    
    new_pop_df['pop_density'] = new_pop_df['population'] / (new_pop_df.geometry.area / (1000**2))

    if buffer > 0:
        pop_df_simple = new_pop_df.geometry.simplify(buffer/20)

        center = new_pop_df.geometry.centroid
        center = center.buffer(buffer,resolution=4)

        inter = utils.intersection_all_with_all(center.geometry,pop_df_simple.geometry)
        inter = np.vectorize(lambda p: p.area)(inter)#getattr(p, 'area', 0))(inter)
        inter = np.divide(
            inter, 
            np.array(pop_df_simple.area)[np.newaxis,:]
        )

        inter *= np.array(new_pop_df['population'])[np.newaxis,:]

        new_pop_df['pop_density_buffer'] = np.sum(inter, axis=1)
        new_pop_df['pop_density_buffer'] = new_pop_df['pop_density_buffer'] / (center.area / (1000**2))
        new_pop_df['pop_density_tile'] = new_pop_df['pop_density'].copy()
        new_pop_df.loc[new_pop_df['pop_density_buffer'] < new_pop_df['pop_density'],'pop_density'] = new_pop_df.loc[new_pop_df['pop_density_buffer'] < new_pop_df['pop_density'],'pop_density_buffer']
        
    return new_pop_df.to_crs(crs)



def add_accessibility(pop_df,accessibility_graph,max_dist:float=500):
    import osm
    import osmnx as ox
    import shapely
    new_pop_df = pop_df.copy()
    if type(accessibility_graph) == gpd.GeoDataFrame:
        new_pop_df['accessibility'] = 999
        geom = list(pop_df.geometry.centroid)
        shapely.prepare(geom) 
        unique_accessibility = list(np.unique(accessibility_graph['accessibility']))
        unique_accessibility.reverse()
        for ls in unique_accessibility:
            ls_i_geometry = accessibility_graph.loc[accessibility_graph['accessibility'] == ls].geometry.union_all()
            inter = shapely.intersects(geom,ls_i_geometry)
            new_pop_df.loc[inter,'accessibility'] = ls

    else:
        pop_ids, edge_ids = osm.nearest_edges(pop_df.geometry.centroid,accessibility_graph,max_dist=max_dist)
        edges = ox.graph_to_gdfs(accessibility_graph,nodes=False)
        edges = edges.loc[edge_ids] 
        edges['pop_ids'] = pop_ids
        edges = edges[['pop_ids','accessibility']]#.groupby('pop_ids').agg('min')
        edges = edges.set_index(['pop_ids'])
        ls = edges['accessibility'].sort_values()
        ls = ls[~ls.index.duplicated(keep='first')]
        new_pop_df.loc[ls.index,'accessibility'] = list(ls)
        new_pop_df['accessibility'] = pd.to_numeric(new_pop_df['accessibility'], errors = 'coerce')
        new_pop_df['accessibility'] = new_pop_df['accessibility'].astype('Int16',errors='raise')
        new_pop_df.loc[new_pop_df['accessibility'].isna(),'accessibility'] = 999

    return new_pop_df

def minimum_accessibility(pop_df,min_accessibility:list,min_density:list):
    min_accessibility = np.sort(min_accessibility)[::-1]
    min_density = np.sort(min_density)
    if len(min_accessibility) != len(min_density):
        raise Exception(f"Length of min_accessibility is {len(min_accessibility)} but length of min_density is {len(min_density)}.")
    
    new_pop_df = pop_df.copy()
    new_pop_df['min_accessibility'] = 999 
    for i in range(len(min_accessibility)):
        ls_i = min_accessibility[i]
        dens_i = min_density[i]
        new_pop_df.loc[new_pop_df['pop_density'] > dens_i,'min_accessibility'] = ls_i

    new_pop_df['min_accessibility'] = new_pop_df['min_accessibility'].astype(int)

    if 'accessibility' in new_pop_df.columns:
        new_pop_df['target_achieved'] = new_pop_df['min_accessibility'] >= new_pop_df['accessibility']

    return new_pop_df

    
