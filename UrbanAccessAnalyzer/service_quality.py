import geopandas as gpd
import pandas as pd
import numpy as np
import warnings

def service_type(service_gdf:gpd.GeoDataFrame,row_values:list=None,columns:list=None):
    """
    The first values of row_values and columns lists are give priority if on a row of the service dataframe there are more than one possible value.
    If row_values and columns are given the order of columns does not matter.
    """
    new_service_gdf = service_gdf.copy()

    if type(columns) == list: 
        new_service_gdf = new_service_gdf[columns + ['geometry']]
            
    if type(row_values) == list:
        new_service_gdf[new_service_gdf[list(new_service_gdf.columns.drop('geometry'))].isin(row_values)==False] = None
        new_service_gdf['service_type'] = None#list(new_service_gdf[new_service_gdf.keys()[0]])
        for i in reversed(row_values):
            rows = np.sum(np.array(new_service_gdf == i),axis=1)
            new_service_gdf.loc[rows == 1, 'service_type'] = i
    else: 
        new_service_gdf['service_type'] = None
        for i in reversed(list(new_service_gdf.columns.drop(['geometry','service_type']))):
            rows = new_service_gdf[i].isna() == False
            new_service_gdf.loc[rows, 'service_type'] = new_service_gdf.loc[rows,i] 


    new_service_gdf = new_service_gdf[new_service_gdf['service_type'].isna() == False].reset_index(drop=True)
    return new_service_gdf[['service_type','geometry']] 

def merge_and_simplify_geometry(service_gdf:gpd.GeoDataFrame,buffer:float=5,min_width:float=10,simplify:float=1,min_area:float=0):
    import shapely
    from . import utils
    new_service_gdf = service_gdf.copy()
    orig_crs = service_gdf.crs
    if orig_crs.is_projected == False:
        new_service_gdf = new_service_gdf.to_crs(new_service_gdf.geometry.estimate_utm_crs())

    if min_area > 0:
        new_service_gdf = new_service_gdf[new_service_gdf.geometry.area > min_area]

    if 'service_type' not in service_gdf.columns:
        warnings.warn("Calling .service_type() before merging geometries as no service_type column was found.")
        new_service_gdf = service_type(new_service_gdf)

    if simplify > 0:
        new_service_gdf.loc[:,'geometry'] = new_service_gdf.geometry.simplify(simplify)
    
    new_service_gdf['orig_area'] = new_service_gdf.geometry.area

    geoms = new_service_gdf.geometry.union_all().buffer(buffer,resolution=4,cap_style='square',join_style='mitre')
    geoms = geoms.buffer(-min_width-buffer,resolution=4,cap_style='square',join_style='mitre')
    geoms = shapely.get_parts(geoms)

    service_types = []
    geometries = []
    inter = utils.intersects_all_with_all(new_service_gdf.geometry.centroid,gpd.GeoSeries(geoms,crs=new_service_gdf.crs))
    for i in range(len(geoms)):
        inter_df = new_service_gdf.loc[inter[:,i],['orig_area','service_type']]
        if len(inter_df) == 0:
            continue

        geometries.append(geoms[i])
        st = list(inter_df.loc[inter_df['orig_area'] == max(inter_df['orig_area']), 'service_type'])[0]
        service_types.append(st) ### What if there are two maximum areas? We take the first without taking row_values into account. 

    new_service_gdf = gpd.GeoDataFrame({'service_type':service_types},geometry=geometries,crs=new_service_gdf.crs)
    new_service_gdf.loc[:,'geometry'] = new_service_gdf.geometry.buffer(min_width,resolution=4,cap_style='square',join_style='mitre')
    return new_service_gdf.to_crs(orig_crs)

def by_row_values(service_gdf:gpd.GeoDataFrame,row_values:list,quality_values:list,columns:list = None,max_class:int=None):
    if (len(row_values) != len(quality_values)):
        raise Exception(f"Length of row_values is {len(row_values)} and quality_values {len(quality_values)} but they shpuld be equal.")
    
    
    if min(quality_values) < 1:
        raise Exception("The minimum accepted class_value (best class) should be 1")
    
    if 'service_type' not in service_gdf.columns: 
        new_service_gdf = service_type(service_gdf,row_values=row_values,columns=columns)
    else:
        new_service_gdf = service_gdf.copy()

    if 'service_quality' not in new_service_gdf.columns:
        new_service_gdf['service_quality'] = 0
    else:
        new_service_gdf.loc[new_service_gdf['service_quality'] == 0, 'service_quality'] = 999
        new_service_gdf.loc[new_service_gdf['service_quality'] > 999, 'service_quality'] = 999
        new_service_gdf.loc[new_service_gdf['service_quality'] < 999, 'service_quality'] -= 1   

    for i in range(len(row_values)):
        new_service_gdf.loc[new_service_gdf['service_type'] == row_values[i],'service_quality'] += quality_values[i]
    
    new_service_gdf.loc[new_service_gdf['service_type'].isin(row_values) == False,'service_quality'] = 999
    new_service_gdf.loc[new_service_gdf['service_quality'] == 0, 'service_quality'] = 999
    new_service_gdf = new_service_gdf[new_service_gdf['service_quality'] < 999]

    if max_class: 
       new_service_gdf.loc[new_service_gdf['service_quality'] > max_class, 'service_quality'] = max_class     
    
    return new_service_gdf

def by_area(service_gdf:gpd.GeoDataFrame,areas:list,max_class:int=None,ascending:bool=True):
    new_service_gdf = service_gdf.copy()
    orig_crs = service_gdf.crs
    areas = np.sort(areas)
    if 'service_quality' in new_service_gdf.columns:            
        new_service_gdf.loc[new_service_gdf['service_quality'] == 0, 'service_quality'] = 999
        new_service_gdf.loc[new_service_gdf['service_quality'] > 999, 'service_quality'] = 999
        new_service_gdf.loc[new_service_gdf['service_quality'] < 999, 'service_quality'] -= 1   
    else:
        new_service_gdf['service_quality'] = 0

    if orig_crs.is_projected == False:
        new_service_gdf = new_service_gdf.to_crs(new_service_gdf.geometry.estimate_utm_crs())

    new_service_gdf = new_service_gdf[new_service_gdf.geometry.area > min(areas)]

    new_service_gdf['service_quality_temp'] = 999

    for i in range(len(areas)):
        if ascending:
            _class = len(areas) - i
        else:
            _class = i 

        new_service_gdf.loc[new_service_gdf.geometry.area > areas[i],'service_quality_temp'] = _class

    new_service_gdf['service_quality'] += new_service_gdf['service_quality_temp']

    new_service_gdf = new_service_gdf.drop(columns=['service_quality_temp'])
    new_service_gdf.loc[new_service_gdf['service_quality'] == 0, 'service_quality'] = 999
    new_service_gdf = new_service_gdf[new_service_gdf['service_quality'] < 999]

    if max_class: 
       new_service_gdf.loc[new_service_gdf['service_quality'] > max_class, 'service_quality'] = max_class  

    return new_service_gdf.to_crs(orig_crs)