"""Offer/demand level-of-service rasters for accessibility analysis.

Kept raster-based (not migrated to Polars): population and service-coverage
data here is naturally gridded, and ``rasterio`` is the correct tool for
reading, reprojecting, and smoothing grids. This module is the demand side
of an offer/demand accessibility comparison -- :func:`level_of_service`
combines a rasterized "offer" layer (e.g. computed from
:mod:`UrbanAccessAnalyzer.isochrones` access scores) with a smoothed
population-density "demand" layer to produce a per-pixel
``difference = demand - offer`` raster.

WorldPop raster download and country/subdivision resolution
(previously ``download_worldpop_population`` and ``get_country_region``
in this module) have moved to
:mod:`pycensus.countries.worldwide.worldpop` -- install the ``census`` extra
(``pip install urbanaccessanalyzer[census]``) to use them. They are
re-exported (lazily) from this module for backwards compatibility.
"""

import copy
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from . import raster_ops as raster_utils

try:
    from pycensus.countries.worldwide.worldpop import download_worldpop_population, get_country_region
except ImportError:

    def download_worldpop_population(*args, **kwargs):
        """Raise ``ImportError`` because the ``census`` extra is not installed.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Raises:
            ImportError: Always -- install ``urbanaccessanalyzer[census]``
                (which pulls in ``pycensus``) to use this function.
        """
        raise ImportError(
            "WorldPop functionality requires the 'census' extra: pip install urbanaccessanalyzer[census]"
        )

    def get_country_region(*args, **kwargs):
        """Raise ``ImportError`` because the ``census`` extra is not installed.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Raises:
            ImportError: Always -- install ``urbanaccessanalyzer[census]``
                (which pulls in ``pycensus``) to use this function.
        """
        raise ImportError(
            "WorldPop functionality requires the 'census' extra: pip install urbanaccessanalyzer[census]"
        )

def ls_str_to_int(arr,ref_list):
    # Create a mapping dict from value -> index
    mapping = {v: i for i, v in enumerate(ref_list)}

    # Vectorized mapping using np.vectorize
    map_func = np.vectorize(lambda x: mapping.get(x, len(ref_list)))

    result = map_func(arr)
    return result

def ls_int_to_str(arr,ref_list):
    # Make lookup table with extra '' for out-of-list values
    lut = np.array(ref_list + [""])

    # Map values back
    decoded = lut[arr]
    return decoded

def level_of_service_difference(offer,demand,level_of_services):
    difference = ls_str_to_int(demand,level_of_services) - ls_str_to_int(offer,level_of_services)
    return difference


def filter_population_by_streets(streets_gdf,population,street_buffer,aoi=None,transform=None,crs=None,min_population:float=0,scale:bool=True,population_column='population'):
    streets_gdf = streets_gdf.to_crs(streets_gdf.estimate_utm_crs())
    streets_gdf.geometry = streets_gdf.geometry.simplify(street_buffer/2).buffer(street_buffer,resolution=4)
    
    if isinstance(population,str):
        raster, transform, crs = raster_utils.read_raster(population,aoi,nodata=0, projected=False)
    elif isinstance(population,np.ndarray):
        if (transform is None) or (crs is None):
            raise Exception("If providing a population np.ndarray transform and crs are required")
        
        raster = copy.copy(population)
    else:
        if scale:
            total_population = np.nansum(population[population_column])

        population = population.to_crs(streets_gdf.crs)
        population = population[~population[population_column].isna()]
        population = population[population[population_column] > min_population]
        population = population[population.intersects(streets_gdf.union_all())]

        if scale:
            population[population_column] *= total_population / np.sum(population[population_column])

        return population
    
    streets_gdf['value'] = 1
    streets_raster = raster_utils.rasterize(gdf=streets_gdf[['value','geometry']].to_crs(crs),shape=raster,transform=transform,crs=crs,value_column='value',background_value=0)
    
    if scale:
        total_population = np.nansum(raster)

    raster[np.isnan(raster)] = 0
    raster[raster <= min_population] = 0 
    raster[streets_raster == 0] = 0 
    if scale:
        raster *= total_population / np.sum(raster)

    if isinstance(population,str):
        return raster, transform, crs 
    
    return raster


def density(population_data:str|gpd.GeoDataFrame|np.ndarray,aoi=None,buffer:float=0, kernel_shape: str = "disk", resolution:float=None, population_column:str=None, min_value:float=0, transform=None, crs=None, return_raster:bool=True):
    if isinstance(population_data,np.ndarray):
        if (transform is None) or (crs is None):
            raise Exception("If provinding raster array transform and crs are required")
        
        raster = copy.copy(population_data)
            
        raster[np.isnan(raster)] = 0 
        raster[np.isinf(raster)] = 0
        raster[raster < 0] = 0 
    else:
        if isinstance(population_data,str):
            population_data_path = population_data 
        else:
            if population_column is None:
                raise Exception("If population_data is a DataFrame the arg population_column is required")
            
            population_data[population_column] = (
                pd.to_numeric(population_data[population_column], errors="coerce")
                .replace([np.inf, -np.inf], 0)
                .fillna(0)
            )
            population_data = population_data[population_data[population_column] > 0]

            pop_utm = population_data.to_crs(population_data.estimate_utm_crs())
            if buffer == 0:
                return population_data[population_column] / (pop_utm.area / (10**6))
            
            population_data_path = raster_utils.rasterize() 

        raster, transform, crs = raster_utils.read_raster(population_data_path,aoi=aoi,nodata=0, projected=False)
        raster[np.isnan(raster)] = 0 
        raster[raster < 0] = 0 

    if crs.is_projected:
        raster_utm, transform_utm, crs_utm = raster, transform, crs
    else:
        raster_utm, transform_utm, crs_utm = raster_utils.reproject(raster,transform,crs,dst_crs='utm')
    
            
    new_raster = raster_utils.buffer_mean(raster_utm,transform_utm,buffer=buffer,kernel_shape=kernel_shape)
    new_raster, _, _ = raster_utils.reproject(
        new_raster,
        transform_utm,
        crs_utm,
        dst_transform=transform,
        dst_crs=crs,
        height=raster.shape[0],
        width=raster.shape[1]
    )
    

    if return_raster:
        return new_raster
    
    if isinstance(population_data,gpd.GeoDataFrame):
        gdf = raster_utils.sample_at_geometries(new_raster,transform,crs,population_data,aoi=aoi)
    else:
        new_raster *= (raster >= min_value)
        gdf = raster_utils.vectorize(new_raster,transform,crs,aoi=aoi,min_value=min_value)
        gdf = gdf.rename(columns={'value':'pop_density'})

    return gdf
    

def density_matrix_to_processing_order(density_matrix, level_of_services):
    # Melt the DataFrame into long format
    melted = density_matrix.melt(
        id_vars="density", var_name="distance", value_name="ls"
    ).dropna(subset=["ls"])
    
    # Map levels of service to an importance ranking
    importance_map = {ls: i for i, ls in enumerate(level_of_services)}
    melted["importance"] = melted["ls"].map(importance_map)
    
    # Sort by importance (A1 highest → F lowest), then by distance descending
    melted = melted.sort_values(["importance", "distance"], ascending=[False, True])
    
    return melted[["density", "distance", "ls"]]



def level_of_service_raster(
        save_path,
        population,
        offer,
        density_matrix:pd.DataFrame,
        level_of_services:list,
        min_population:float=0,
        polygons=None,
        aoi=None,
        transform=None,
        crs=None,
        resolution=100,
        street_buffer:float=50,
        level_of_service_column:str='level_of_service',
    ):
    if isinstance(population,str):
        pop_raster, transform, crs = raster_utils.read_raster(population,aoi=aoi,nodata=0)
    elif isinstance(population,np.ndarray):
        if (transform is None) or (crs is None):
            raise Exception("If providing a population np.ndarray transform and crs are required")
        
        pop_raster = copy.copy(population)
    else:
        pop_raster, transform, crs = raster_utils.rasterize(population)

    offer = offer.to_crs(offer.estimate_utm_crs())

    geom_types = offer.geometry.geom_type.unique()

    if set(geom_types).issubset({'LineString', 'MultiLineString'}):
        # All geometries are line types
        offer.geometry = offer.geometry.simplify(street_buffer / 2).buffer(street_buffer, resolution=4)

    elif not set(geom_types).issubset({'Polygon', 'MultiPolygon', 'GeometryCollection'}):
        raise ValueError(f"Mixed geometry types (Poygons and Lines) detected for offer geometry: {geom_types}")

    offer_raster = raster_utils.rasterize(
        gdf=offer,
        shape=pop_raster,
        transform=transform,
        crs=crs,
        value_column=level_of_service_column,
        value_order=level_of_services
    )

    population_buffers = np.unique([0,*density_matrix.columns[1:]])
    population_buffers = [int(i) for i in population_buffers]

    density_array = []
    for b in population_buffers:
        density_b = density(pop_raster,buffer=b,min_value=min_population,transform=transform,crs=crs,return_raster=True)
        density_array.append(density_b)
        if b == 0:
            continue
        
        for i in range(len(density_array)):
            density_array[-1] = np.minimum(density_array[-1], density_array[i])

    process_order = density_matrix_to_processing_order(density_matrix,level_of_services)

    demand_raster = np.zeros(density_array[0].shape,dtype='<U2')
    for density_i, distance, ls in process_order[['density', 'distance', 'ls']].itertuples(index=False, name=None):
        index = population_buffers.index(distance)
        demand_raster[density_array[index] > density_i] = ls

    new_pop_raster, new_transform, new_crs = raster_utils.reproject_global(pop_raster,transform,crs,dst_crs=3857,dst_nodata=0,resolution=resolution)
    os.makedirs(save_path,exist_ok=True)

    with rasterio.open(
        os.path.normpath(save_path+"/population.tif"),
        "w",
        driver="GTiff",
        height=new_pop_raster.shape[0],
        width=new_pop_raster.shape[1],
        count=1,
        dtype=new_pop_raster.dtype,
        crs=new_crs,                 # new CRS from reprojection
        transform=new_transform,     # aligned transform
        nodata=0,                    # same as dst_nodata
        compress="lzw"               # optional: makes file smaller
    ) as dst:
        dst.write(new_pop_raster, 1)


    for i in range(len(density_array)):
        b = population_buffers[i]
        new_density_raster, new_transform, new_crs = raster_utils.reproject_global(density_array[i],transform,crs,dst_crs=3857,dst_nodata=0,resolution=resolution)

        with rasterio.open(
            os.path.normpath(save_path+f"/population_density_{b}.tif"),
            "w",
            driver="GTiff",
            height=new_density_raster.shape[0],
            width=new_density_raster.shape[1],
            count=1,
            dtype=new_density_raster.dtype,
            crs=new_crs,                 # new CRS from reprojection
            transform=new_transform,     # aligned transform
            nodata=0,                    # same as dst_nodata
            compress="lzw"               # optional: makes file smaller
        ) as dst:
            dst.write(new_density_raster, 1)

    offer_raster = ls_str_to_int(offer_raster,level_of_services) 
    new_offer_raster, new_transform, new_crs = raster_utils.reproject_global(offer_raster,transform,crs,dst_crs=3857,dst_nodata=len(level_of_services),resolution=resolution)

    with rasterio.open(
        os.path.normpath(save_path+f"/offer.tif"),
        "w",
        driver="GTiff",
        height=new_offer_raster.shape[0],
        width=new_offer_raster.shape[1],
        count=1,
        dtype=new_offer_raster.dtype,
        crs=new_crs,                 # new CRS from reprojection
        transform=new_transform,     # aligned transform
        nodata=len(level_of_services), # same as dst_nodata
        compress="lzw"               # optional: makes file smaller
    ) as dst:
        dst.write(new_offer_raster, 1)

    demand_raster = ls_str_to_int(demand_raster,level_of_services) 
    new_demand_raster, new_transform, new_crs = raster_utils.reproject_global(demand_raster,transform,crs,dst_crs=3857,dst_nodata=len(level_of_services),resolution=resolution)

    with rasterio.open(
        os.path.normpath(save_path+f"/demand.tif"),
        "w",
        driver="GTiff",
        height=new_demand_raster.shape[0],
        width=new_demand_raster.shape[1],
        count=1,
        dtype=new_demand_raster.dtype,
        crs=new_crs,                 # new CRS from reprojection
        transform=new_transform,     # aligned transform
        nodata=len(level_of_services),                    # same as dst_nodata
        compress="lzw"               # optional: makes file smaller
    ) as dst:
        dst.write(new_demand_raster, 1)

    difference = new_demand_raster - new_offer_raster

    with rasterio.open(
        os.path.normpath(save_path+f"/difference.tif"),
        "w",
        driver="GTiff",
        height=difference.shape[0],
        width=difference.shape[1],
        count=1,
        dtype=difference.dtype,
        crs=new_crs,                 # new CRS from reprojection
        transform=new_transform,     # aligned transform
        nodata=None,                    # same as dst_nodata
        compress="lzw"               # optional: makes file smaller
    ) as dst:
        dst.write(difference, 1)

    return None





def level_of_service(
        population,
        offer,
        density_matrix:pd.DataFrame,
        level_of_services:list,
        min_population:float=0,
        polygons=None,
        aoi=None,
        transform=None,
        crs=None,
        resolution=100,
        street_buffer:float=50,
        level_of_service_column:str='level_of_service',
    ):
    if isinstance(population,str):
        pop_raster, transform, crs = raster_utils.read_raster(population,aoi=aoi,nodata=0)
    elif isinstance(population,np.ndarray):
        if (transform is None) or (crs is None):
            raise Exception("If providing a population np.ndarray transform and crs are required")
        
        pop_raster = copy.copy(population)
    else:
        pop_raster, transform, crs = raster_utils.rasterize(population)

    offer = offer.to_crs(offer.estimate_utm_crs())
    offer.geometry = offer.geometry.simplify(street_buffer/2).buffer(street_buffer,resolution=4)
    offer_raster = raster_utils.rasterize(
        gdf=offer,
        shape=pop_raster,
        transform=transform,
        crs=crs,
        value_column=level_of_service_column,
        value_order=level_of_services
    )

    population_buffers = np.unique([0,*density_matrix.columns[1:]])
    population_buffers = [int(i) for i in population_buffers]

    density_array = []
    for b in population_buffers:
        density_b = density(pop_raster,buffer=b,min_value=min_population,transform=transform,crs=crs,return_raster=True)
        density_array.append(density_b)
        if b == 0:
            continue
        
        for i in range(len(density_array)):
            density_array[-1] = np.minimum(density_array[-1], density_array[i])

    process_order = density_matrix_to_processing_order(density_matrix,level_of_services)

    demand_raster = np.zeros(density_array[0].shape,dtype='<U2')
    for density_i, distance, ls in process_order[['density', 'distance', 'ls']].itertuples(index=False, name=None):
        index = population_buffers.index(distance)
        demand_raster[density_array[index] > density_i] = ls

    if polygons is not None:
        gdf = raster_utils.sample_at_geometries(polygons,pop_raster,transform,crs,keep_nodata=True,nodata=0,min_value=min_population)
        gdf = gdf.rename(columns={'value':'population'}) 
    else: 
        # Not the most efficient way as many None level of service cells have to be created 
        gdf = raster_utils.vectorize(pop_raster,transform,crs,keep_nodata=True,nodata=0,min_value=min_population)
        gdf = gdf.rename(columns={'value':'population'})

    gdf['population'] = gdf['population'].astype(float).fillna(0)
    for i in range(len(population_buffers)):
        gdf[f'pop_density_{population_buffers[i]}'] = density_array[i].flatten()

    difference = ls_str_to_int(demand_raster,level_of_services) - ls_str_to_int(offer_raster,level_of_services)

    gdf['level_of_service_offer'] = offer_raster.flatten()
    gdf['level_of_service_demand'] = demand_raster.flatten()
    gdf['level_of_service_difference'] = difference.flatten()

    gdf = gdf[['id','population', *gdf.columns[3:],'geometry']]
    gdf = gdf[gdf['population'] > min_population].reset_index(drop=True)

    return gdf 
