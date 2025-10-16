import geopandas as gpd
import pandas as pd
import requests
from datetime import datetime, date
import tempfile
import os
import numpy as np
import shapely
import pycountry
from difflib import get_close_matches
import warnings 
import numpy as np
import geopandas as gpd
import warnings
from tqdm import tqdm
from . import raster_utils
import copy

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


def get_country_region(lat, lon, code_format="alpha_2",get_region:bool=True):
    """
    Reverse geocode lat/lon to country and subdivision.

    Parameters
    ----------
    lat, lon : float
        Coordinates
    code_format : str, optional
        Which country code to return: "alpha_2", "alpha_3", "numeric", or "name".
        Default = "alpha_2".
    """
    url = "https://nominatim.openstreetmap.org/reverse"
    headers = {"User-Agent": "pyGTFSHandler/0.1.0 (https://blogs.upm.es/aga/en/)"}
    params = {"lat": lat, "lon": lon, "format": "json", "zoom": 10, "addressdetails": 1}

    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json().get("address", {})

    # Start with ISO2 code from Nominatim
    country_code = data.get("country_code", "").upper()
    region_name = data.get("state") or data.get("region")
    subdivision_code = None

    # Convert to requested format
    if country_code:
        country = pycountry.countries.get(alpha_2=country_code)
        if country:
            if code_format == "alpha_3":
                country_code = country.alpha_3
            elif code_format == "numeric":
                country_code = country.numeric
            elif code_format == "name":
                country_code = country.name
            else:  # default is alpha_2
                country_code = country.alpha_2

    if not get_region:
        return country_code 
    
    if country_code and region_name:
        try:
            subdivisions = list(pycountry.subdivisions.get(country_code=data.get("country_code", "").upper()))
            subdivision_names = [subdiv.name for subdiv in subdivisions]

            # Exact match (case-insensitive)
            for subdiv in subdivisions:
                if subdiv.name.lower() == region_name.lower():
                    subdivision_code = subdiv.code
                    break

            # If no exact match, try fuzzy matching on subdivision names
            if subdivision_code is None:
                close_matches = get_close_matches(region_name, subdivision_names, n=3, cutoff=0.6)
                if close_matches:
                    for match_name in close_matches:
                        for subdiv in subdivisions:
                            if subdiv.name == match_name:
                                subdivision_code = subdiv.code
                                break
                        if subdivision_code:
                            break
                    if subdivision_code:
                        warnings.warn(
                            f"Fuzzy match used for region '{region_name}'. Matched with '{match_name}'.",
                            UserWarning,
                        )

            # fallback: pycountry's search_fuzzy
            if subdivision_code is None:
                fuzzy_matches = pycountry.subdivisions.search_fuzzy(region_name)
                for match in fuzzy_matches:
                    if match.country_code == data.get("country_code", "").upper():
                        subdivision_code = match.code
                        warnings.warn(
                            f"Fuzzy match used via pycountry.search_fuzzy for region '{region_name}'. Matched with '{match.name}'.",
                            UserWarning,
                        )
                        break

        except LookupError:
            subdivision_code = None

    return country_code, subdivision_code


def download_worldpop_population(
    aoi: gpd.GeoDataFrame, 
    date: date|datetime|int, 
    folder:str=None, 
    overwrite:bool=False, 
    resolution:str="100m", 
    dataset:str='pop', 
    subset:str="wpgpunadj", 
    chunk_size:int=1048576
) -> str:
    
    """
    Download WorldPop population raster for a given AOI and year.

    Parameters
    ----------
    aoi : geopandas.GeoDataFrame
        GeoDataFrame containing the AOI polygon (must have a CRS set, ideally EPSG:4326).
    date : datetime|date
        Date object; only the year is used to query WorldPop.
    out_dir : str, optional
        Directory to save the downloaded file. If None, a temporary directory is used.
    min_pop : float, optional
        Minimum population per cell to be valid
    resolution : str, optional 
        Resolution of the worlpop dataset "100m" or "1km"
    Returns
    -------
    filepath : str
        Path to the downloaded GeoTIFF file.

    All population datasets available are here https://hub.worldpop.org/rest/data/
    """

    if isinstance(date,int):
        date = datetime(year=date,month=1,day=1)

    if folder == "":
        folder = os.getcwd()

    # Ensure CRS is WGS84
    if aoi.crs is None:
        raise ValueError("AOI GeoDataFrame must have a CRS defined (e.g., EPSG:4326).")

    aoi = aoi.to_crs(epsg=4326)

    # Use centroid to get country ISO3
    centroid = aoi.union_all().centroid
    lon, lat = centroid.x, centroid.y

    country_code = get_country_region(lat, lon, code_format='alpha_3', get_region=False)
    if not country_code:
        raise ValueError("Could not resolve country ISO3 code from AOI centroid.")

    iso3 = country_code.upper()
    if date is datetime:
        date = date.date 
        
    year = date.year

    url = None 
    if (dataset == "pop"):
        if year < 2015:
            if (resolution == "100m") and (subset is None):
                subset = "wpgpunadj"
            elif (resolution == "1km") and (subset is None):
                subset = "wpicuadj1km"

            url = f"https://hub.worldpop.org/rest/data/{dataset}/{subset}?iso3={iso3}"

        elif 2015 <= year <= 2030:
            if resolution is None:
                resolution = "100m"
            subset = f"G2_CN_POP_R25A_{resolution}"
            url = f"https://hub.worldpop.org/rest/data/pop/{subset}?iso3={iso3}"

        else:
            raise ValueError(f"No WorldPop dataset available for year {year}")
        
    elif (dataset == "age_structures"):
        if resolution is None:
            resolution = "100m"

        if subset is None:
            subset = f"G2_CN_Age_2024_{resolution}"

        if (subset == "under_18") or (subset == "U18"):
            subset = f"G2_Age_U18_R25A_{resolution}" 

        url = f"https://hub.worldpop.org/rest/data/{dataset}/{subset}?iso3={iso3}"

    r = requests.get(url)
    r.raise_for_status()
    data = r.json()["data"]

    # Find dataset for the requested year
    dataset = next((d for d in data if str(d.get("popyear")) == str(year)), None)
    if dataset is None:
        raise ValueError(f"No WorldPop population dataset available for {iso3} in {year}.")

    # Get download link (first file URL)
    file_url = dataset["files"][0]

    if folder is None:
        # Prepare output path
        out_dir = tempfile.gettempdir()
        raster_path = os.path.join(out_dir, os.path.basename(file_url))
        base_path, fname = os.path.split(raster_path)
        name, _ = os.path.splitext(fname)
        raster_path = os.path.join(base_path,name) + ".tif"
    else:
        os.makedirs(folder,exist_ok=True)
        raster_path = os.path.join(folder, os.path.basename(file_url))

    if os.path.isfile(raster_path) and (not overwrite):
        print(f"Raster population path {raster_path} exists. Skipping download...")

    else:
        # Download file
        with requests.get(file_url, stream=True) as rfile:
            rfile.raise_for_status()
            total_size = int(rfile.headers.get('content-length', 0))
    
            with open(raster_path, "wb") as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc="Downloading",
                ncols=100
            ) as pbar:
                for chunk in rfile.iter_content(chunk_size=chunk_size):
                    if chunk:  # skip keep-alive chunks
                        f.write(chunk)
                        pbar.update(len(chunk))

    return raster_path

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


def density(population_data:str|gpd.GeoDataFrame|np.ndarray,aoi=None,buffer:float=0,resolution:float=None, population_column:str=None, min_value:float=0, transform=None, crs=None, return_raster:bool=True):
    if isinstance(population_data,np.ndarray):
        if (transform is None) or (crs is None):
            raise Exception("If provinding raster array transform and crs are required")
        
        raster = copy.copy(population_data)
            
        raster[np.isnan(raster)] = 0 
        raster[raster < 0] = 0 
    else:
        if isinstance(population_data,str):
            population_data_path = population_data 
        else:
            if population_column is None:
                raise Exception("If population_data is a DataFrame the arg population_column is required")
            
            population_data = population_data[~population_data[population_column].isna()]
            population_data = population_data[~population_data[population_column].isnull()]
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
    
            
    new_raster = raster_utils.buffer_mean(raster_utm,transform_utm,buffer=buffer)
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
        resolution=None,
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
