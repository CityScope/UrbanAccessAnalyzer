import UrbanAccessAnalyzer.configs as configs
import UrbanAccessAnalyzer.graph_processing as graph_processing
import UrbanAccessAnalyzer.isochrones as isochrones
import UrbanAccessAnalyzer.h3_utils as h3_utils
import UrbanAccessAnalyzer.poi_utils as poi_utils
import UrbanAccessAnalyzer.population as population
import UrbanAccessAnalyzer.api_keys as api_keys
import UrbanAccessAnalyzer.geometry_utils as geometry_utils
import UrbanAccessAnalyzer.quality as quality_utils
from UrbanAccessAnalyzer.census import us_census
import h3
import osmnx as ox
import os 
import zipfile
import pandas as pd
import geopandas as gpd 
import numpy as np
import warnings 
import polars as pl 
from datetime import time

def process_with_quality_matrix(
        pois,
        aoi,
        osm_xml_path,
        pbf_path=configs.PBF_OSM_PATH,
        network_filter="walk+bike+primary",
        min_edge_length=50,
        graph_path=None,
        quality_matrix=[200,300,400,500,600,700,800,900,1000,1250,1500,1750,2000,2500],
        h3_resolution=None,
        accessibility_streets_path=None,
        accessibility_h3_path=None,
        best_quality_mode=None,
        overwrite=False,
    ):
    if best_quality_mode is None:
        if np.ndim(quality_matrix) == 1:
            warnings.warn(
                "Inferred that accessibility represents a distance and not a quality. "
                "best_quality_mode set to 'min'."
            )
            best_quality_mode = "min"
        else:
            best_quality_mode = "max"

    if not(overwrite) and accessibility_streets_path is not None and os.path.isfile(accessibility_streets_path):
        accessibility_edges = gpd.read_file(accessibility_streets_path)
        if h3_resolution is None:
            return accessibility_edges
    
    else:
        pois = pois.copy()
        if graph_path is not None and os.path.isfile(graph_path):
            G = ox.load_graphml(graph_path)
        else:
            G = graph_processing.download_and_create_graph(
                osm_xml_path,
                pbf_path,
                aoi,
                network_filter=network_filter,
                min_edge_length=min_edge_length
            )
            if graph_path is not None:
                ox.save_graphml(G,graph_path)

        _, street_edges = ox.graph_to_gdfs(G)
        pois = pois.to_crs(street_edges.crs)
        pois = poi_utils.polygons_to_points(pois,street_edges)
        G, osmids = graph_processing.add_points_to_graph(
            pois,
            G,
            max_dist=100+min_edge_length, # Maximum distance from point to graph edge to project the point
            min_edge_length=min_edge_length # Minimum edge length after adding the new nodes
        )
        pois['osmid'] = osmids # Add the ids of the nodes in the graph to points

        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
        edges_gdf = edges_gdf[~edges_gdf.geometry.centroid.intersects(pois.to_crs(edges_gdf.crs).union_all())]
        nodes_gdf = nodes_gdf[
            nodes_gdf.index.isin(edges_gdf.reset_index()['u']) | 
            nodes_gdf.index.isin(edges_gdf.reset_index()['v'])
        ]
        G = ox.graph_from_gdfs(nodes_gdf,edges_gdf)

        accessiblity_graph = isochrones.graph(
            G,
            pois,
            quality_matrix, # If poi_quality_col is None it could be a list of distances
            poi_quality_col = None, # If all points have the same quality this could be None
            min_edge_length = min_edge_length # Do not add new nodes if there will be an edge with less than this length
        )
        # Save edges as gpkg
        _, accessibility_edges = ox.graph_to_gdfs(accessiblity_graph)

    if not(overwrite) and accessibility_h3_path is not None and os.path.isfile(accessibility_h3_path):
        return (
            accessibility_edges, 
            pd.read_csv(accessibility_h3_path)
        )
    else:
        return_tuple = (accessibility_edges)
        if accessibility_streets_path is not None and os.path.isfile(accessibility_streets_path):
            accessibility_edges.to_file(accessibility_streets_path)

        if h3_resolution:
            accessibility_h3_df = h3_utils.from_gdf(
                accessibility_edges,
                resolution=h3_resolution,
                columns=['accessibility'],
                contain="overlap",
                method=best_quality_mode,
                buffer=10
            )
            return_tuple = (accessibility_edges,accessibility_h3_df)
            if accessibility_h3_path is not None and os.path.isfile(accessibility_h3_path):
                accessibility_h3_df.to_csv(accessibility_h3_path)

        return return_tuple 


def worldpop_to_h3_accessibility(
    accessibility_h3_df,
    aoi,
    worlpop_path=configs.WORLDPOP_PATH,
    year=2025,
    resolution:str="100m", 
    dataset:str='pop', 
    subset:str="wpgpunadj", 
    results_h3_path=None,
    overwrite=False,
):
    if not(overwrite) and results_h3_path is not None and os.path.isfile(results_h3_path):
        return pd.read_csv(results_h3_path)
    
    population_file = population.download_worldpop_population(
        aoi,
        year,
        folder=configs.WORLDPOP_PATH,
        resolution=resolution,
        dataset=dataset,
        subset=subset,
    )
    # If downloaded the U18 file unzip it
    if ".zip" in population_file:
        zip_path = population_file

        # Extract to the same directory as the zip file
        extract_dir = os.path.splitext(zip_path)[0]
        os.makedirs(extract_dir, exist_ok=True)

        # Decompress the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Find the .tif file that contains '_T_'
        for file_name in os.listdir(extract_dir):
            if file_name.lower().endswith('.tif') and '_T_' in file_name:
                population_file = os.path.join(extract_dir, file_name)
                break
        else:
            raise FileNotFoundError("No .tif file containing '_T_' found in the zip archive.")
        
    h3_resolution = h3.get_resolution(accessibility_h3_df["h3_cell"].iloc[0])
    pop_h3_df = h3_utils.from_raster(population_file,aoi=aoi,resolution=h3_resolution)
    pop_h3_df = pop_h3_df.rename(columns={'value':'population'})
    results_h3_df = accessibility_h3_df.merge(pop_h3_df,left_index=True,right_index=True,how='outer')
    results_h3_df = h3_utils.to_gdf(results_h3_df).to_crs(aoi.crs)
    results_h3_df = results_h3_df[results_h3_df.intersects(aoi.union_all())]
    if results_h3_path is not None:
        results_h3_df.to_file(results_h3_path)

    return results_h3_df


def accessibility_h3_to_us_census(
    accessibility_h3_df,
    aoi,
    census_levels=["block","blockgroup"],
    pygris_path=configs.US_CENSUS_PYGRIS_PATH,
    fields_path=configs.US_CENSUS_FIELDS_PATH,
    year=2024,
    census_fields_categories=us_census.CENSUS_FIELDS_CATEGORIES,
    best_quality_mode=None,
    results_path=None,
):
    if best_quality_mode is None:
        if accessibility_h3_df["accessibility"].max() > 100:
            warnings.warn(
                "Inferred that accessibility represents a distance and not a quality. "
                "best_quality_mode set to 'min'."
            )
            best_quality_mode = "min"
        else:
            best_quality_mode = "max"

    if isinstance(census_levels,str):
        census_levels = [census_levels]

    census_levels = list(np.unique(["block"] + census_levels))

    us_nation = us_census.load_shapes(level="nation",year=year,cache=True,pygris_path=pygris_path)
    if ~aoi.to_crs(us_nation.crs).intersects(us_nation.union_all()):
        raise Exception("aoi is not in the US")

    us_states = us_census.load_shapes("states", year=year,cache=True,pygris_path=pygris_path)
    aoi_states = us_states.loc[
        us_states.geometry.intersects(
            aoi.to_crs(us_states.crs).union_all()
        ),
        "NAME",
    ].to_list()

    census_geometry_block = us_census.load_shapes(
        "block", state=aoi_states, cb=True, year=year,cache=True,pygris_path=pygris_path
    )
    geoid_col = us_census.pick_geoid_column(census_geometry_block.columns)
    census_geometry_block = census_geometry_block.rename(columns={geoid_col:"GEOID"})
    census_geometry_block["GEOID"] = census_geometry_block["GEOID"].astype(int)
    accessibility_h3_df["accessibility"] = accessibility_h3_df["accessibility"].astype(float)
    accessibility_h3_block = h3_utils.to_gdf(accessibility_h3_df,h3_column="h3_cell")
    accessibility_h3_block.geometry = accessibility_h3_block.geometry.centroid
    accessibility_h3_block = geometry_utils.resample_gdf(
        accessibility_h3_block[["accessibility","geometry"]],
        census_geometry_block[["GEOID","geometry"]],
        contain="centroid",
        method=best_quality_mode,
        id_column="GEOID",
    ).reset_index(drop=True)

    valid_census_levels = ["state","county","tract","blockgroup","block"]

    census_dict = {}
    for i in range(len(census_levels)):
        level = census_levels[i]
        if results_path is not None and os.path.isfile(results_path+f"/us_census_{level}.gpkg"):
            census_gdf = gpd.read_file(results_path+f"/us_census_{level}.gpkg")
            census_dict[level] = census_gdf 
            continue

        if level in valid_census_levels:
            census_gdf = []
            for state in aoi_states:
                census_geometry = us_census.load_shapes(
                    level, state=state, cb=True, year=year,cache=True,pygris_path=pygris_path
                )
                geoid_col = us_census.pick_geoid_column(census_geometry.columns)
                census_geometry = census_geometry.rename(columns={geoid_col:"GEOID"})
                census_geometry["GEOID"] = census_geometry["GEOID"].astype(int)
                census_geometry = census_geometry[
                    census_geometry.geometry.intersects(aoi.to_crs(census_geometry.crs).union_all())
                ]

                census_fields = us_census.load_fields(
                    state=state,
                    api_key=api_keys.US_CENSUS,
                    categories = census_fields_categories,
                    level = level,
                    cache=True,
                    fields_path=fields_path,
                )
                geoid_col = us_census.pick_geoid_column(census_fields.columns)
                census_fields = census_fields.rename(columns={geoid_col:"GEOID"})
                census_i = us_census.join_census(
                    census_geometry,
                    census_fields,
                )
                census_gdf.append(census_i)

            census_gdf = pd.concat(census_gdf)
            census_gdf = us_census.compute_densities_and_ratios(
                census_gdf,
                categories=us_census.CENSUS_FIELDS_CATEGORIES
            )
            
            if level == "block":
                census_gdf = census_gdf.merge(accessibility_h3_block[["GEOID","accessibility"]],on="GEOID",how="left")
            else:
                census_gdf = us_census.resample(
                    census_dict[census_levels[i-1]]["accessibility",f"{year}_population_total","geometry"],
                    census_gdf,
                    columns=["accessiblity"],
                    weights=[f"{year}_population_total"]
                ).reset_index(drop=True)
        else:
            census_geometry = us_census.load_shapes(
                level, state=aoi_states, cb=True, year=year,cache=True,pygris_path=pygris_path
            )
            geoid_col = us_census.pick_geoid_column(census_geometry.columns)
            census_geometry = census_geometry.rename(columns={geoid_col:"GEOID"})
            census_geometry["GEOID"] = census_geometry["GEOID"].astype(int)
            census_geometry = census_geometry[
                census_geometry.geometry.intersects(aoi.to_crs(census_geometry.crs).union_all())
            ]
        
            census_gdf = census_dict[us_census.CENSUS_RESAMPLE[census_levels[i]]].copy()
            census_gdf = census_gdf[
                [col for col in census_gdf.columns 
                if "_ratio" not in col and "_density" not in col]
            ]
            census_gdf = us_census.resample(
                census_gdf,
                census_geometry,
                categories=us_census.CENSUS_FIELDS_CATEGORIES,
                columns=["accessiblity"],
                weights=[f"{year}_population_total"]
            ).reset_index(drop=True)

            census_gdf = us_census.compute_densities_and_ratios(
                census_gdf,
                categories=us_census.CENSUS_FIELDS_CATEGORIES
            )

        if results_path is not None:
            census_gdf.to_file(results_path+f"/us_census_{level}.gpkg")

        census_dict[level] = census_gdf 

    return census_dict


        
def get_transit_grids_and_quality_funcs(
    # --- Transit modes mapping ---
    route_type_to_mode = {
        'bus': 'all',
        'tram': [0, 1, 2, 4, 5, 6, 7],
        'rail': [1, 2]
    },

    # --- Quality scoring elasticities ---
    headway_elasticity = 0.35,
    walk_elasticity = 0.25,
    speed_elasticity = 0.2,

    # --- Mode-specific quality factors ---
    mode_factor = {
        'bus': 0.85,
        'tram': 0.92,
        'rail': 1
    },

    # --- Accessibility score grid ---
    n_accessibility_scores = 10,

    # --- Parameter bounds ---
    max_headway = 1440,       # minutes
    max_walk_distance = 2000, # meters
    max_speed = 150,          # km/h
    min_headway = 5,
    min_walk_distance = 100,
    min_speed = 5,

    # --- Reference quality points for calibration ---
    best_quality = {'headway': 5, 'mode': 'rail', 'speed': 30, 'distance': 100},
    worst_quality = {'headway': 720, 'mode': 'bus', 'speed': 10, 'distance': 2000},
):
    # ============================================================
    # ===================== QUALITY FUNCTIONS ===================
    # ============================================================

    def headway_quality(headway):
        return quality_utils.elasticity_based_quality(headway, min_headway, -headway_elasticity)

    def walk_quality(distance):
        return quality_utils.elasticity_based_quality(distance, min_walk_distance, -walk_elasticity)

    def speed_quality(speed):
        return quality_utils.elasticity_based_quality(speed, max_speed, speed_elasticity)

    def mode_quality(mode):
        if isinstance(mode, str):
            return mode_factor[mode]
        else:
            mode_arr = np.array(mode)
            vectorized_lookup = np.vectorize(lambda m: mode_factor[m])
            return vectorized_lookup(mode_arr)

    def stop_quality(headway, mode, speed):
        return headway_quality(headway) * mode_quality(mode) * speed_quality(speed)

    stop_quality = quality_utils.calibrate_quality_func(
        stop_quality,
        min_quality=1/n_accessibility_scores,
        max_quality=1,
        min_point=(worst_quality['headway'], worst_quality['mode'], worst_quality['speed']),
        max_point=(best_quality['headway'], best_quality['mode'], best_quality['speed']),
    )

    def access_quality(headway, mode, speed, distance):
        return stop_quality(headway, mode, speed) * walk_quality(distance)

    access_quality = quality_utils.calibrate_quality_func(
        access_quality,
        min_quality=1/n_accessibility_scores,
        max_quality=1,
        min_point=(worst_quality['headway'], worst_quality['mode'], worst_quality['speed'], worst_quality['distance']),
        max_point=(best_quality['headway'], best_quality['mode'], best_quality['speed'], best_quality['distance']),
    )

    # Adaptive grids for quality scoring
    headway_grid, mode_grid, speed_grid, distance_grid = quality_utils.build_adaptive_grids(
        access_quality,
        variables=[
            [min_headway, max_headway],
            list(route_type_to_mode.keys()),
            [min_speed, max_speed],
            [min_walk_distance, max_walk_distance],
        ],
        delta=1/n_accessibility_scores
    )
    speed_grid = [0, *speed_grid]

    return (headway_grid, mode_grid, speed_grid, distance_grid), access_quality, stop_quality


def most_frequent_row_range(values, bins=5):
    values = np.asarray(values)

    # Thresholding
    threshold = 0.5 * np.percentile(values, 90)
    cleaned = values.copy()
    cleaned[cleaned < threshold] = 0
    nonzero_mask = cleaned > 0
    nz_vals = cleaned[nonzero_mask]
    nz_idx = np.where(nonzero_mask)[0]

    if len(nz_vals) == 0:
        return []
    elif len(nz_vals) == 1:
        return [int(nz_idx[0])]  # single value

    # Histogram to find high-frequency bins
    counts, edges = np.histogram(nz_vals, bins=bins)
    max_count = counts.max()
    max_bins = np.where(counts == max_count)[0]

    # Include all tied max-count bins in the bounds
    start = edges[max_bins[0]]
    end = edges[max_bins[-1] + 1]

    in_bin = (nz_vals >= start) & (nz_vals <= end)
    vals_in = nz_vals[in_bin]
    idx_in = nz_idx[in_bin]

    if len(vals_in) == 0:
        return []

    # Compute mean and std of values in high-frequency bin
    mean_val = np.mean(vals_in)
    std_val = np.std(vals_in)

    # Define lower and upper bounds (mean ± 1 std)
    lower_bound = mean_val - std_val
    upper_bound = mean_val + std_val

    # Find all indices corresponding to values within this range
    in_range_mask = (vals_in >= lower_bound) & (vals_in <= upper_bound)
    indices_in_range = idx_in[in_range_mask]

    return indices_in_range.tolist()


def most_frequent_row_index(values, bins=5):
    if isinstance(values,pl.LazyFrame):
        values = values.collect()

    if isinstance(values,pl.DataFrame):
        values = values.to_pandas()
    
    if isinstance(values,pd.DataFrame):
        if "date" in values.columns:
            values['date'] = pd.to_datetime(values['date'])
            values = values.dropna(subset=['date']).sort_values('date')
        else:
            raise Exception("Column 'date' is mandatory in values if passing a DataFrame instead of a list.")
        
        if "service_intensity" in values.columns:
            values = values.dropna(subset=["service_intensity"])
            values['service_intensity'] = values['service_intensity'].astype(float)
        else:
            raise Exception("Column 'service_intensity' is mandatory in values if passing a DataFrame instead of a list.")
        
        values = values.sort_values("date")
        if "file_id" in values.columns:
            id_col = "file_id"
        elif "gtfS_name" in values.columns:
            id_col = "gtfs_name"
        else:
            id_col = None
            values = list(values['service_intensity'])
        
    else:
        id_col = None

    if id_col is None:
        indices = most_frequent_row_range(values, bins=bins)
        if indices is None:
            return None 
        # Compute mean of the selected values
        mean_val = np.mean(values[indices])
        # Compute absolute differences between all values and the mean
        diffs = np.abs(values - mean_val)
        if len(diffs) == 0:
            return None
        # Find the minimum difference
        min_diff = np.min(diffs)
        # Find all indices where difference equals minimum
        nearest_indices = np.where(diffs == min_diff)[0]
        # Return first nearest index or None
        if nearest_indices.size > 0:
            return int(nearest_indices[0])
        else:
            return None
    else:
        indices = []

        # Loop over each unique ID
        for uid in values[id_col].unique():
            # Get the service_intensity column for this ID
            intensity_series = values.loc[values[id_col] == uid, "service_intensity"]

            # Get list of row indices in the most frequent range
            indices_i = most_frequent_row_range(intensity_series, bins=bins)
        
            # Map back to the DataFrame's actual index
            if indices_i is not None:  # only add if non-empty
                indices += list(intensity_series.iloc[indices_i].index)

        if len(indices)==0:
            return None  # no valid indices found

        # From the selected rows, find the index of the max service_intensity
        max_idx = values.loc[indices, "service_intensity"].idxmax()
        return max_idx


def get_trasit_stop_quality(
    gtfs,
    headway_grid, 
    mode_grid, 
    speed_grid,
    stop_quality_func,
    route_type_to_mode={
        'bus': 'all',
        'tram': [0, 1, 2, 4, 5, 6, 7],
        'rail': [1, 2]
    },
    start_time=time(hour=8),
    end_time=time(hour=20),
    stop_id="parent_station",
    stops_path=None,
):
    if stops_path is not None and os.path.isfile(stops_path):
        stop_headway_df = gpd.read_file(stops_path)
    else:
        # ============================================================
        # ===================== GTFS PROCESSING ======================
        # ============================================================
        # Select study day
        service_intensity = gtfs.get_service_intensity_in_date_range(by_feed=True).to_pandas()
        idx = most_frequent_row_index(service_intensity)
        selected_day = service_intensity.iloc[idx]['date'].to_pydatetime()

        # Compute stop speed at selected day
        speed_by = "trip_id"
        stop_speed_lf = gtfs.get_speed_at_stops(
            date=selected_day,
            start_time=start_time,
            end_time=end_time,
            route_types='all',
            by=speed_by,
            at=stop_id,
            how="mean",
            direction='both',
            time_step=15
        )
        if isinstance(stop_speed_lf, pl.DataFrame):
            stop_speed_lf = stop_speed_lf.lazy()

        # Filter GTFS for selected day/time
        gtfs_lf = gtfs.filter(date=selected_day, start_time=start_time, end_time=end_time)
        gtfs_lf = gtfs_lf.join(stop_speed_lf.select([stop_id, speed_by, 'speed']),
                                on=[stop_id, speed_by], how='left')


        # ============================================================
        # ===================== HEADWAY CALCULATION =================
        # ============================================================
        stop_headway_df = []

        for mode in mode_grid:
            gtfs_selection = gtfs._filter_by_route_type(gtfs_lf, route_type_to_mode[mode])
            gtfs_length = gtfs_selection.select(pl.len()).collect().item()
            if gtfs_length == 0:
                continue

            for i in range(len(speed_grid)-1):
                gtfs_selection_i = gtfs_selection.filter(pl.col("speed") >= speed_grid[i])
                gtfs_length_i = gtfs_selection_i.select(pl.len()).collect().item()
                if gtfs_length_i == gtfs_length or gtfs_length_i == 0:
                    continue
                gtfs_length = gtfs_length_i

                df = gtfs._get_headway_at_stops(
                    gtfs_selection_i,
                    date=selected_day,
                    start_time=start_time,
                    end_time=end_time,
                    by="shape_direction",
                    at=stop_id,
                    how="best",
                    n_divisions=1
                ).with_columns(
                    pl.lit(mode).alias("mode"),
                    pl.lit(speed_grid[i+1]).alias("speed_grid")
                )
                stop_headway_df.append(df)

        stop_headway_df = pl.concat(stop_headway_df).to_pandas()
        stop_headway_df["headway_grid"] = np.array(headway_grid)[
            np.searchsorted(np.array(headway_grid), stop_headway_df["headway"], side="left")
        ]

        # Add coordinates and route names
        stop_headway_df = gtfs.add_stop_coords(stop_headway_df)
        stop_headway_df = gtfs.add_route_names(stop_headway_df)
        stop_headway_df = gpd.GeoDataFrame(
            stop_headway_df,
            geometry=gpd.points_from_xy(stop_headway_df['stop_lon'], stop_headway_df['stop_lat']),
            crs=4326
        )
        stop_headway_df = stop_headway_df[stop_headway_df.geometry.is_valid]
        stop_headway_df["stop_quality"] = stop_headway_df.apply(
            lambda row: stop_quality_func(row["headway"], row["mode"], row["speed_grid"]), axis=1
        )
        stop_headway_df["stop_quality_grid"] = stop_headway_df.apply(
            lambda row: stop_quality_func(row["headway_grid"], row["mode"], row["speed_grid"]), axis=1
        ).round(3)
        stop_headway_df = stop_headway_df.sort_values("stop_quality").drop_duplicates(stop_id, keep="last")
        stop_headway_df = stop_headway_df.sort_values(stop_id).reset_index(drop=True)
        if stops_path is not None:
            stop_headway_df.to_file(stops_path)

    return stop_headway_df
