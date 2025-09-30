import os
import requests
import tempfile
import shapely
import geopandas as gpd
import pandas as pd
import osmnx as ox
from osm2geojson import json2geojson

from . import utils


def _write_poly_file(aoi: gpd.GeoDataFrame | gpd.GeoSeries, poly_path: str):
    """
    Write AOI geometry to a .poly file in Osmosis format, correctly
    handling Polygons, MultiPolygons, and interior rings (holes).
    """
    # Ensure a single, valid geometry in WGS84 projection
    geom = aoi.to_crs(4326).union_all()

    with open(poly_path, "w") as f:
        f.write("aoi\n")  # Polygon name

        def write_ring(coords, ring_id, is_hole=False):
            """Helper to write a single ring to the file."""
            prefix = "!" if is_hole else ""
            f.write(f"{prefix}{ring_id}\n")
            for x, y in coords:
                # Use sufficient precision for coordinates
                f.write(f"  {x:.7f} {y:.7f}\n")
            f.write("END\n")

        ring_counter = 1

        # Standardize geometry into a list of polygons
        polygons = []
        if geom.geom_type == "Polygon":
            polygons.append(geom)
        elif geom.geom_type == "MultiPolygon":
            polygons.extend(geom.geoms)
        else:
            raise ValueError(
                f"Unsupported geometry type for .poly file: {geom.geom_type}"
            )

        for poly in polygons:
            # Write the exterior ring
            write_ring(poly.exterior.coords, ring_counter)
            ring_counter += 1
            # Write all interior rings (holes)
            for interior in poly.interiors:
                write_ring(interior.coords, ring_counter, is_hole=True)
                ring_counter += 1

        f.write("END\n")  # Final END for the entire polygon definition


def download_geofabrik(
    aoi: gpd.GeoDataFrame | gpd.GeoSeries, output_folder: str = None
):
    """
    Finds the smallest Geofabrik region that contains the AOI and downloads the PBF file.

    Args:
        aoi: A geopandas GeoDataFrame, GeoSeries, or a single shapely geometry.
        output_folder: The folder where the downloaded PBF file will be saved.

    Returns:
        The full path to the downloaded .osm.pbf file.
    """
    # Ensure the input AOI is a single shapely geometry in the correct CRS (EPSG:4326)

    aoi = aoi.to_crs(4326)
    aoi_geom = aoi.union_all()
    if not aoi_geom.is_valid:
        print("Validity problem:", shapely.validation.explain_validity(aoi_geom))

    # Load Geofabrik regions metadata from the current index file
    url = "https://download.geofabrik.de/index-v1.json"
    print(f"Fetching Geofabrik index from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    candidate_regions = []

    # Iterate through the flat list of features in the GeoJSON
    for feature in data["features"]:
        properties = feature.get("properties", {})

        # Region is only a candidate if it has a PBF URL and a geometry
        if not properties.get("urls", {}).get("pbf") or not feature.get("geometry"):
            continue

        try:
            # Create a shapely geometry from the GeoJSON feature
            region_geom = shapely.geometry.shape(feature["geometry"])
        except Exception as e:
            print(
                f"Warning: Could not process geometry for {properties.get('name', 'N/A')}: {e}"
            )
            continue

        # Check if the region's polygon completely contains the AOI.
        # This is more accurate than comparing bounding boxes.
        if region_geom.contains(aoi_geom):
            area = region_geom.area
            candidate_regions.append((area, properties))

    if not candidate_regions:
        raise ValueError(
            "No Geofabrik region was found to contain your AOI. Please ensure the AOI is correct and within a single region."
        )

    # Sort candidates by area (smallest first) to find the best fit
    candidate_regions.sort(key=lambda x: x[0])
    best_region = candidate_regions[0][1]

    pbf_url = best_region.get("urls", {}).get("pbf")

    # Prepare output filename & path
    safe_name = utils.sanitize_filename(best_region.get("name", "region"))
    filename = f"{safe_name}.osm.pbf"

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, filename)
    else:
        output_file = filename

    if os.path.exists(output_file):
        print(f"File '{output_file}' already exists. Skipping download.")
        return output_file

    print(f"Downloading '{best_region['name']}' from {pbf_url} ...")
    pbf_response = requests.get(pbf_url, stream=True)
    pbf_response.raise_for_status()

    with open(output_file, "wb") as f:
        for chunk in pbf_response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded geofabrik to {output_file}")
    return output_file


def build_osmium_filter_args(tag_filters: dict[str, set[str] | None]) -> str:
    parts = []
    for k, vs in tag_filters.items():
        if vs is None or len(vs) == 0:
            # Filter by key only (any value)
            parts.append(f"w/{k}")
        else:
            for v in vs:
                parts.append(f"w/{k}={v}")
    return " ".join(parts)


def osmium_network_filter(network_type: str) -> dict[str, set[str] | None]:
    walk_highways = {
        "footway",
        "pedestrian",
        "path",
        "living_street",
        "steps",
        "residential",
        "service",
        "unclassified",
        "track",
    }
    bike_highways = {
        "cycleway",
        "path",
        "residential",
        "living_street",
        "unclassified",
        "service",
        "track",
    }
    drive_highways = {
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "residential",
        "unclassified",
        "service",
        "living_street",
    }
    # all_highways = {
    #     "motorway",
    #     "motorway_link",
    #     "trunk",
    #     "trunk_link",
    #     "primary",
    #     "primary_link",
    #     "secondary",
    #     "secondary_link",
    #     "tertiary",
    #     "tertiary_link",
    #     "residential",
    #     "unclassified",
    #     "service",
    #     "living_street",
    #     "track",
    #     "path",
    #     "cycleway",
    #     "footway",
    #     "pedestrian",
    #     "steps",
    # }
    primary_highways = {
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "residential",
        "unclassified",
        "service",
        "living_street",
    }

    if network_type == "walk":
        # ways with walk highways + foot=yes ways
        tag_filters = {
            "highway": walk_highways,
            "foot": {"yes"},
        }

    elif network_type == "bike":
        # ways with bike highways + bicycle=yes ways
        tag_filters = {
            "highway": bike_highways,
            "bicycle": {"yes"},
        }

    elif network_type == "drive":
        # only drive highways
        tag_filters = {
            "highway": drive_highways,
        }

    elif network_type == "all":
        # all highways
        tag_filters = {
            "highway": None,
        }

    elif network_type == "walk+bike":
        # combine walk+bike highways, plus foot=yes or bicycle=yes, exclude drive highways
        combined = walk_highways.union(bike_highways)
        tag_filters = {
            "highway": combined,
            "foot": {"yes"},
            "bicycle": {"yes"},
        }
    elif network_type == "walk+bike+primary":
        # combine walk+bike highways, plus foot=yes or bicycle=yes, exclude drive highways
        combined = walk_highways.union(bike_highways)
        combined = combined.union(primary_highways)
        tag_filters = {
            "highway": combined,
            "foot": {"yes"},
            "bicycle": {"yes"},
        }

    else:
        raise ValueError(f"Unknown network_type: {network_type}")

    return build_osmium_filter_args(tag_filters)


def geofabrik_to_osm(
    output_file: str,
    input_file: str = "",
    aoi: gpd.GeoDataFrame | gpd.GeoSeries = None,
    osmium_filter_args: str = "",
    overwrite: bool = False,
):
    if os.path.isfile(output_file) and (not overwrite):
        print(f"File '{output_file}' already exists. Skipping conversion.")
        return output_file

    # Your download logic (assumed to be working)
    if not input_file or not os.path.isfile(input_file):
        input_path = utils.get_folder(input_file)
        print(
            f"File {input_file} does not exist. Downloading best matching geofabrik file."
        )
        input_file = download_geofabrik(aoi, input_path)

    current_file = input_file
    tag_filter_temp_file_path = None

    try:
        # Step 1: Apply tag filter (if any) to an intermediate temporary file
        if osmium_filter_args:
            # Create a temp file, get its name, and close it immediately
            with tempfile.NamedTemporaryFile(suffix=".pbf", delete=False) as f:
                tag_filter_temp_file_path = f.name

            print(f"Applying tag filter: {osmium_filter_args}")
            cmd_filter = f"osmium tags-filter --overwrite {current_file} {osmium_filter_args} -o {tag_filter_temp_file_path}"
            if os.system(cmd_filter) != 0:
                raise RuntimeError("Tag filtering failed")
            current_file = tag_filter_temp_file_path  # The next step uses this new file

        # Step 2: Apply geometry filter or just convert
        if aoi is not None:
            # Create, write, and CLOSE the .poly file BEFORE using it
            poly_temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".poly", delete=False) as f:
                    poly_temp_file_path = f.name

                print("Creating .poly file for AOI clipping...")
                _write_poly_file(
                    aoi, poly_temp_file_path
                )  # Write to the closed file's path

                print("Extracting by geometry...")
                cmd_extract = f"osmium extract --overwrite --polygon {poly_temp_file_path} {current_file} -o {output_file} -f osm"
                if os.system(cmd_extract) != 0:
                    raise RuntimeError("Geometry extraction failed")
            finally:
                # Ensure the .poly file is always cleaned up
                if poly_temp_file_path and os.path.exists(poly_temp_file_path):
                    os.remove(poly_temp_file_path)
        else:
            # No AOI, just convert the format
            print("No AOI provided, converting to .osm format...")
            cmd_convert = f"osmium cat {current_file} -o {output_file} -f osm"
            if os.system(cmd_convert) != 0:
                raise RuntimeError("OSM conversion failed")

    finally:
        # Final cleanup: remove the intermediate tag-filtered file if it was created
        if tag_filter_temp_file_path and os.path.exists(tag_filter_temp_file_path):
            os.remove(tag_filter_temp_file_path)

    print("Finished. Final output:", output_file)
    return output_file


def download_street_graph(
    bounds: gpd.GeoSeries | gpd.GeoDataFrame,
    network_type: str = "walk",
    custom_filter=None,
):
    if bounds.crs.is_projected:
        crs = bounds.crs
    else:
        crs = bounds.estimate_utm_crs()

    # network_type (string {"all", "all_public", "bike", "drive", "drive_service", "walk"}) – what type of street network to get if custom_filter is None
    G = ox.graph.graph_from_polygon(
        bounds.to_crs(4326).union_all(),
        network_type=network_type,
        simplify=True,
        retain_all=False,
        truncate_by_edge=True,
        custom_filter=custom_filter,
    )
    G = ox.projection.project_graph(G, to_crs=crs)

    return G


def overpass_api_query(query: str, bounds: gpd.GeoDataFrame | gpd.GeoSeries):
    bbox = bounds.to_crs(4326).total_bounds
    bbox = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    query = query.replace("{{bbox}}", bbox)
    query = query.replace("{bbox}", bbox)
    query = query.replace("bbox", bbox)
    query = query.replace("[out:xml]", "[out:json]")
    overpass_url = "https://overpass-api.de/api/interpreter"
    response = requests.get(overpass_url, params={"data": query})
    geojson_response = json2geojson(response.json())
    gdf = gpd.GeoDataFrame.from_features(geojson_response, crs=4326).reset_index(
        drop=True
    )
    new_gdf = gdf["tags"].apply(pd.Series)
    if "type" in new_gdf.columns:
        new_gdf = new_gdf.rename(columns={"type": "geometry_type"})

    gdf = pd.concat([gdf.drop(columns=["tags"]), new_gdf], axis=1).reset_index(
        drop=True
    )
    gdf = gdf.loc[:, ~gdf.columns.duplicated()]
    return gdf.to_crs(bounds.crs)


# def green_areas(bounds, pedestrian_graph=None):
#     query = """
#         [out:json][timeout:25];
#         (
#         node[leisure = "garden"]({{bbox}});
#         node[leisure = "park"]({{bbox}});
#         node[landuse = "greenfield"]({{bbox}});
#         node[landuse = "grass"]({{bbox}});
#         node[landuse = "forest"]({{bbox}});
#         way[leisure = "garden"]({{bbox}});
#         way[leisure = "park"]({{bbox}});
#         way[landuse = "greenfield"]({{bbox}});
#         way[landuse = "grass"]({{bbox}});
#         way[landuse = "forest"]({{bbox}});
#         relation[leisure = "garden"]({{bbox}});
#         relation[leisure = "park"]({{bbox}});
#         relation[landuse = "greenfield"]({{bbox}});
#         relation[landuse = "grass"]({{bbox}});
#         relation[landuse = "forest"]({{bbox}});
#         );
#         out body;
#         >;
#         out skel qt;
#     """
#     green_areas_gdf = overpass_api_query()  # query,bounds)

#     # green_areas_gdf = service_quality.service_type(green_areas_gdf,row_values=['park','forest','garden','grass','greenfield'])
#     # green_areas_gdf = service_quality.merge_and_simplify_geometry(green_areas_gdf,buffer=5,min_width=10,simplify=1,min_area=200)

#     if pedestrian_graph is not None:
#         edges_union = (
#             ox.graph_to_gdfs(pedestrian_graph, nodes=False)
#             .to_crs(green_areas_gdf.crs)
#             .union_all()
#         )
#         geoms = list(green_areas_gdf.geometry)
#         shapely.prepare(geoms)
#         green_areas_gdf = green_areas_gdf[shapely.intersects(geoms, edges_union)]

#     return green_areas_gdf.to_crs(bounds.crs)


def bus_stops(bounds):
    query = """
        [out:json][timeout:25];
        (
        node["highway"="bus_stop"]({{bbox}});
        );
        out body;
        >;
        out skel qt;
    """
    stops = overpass_api_query(query, bounds)
    return stops.to_crs(bounds.crs)
