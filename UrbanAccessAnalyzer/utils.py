import shapely
import geopandas as gpd
import os
import warnings
import numpy as np

gpd.options.io_engine = "pyogrio"
PYOGRIO_USE_ARROW=1


def ipyleaflet_drawable_map(center=[0, 0], zoom=11, height="800px"):
    import ipyleaflet
    from ipyleaflet import DrawControl, Map
    from shapely.geometry import shape
    from IPython.display import display
    
    """
    Creates an interactive ipyleaflet map with drawing controls (squares & polygons).
    
    Returns:
        - m: The ipyleaflet map object
        - get_drawn_geometries: Function to return stored geometries as a GeoDataFrame
    """
    # Create a map
    m = Map(center=center, zoom=zoom, scroll_wheel_zoom=True, layout={'height': height})

    google_hybrid = ipyleaflet.TileLayer(
        url="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        name="Google Hybrid",
        attribution="Google"
    )
    m.add_layer(google_hybrid)

    # Create a DrawControl object with squares (rectangles) and polygons
    draw_control = DrawControl(
        rectangle={"shapeOptions": {"color": "blue"}},  # Allow squares & rectangles
        polygon={"shapeOptions": {"color": "blue"}},    # Allow polygons
        circle={},
        polyline={},
        marker={},
        circlemarker={}
    )

    # Add the DrawControl to the map
    m.add_control(draw_control)

    # Initialize an empty list to store drawn geometries
    drawn_geometries = []

    # Handle drawn geometries
    def handle_draw(self, action, geo_json):
        """Callback function to store drawn geometries."""
        if action == 'created':  # Store new geometries only
            geometry = shape(geo_json['geometry'])  # Convert GeoJSON to Shapely geometry
            drawn_geometries.append(geometry)
            print(f"New geometry added: {geometry}")

    # Register the handle_draw function
    draw_control.on_draw(handle_draw)

    def get_drawn_geometries():
        """Returns the drawn geometries as a GeoDataFrame."""
        if drawn_geometries:
            gdf = gpd.GeoDataFrame({"geometry": drawn_geometries}, crs="EPSG:4326")
            return gdf
        else:
            return None  # No geometries drawn yet
    
    return m, get_drawn_geometries

def intersects_all_with_all(G:gpd.GeoDataFrame|gpd.GeoSeries,g:gpd.GeoDataFrame|gpd.GeoSeries):
    g = g.to_crs(G.crs)
    _g = np.array(np.repeat(np.transpose(np.array(g.geometry)[np.newaxis,:]),len(G),axis=1))
    _G = list(G.geometry)
    shapely.prepare(_G)
    shapely.prepare(_g)
    return shapely.intersects(_G,_g).transpose()

def alpha_shape(geoseries:gpd.GeoSeries,buffer:float=0,max_segment_len:float=0,alpha:float=0.8):
    from alpha_shapes import Alpha_Shaper
    
    crs = geoseries.crs 
    if crs.is_projected == False:
        utm = geoseries.estimate_utm_crs()
    
    geoseries = geoseries.to_crs(utm)
    geom = shapely.unary_union(geoseries.geometry)


    if buffer > 0:
        geom_b = geom.buffer(buffer,cap_style='square')
        parts = shapely.get_parts(geom_b)
        for i in range(len(parts)): 
            parts[i] = geom.intersection(parts[i])
            if (max_segment_len > 0) and ("LineString" in str(type(parts[i]))):
                parts[i] = shapely.segmentize(parts[i],max_segment_len) 

    elif "Polygon" in str(type(geom)):
        parts = shapely.get_parts(geom)
    else:
        parts = [geom]

    result = []
    for p in parts:
        coords = shapely.get_coordinates(shapely.extract_unique_points(p))
        coords = coords.tolist() 
        if len(coords) < 3:
            continue
        try:
            shaper = Alpha_Shaper(coords)
        except:
            print("Alpha Shaper error with points ",coords) 
            continue

        a,_ = shaper.optimize()
        poly = shaper.get_shape(alpha=a*alpha)
        result.append(poly)

    result = gpd.GeoSeries(result,crs=utm)
    result = result.to_crs(crs)

    return result 

def remove_small_objects_and_holes(geometry:gpd.GeoSeries | gpd.GeoDataFrame,remove_small_objects:float=0,remove_small_holes:float=0):
    crs = geometry.crs 
    if crs.is_projected == False:
        geometry = geometry.to_crs(geometry.geometry.estimate_utm_crs())
    
    geometry = geometry.buffer(-remove_small_objects/2)
    geometry = geometry.buffer(remove_small_objects/2 + remove_small_holes / 2)
    geometry = geometry.buffer(-remove_small_holes/2)
    geometry = geometry.to_crs(crs)
    return geometry


      
def complex_split(geom, splitter,tolerance:float=1.0e-4):
    from shapely.geometry import GeometryCollection, Point, MultiPoint, Polygon, MultiPolygon
    from shapely.ops import split, snap
    """Split a complex linestring by another geometry without splitting at
    self-intersection points.

    Parameters
    ----------
    geom : LineString
        An optionally complex LineString.
    splitter : Geometry
        A geometry to split by.

    Warnings
    --------
    A known vulnerability is where the splitter intersects the complex
    linestring at one of the self-intersecting points of the linestring.
    In this case, only one the first path through the self-intersection
    will be split.

    Examples
    --------
    >>> complex_line_string = LineString([(0, 0), (1, 1), (1, 0), (0, 1)])
    >>> splitter = LineString([(0, 0.5), (0.5, 1)])
    >>> complex_split(complex_line_string, splitter).wkt
    'GEOMETRYCOLLECTION (LINESTRING (0 0, 1 1, 1 0, 0.25 0.75), LINESTRING (0.25 0.75, 0 1))'

    Return
    ------
    GeometryCollection
        A collection of the geometries resulting from the split.
    """
    #if geom.is_simple:
    #    return split(geom, splitter)
    
    if isinstance(splitter, Point) or isinstance(splitter, MultiPoint):
        splitter = splitter.buffer(tolerance/2.01)

    if isinstance(splitter, Polygon):
        splitter = splitter.exterior

    if isinstance(splitter, MultiPolygon):
        splitter = splitter.boundary

    # Ensure that intersection exists and is zero dimensional.
    relate_str = geom.relate(splitter)

    if relate_str[0] == '1':
        raise ValueError('Cannot split LineString by a geometry which intersects a '
                         'continuous portion of the LineString.')
    if not (relate_str[0] == '0' or relate_str[1] == '0'):
        return GeometryCollection((geom,))

    intersection_points = geom.intersection(splitter)
    intersection_points = shapely.union_all(snap(intersection_points,intersection_points,tolerance=tolerance))
    # This only inserts the point at the first pass of a self-intersection if
    # the point falls on a self-intersection.
    snapped_geom = snap(geom, intersection_points, tolerance=tolerance)  # may want to make tolerance a parameter.
    # A solution to the warning in the docstring is to roll your own split method here.
    # The current one in shapely returns early when a point is found to be part of a segment.
    # But if the point was at a self-intersection it could be part of multiple segments.
    return split(snapped_geom, intersection_points)


def gdf_from_file(file,layer:int|str=None,columns=None,bounds:gpd.GeoSeries=None,crs=None,reset_index:bool=True):
    from pyogrio import read_info, read_dataframe


    if not os.path.isfile(file):
        raise Exception(f"file {file} not found")
    
    info = read_info(file)
    file_crs = info['crs']
    if type(file_crs) == None or len(file_crs) == 0:
        file_crs = crs

    if type(bounds) != type(None):
        bounds = bounds.to_crs(file_crs).union_all()

    x = read_dataframe(file,layer=layer,columns=columns,mask=bounds,use_arrow=True)

    if ("GeoDataFrame" not in str(type(x))) and ("GeoSeries" not in str(type(x))):
        if 'geometry' in x.keys():
            from shapely import wkt
            x['geometry'] = x['geometry'].apply(wkt.loads)
            x = gpd.GeoDataFrame(x, crs=crs)
        else:
            warnings.warn("No geometry column found.")

    if type(x.crs) == type(None):
        x.crs = crs 
    elif type(crs) != type(None):
        x = x.to_crs(crs)

    if type(x.crs) == type(None):
        raise Exception("crs not defined")
    
    x = x[x.is_valid]
    if reset_index:
        x = x.reset_index(drop=True) 
    
    return x    

def raster_crop(input_path:str, bounds:gpd.GeoSeries):
    from rasterio.windows import from_bounds#, bounds as window_bounds
    if len(bounds) > 1:
        import geometry
        bounds = geometry.merge_gdf(bounds)
    # Open the GeoTIFF file
    try:
        src = rio.open(input_path,'r+')
    except:
        src = rio.open(input_path,'r')

    crs = validate_crs(src)
    # Check and reproject the bounding box coordinates if needed
    if bounds.crs != crs:
        bounds = bounds.to_crs(crs)

    # Get the window coordinates based on the bounding box
    window = from_bounds(*bounds.total_bounds, transform=src.transform)

    # Read the data from the specified window
    cropped_data = src.read(window=window)

    # Update the metadata for the cropped GeoTIFF
    meta = src.meta
    meta['width'], meta['height'] = window.width, window.height
    meta['transform'] = src.window_transform(window)
    meta['crs'] = crs
    src.close()
    return cropped_data, meta

def raster_to_gdf(raster, meta, bounds:gpd.GeoSeries=None):
    from shapely.geometry import box
    from shapely import prepare, intersects

    # Extract metadata
    transform = meta['transform']
    height, width = raster.shape[1:]

    # Initialize lists to store geometries and pixel values
    geometries = []
    values = []

    # Iterate over each pixel in the window
    for i in range(height):
        for j in range(width):
            # Get pixel value
            value = raster[0, i, j]  # Accessing the first band assuming single-band raster

            # Calculate pixel coordinates in CRS
            lon, lat = rio.transform.xy(transform, i, j)

            # Create bounding box geometry for the pixel
            minx, miny = rio.transform.xy(transform, i - 0.5, j - 0.5)
            maxx, maxy = rio.transform.xy(transform, i + 0.5, j + 0.5)
            pixel_box = box(minx, miny, maxx, maxy)

            geometries.append(pixel_box)
            values.append(value)

    # Create a GeoDataFrame
    df = gpd.GeoDataFrame({'value': values, 'geometry': geometries}, crs=validate_crs(meta['crs']))

    if bounds is not None:
        bounds = bounds.to_crs(df.crs)
        geoms = list(df.geometry.centroid)
        prepare(geoms)
        df = df.loc[intersects(geoms,bounds.geometry.union_all())].reset_index(drop=True)


    return df
