import geopandas as gpd
from pathlib import Path
from typing import Union
import numpy as np
import shapely
from rasterio.transform import xy, Affine, array_bounds
from pyproj import Transformer
import warnings 
import numpy as np
import rasterio as rio
from rasterio.crs import CRS
import rasterio.warp
from rasterio.warp import Resampling, calculate_default_transform, transform_bounds
from rasterio.features import rasterize as rio_rasterize
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.merge import merge as rio_merge
from rasterio.transform import from_origin
import geopandas as gpd
from pathlib import Path
from typing import Union, Tuple, Optional, List
from pyproj import Transformer
import math
from scipy.ndimage import convolve
from skimage.morphology import disk # For density_raster kernel
from rasterio.mask import mask


# Define a common WGS84 CRS object for convenience
WGS84_CRS = CRS.from_epsg(4326)


def validate_crs(src:rio.io.DatasetReader|rio.io.DatasetWriter|dict|CRS|str|int):
    import re
    if type(src) is dict:
        crs = src['crs']
    elif type(src) is CRS:
        crs = src
    elif type(src) is str:
        crs = CRS.from_string(src)
    elif type(src) is int:
        crs = CRS.from_epsg(src)
    else:
        try:
            crs = src.crs 
        except:
            raise Exception(f"src type {type(src)} not accepted: {src}")

    warnings.filterwarnings(
        "ignore",
        message=re.escape(
            "You will likely lose important projection information when converting to a PROJ string from another format. See: https://proj.org/faq.html#what-is-the-best-format-for-describing-coordinate-reference-systems"
        ),
        category=UserWarning
    )
    if len(crs.to_proj4()) == 0:
        crs_str = crs.to_wkt()
        
        if "LOCAL_CS" in crs_str:
            if "ETRS89-extended / LAEA Europe" in crs_str:
                crs = CRS.from_epsg(3035)

                if (type(src) == rio.io.DatasetReader) and (src.mode != 'r'):
                    src.crs = crs

                return crs
            # Add more mappings as needed
            # elif "Another projection" in crs_str:
            #     return CRS.from_epsg(some_epsg_code)
            else:
                raise ValueError("Unknown LOCAL_CS definition; manual intervention needed.")
        else:
            raise ValueError("CRS is invalid, but not due to LOCAL_CS.")
    else:
        return crs.to_epsg() # to_proj4()
    

def extract_affine_params(transform: Union[Affine, np.ndarray]) -> Tuple[float, float, float, float, float, float]:
    """
    Extracts the six affine transformation parameters (a, b, c, d, e, f).

    This utility handles both rasterio.Affine objects and 3x3 NumPy arrays,
    which are common ways to represent affine transformations in GIS workflows.

    The affine transformation maps pixel coordinates (col, row) to map
    coordinates (x, y) as follows:
    x = a * col + b * row + c
    y = d * col + e * row + f

    Parameters
    ----------
    transform : rasterio.Affine or np.ndarray
        The input affine transformation.

    Returns
    -------
    tuple
        A tuple of six float values: (a, b, c, d, e, f).
        
    Raises
    ------
    ValueError
        If the transform format is not a rasterio.Affine or a 3x3 NumPy array.
    """
    if isinstance(transform, Affine):
        return transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    elif isinstance(transform, np.ndarray) and transform.shape == (3, 3):
        # The 3x3 matrix is of the form: [ [a, b, c], [d, e, f], [0, 0, 1] ]
        # NumPy array elements are typically already float if used for affine transforms.
        a, b, c = transform[0]
        d, e, f = transform[1]
        return float(a), float(b), float(c), float(d), float(e), float(f)
    else:
        raise ValueError("Unrecognized transform format. Must be rasterio.Affine or a 3x3 NumPy array.")

def reproject(
    data: np.ndarray,
    transform: Affine,
    crs: CRS,
    src_nodata: Optional[Union[int, float]] = None,
    dst_nodata: Optional[float] = np.nan,
    dst_transform = None,
    dst_crs: Optional[Union[CRS,int,str]] = 'utm',
    width:int|None=None,
    height:int|None=None
) -> Tuple[np.ndarray, Affine, CRS]:
    """
    Reprojects a raster to its local UTM zone while preserving the pixel grid.
    Returns float64 array with dst_nodata.
    """
    # Determine target UTM CRS using raster center
    left, bottom, right, top = array_bounds(data.shape[0], data.shape[1], transform)
    if (dst_crs is None) or (dst_crs == 'utm') or ('project' in str(dst_crs)):
        raster_geom = gpd.GeoSeries([shapely.geometry.box(left, bottom, right, top)], crs=crs)
        dst_crs = raster_geom.estimate_utm_crs()  # projected CRS

    if height is None:
        height = data.shape[0]

    if width is None: 
        width = data.shape[1]

    if dst_transform is None:
        # Calculate aligned projected transform
        dst_transform, width, height = calculate_default_transform(
            crs, dst_crs, width, height, left, bottom, right, top
        )

    # Allocate destination array
    dst_array = np.empty((height, width), dtype=np.float64)

    # Reproject
    rasterio.warp.reproject(
        source=data,
        destination=dst_array,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata
    )

    # Fill any remaining src_nodata
    if src_nodata is not None:
        dst_array[dst_array == src_nodata] = dst_nodata

    return dst_array, dst_transform, dst_crs


def reproject_global(
    data: np.ndarray,
    transform: rasterio.Affine,
    crs: CRS,
    src_nodata: Optional[Union[int, float]] = None,
    dst_nodata: Optional[float] = np.nan,
    dst_crs: Union[CRS,int,str] = 'EPSG:3857',  # Web Mercator by default
    resolution: float = 1000.0  # pixel size in meters
) -> Tuple[np.ndarray, rasterio.Affine, CRS]:
    """
    Reprojects a raster to a fixed global Web Mercator grid, ensuring perfect alignment
    with other rasters reprojected this way.

    Parameters
    ----------
    data : np.ndarray
        Source raster array.
    transform : Affine
        Source affine transform.
    crs : CRS
        Source CRS.
    src_nodata : int|float, optional
        Source nodata value.
    dst_nodata : float, optional
        Destination nodata value.
    dst_crs : CRS/int/str
        Target CRS (default EPSG:3857).
    resolution : float
        Pixel size in target CRS units (default 1000 meters).

    Returns
    -------
    dst_array : np.ndarray
        Reprojected raster array.
    dst_transform : Affine
        Destination affine transform.
    dst_crs : CRS
        Destination CRS.
    """
    dst_crs = CRS.from_user_input(dst_crs)

    # Compute bounds of source raster in destination CRS
    left, bottom, right, top = transform_bounds(crs, dst_crs, *rasterio.transform.array_bounds(data.shape[0], data.shape[1], transform))

    # Snap bounds to fixed global grid aligned to 0,0 origin
    left = np.floor(left / resolution) * resolution
    bottom = np.floor(bottom / resolution) * resolution
    right = np.ceil(right / resolution) * resolution
    top = np.ceil(top / resolution) * resolution

    width = int(np.round((right - left) / resolution))
    height = int(np.round((top - bottom) / resolution))

    # Create transform aligned to global grid
    dst_transform = from_origin(left, top, resolution, resolution)

    # Allocate destination array
    dst_array = np.empty((height, width), dtype=np.float64)

    # Reproject
    rasterio.warp.reproject(
        source=data,
        destination=dst_array,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata
    )

    # Replace source nodata with destination nodata if needed
    if src_nodata is not None:
        dst_array[dst_array == src_nodata] = dst_nodata

    return dst_array, dst_transform, dst_crs


def raster_crop(
    src_input: Union[str, Path, rio.io.DatasetReader],
    aoi: Optional[Union[gpd.GeoDataFrame, gpd.GeoSeries]] = None,
    nodata: float = np.nan,
    projected: bool = False
) -> Tuple[np.ndarray, Affine, CRS, int, int]:
    _src = None
    if isinstance(src_input, (str, Path)):
        file_path = Path(src_input)
        if not file_path.exists():
            raise FileNotFoundError(f"Raster file not found: {file_path}")
        _src = rio.open(file_path)
    elif isinstance(src_input, rio.io.DatasetReader):
        _src = src_input
    else:
        raise TypeError("src_input must be a path or an open rasterio dataset.")

    try:
        current_crs = _src.crs
        current_src_nodata = _src.nodata if _src.nodata is not None else np.nan

        if aoi is None:
            # Only if no AOI: read full raster (may be huge!)
            current_data = _src.read(1, masked=True)
            current_data = current_data.astype('float64')
            current_data = np.where(current_data.mask, nodata, current_data.data)
            current_transform = _src.transform
        else:
            # Unify geometry
            if isinstance(aoi, gpd.GeoDataFrame):
                geom_to_crop = aoi.geometry.to_crs(current_crs)
            elif isinstance(aoi, gpd.GeoSeries):
                geom_to_crop = aoi.to_crs(current_crs)
            else:
                raise TypeError("AOI must be a GeoDataFrame or GeoSeries.")

            if geom_to_crop.is_empty.any():
                raise ValueError("AOI is empty")

            # Convierte la geometría a una lista de shapes
            shapes = [geom for geom in geom_to_crop]

            # Usa rasterio.mask.mask para recortar según la geometría exacta del AOI
            current_data, current_transform = mask(
                _src,
                shapes=shapes,
                crop=True,
                filled=False,
                nodata=current_src_nodata,
            )
            current_data = current_data[0]  # Toma la primera banda
            current_data = current_data.astype('float64')
            current_data = np.where(current_data.mask, nodata, current_data.data)

        # Optional: reproject to UTM or other CRS
        if projected:
            data, transform, crs = reproject(
                current_data,
                current_transform,
                current_crs,
                src_nodata=nodata,
                dst_nodata=nodata,
                dst_crs='utm'
            )
        else:
            data = current_data
            transform = current_transform
            crs = current_crs

        height, width = data.shape
        return data, transform, crs, height, width
    finally:
        if isinstance(src_input, (str, Path)) and _src is not None:
            _src.close()


def read_raster(path: Union[str, Path], aoi: Optional[Union[gpd.GeoDataFrame, gpd.GeoSeries]] = None, nodata: float = np.nan, projected:bool=False) -> Tuple[np.ndarray, Affine, CRS]:
    """
    Reads a raster from a given path, optionally crops it to an AOI, and
    reprojects it to UTM if its native CRS is geographic.
    The output raster data will always be of float64 dtype with np.nan representing nodata.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the GeoTIFF raster file.
    aoi : Optional[Union[gpd.GeoDataFrame, gpd.GeoSeries]], optional
        Area of Interest to crop the raster. If None, the entire raster is processed.

    Returns
    -------
    Tuple[np.ndarray, Affine, CRS]
        A tuple containing:
        - data (np.ndarray): The raster data (cropped, reprojected, float64, nan for nodata).
        - transform (Affine): The affine transform of the output raster.
        - crs (CRS): The CRS of the output raster (UTM if source was geographic).
    """
    # raster_crop now handles reprojection, cropping, and consistent nodata/dtype output
    data, transform, crs, _, _ = raster_crop(path, aoi, nodata=nodata, projected=projected)
    
    return data, transform, crs


def vectorize(
    raster_array: np.ndarray,
    transform: Affine,
    crs: CRS,
    aoi: Optional[Union[gpd.GeoDataFrame, gpd.GeoSeries]] = None,
    min_value: Optional[float] = None,
    keep_nodata: bool = False,
    nodata: Optional[Union[float, int, str]] = None,
) -> gpd.GeoDataFrame:
    """
    Vectorizes a raster numpy array (float, int, or str), returning a GeoDataFrame of pixel polygons
    with their corresponding raster values. Polygons are in EPSG:4326.

    Nodata handling rules:
    - float arrays: nodata defaults to np.nan
    - int arrays: nodata can be np.nan (array cast to float internally)
    - str arrays: nodata defaults to np.nan, BUT '' is always treated as nodata regardless

    Parameters
    ----------
    raster_array : np.ndarray
        The input raster data array. Can be float, int, or str.
    transform : Affine
        The affine transform of the raster_array.
    crs : CRS
        The CRS of the raster_array.
    aoi : Optional[gpd.GeoDataFrame | gpd.GeoSeries], optional
        Area of interest. If provided, only pixels intersecting the AOI are included.
    min_value : Optional[float], optional
        Threshold for numeric arrays. If keep_nodata=True, values <= min_value become None,
        else they are excluded. Ignored for str arrays.
    keep_nodata : bool, default=False
        If True, nodata pixels (NaN, explicit nodata, or '') are included with value=None.
    nodata : float | int | str, optional
        Explicit nodata value (overrides defaults, except '' is *always* nodata for str).

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with columns:
        - 'id': unique pixel identifier
        - 'geometry': pixel polygon in EPSG:4326
        - 'value': raster value or None
    """
    dtype_kind = raster_array.dtype.kind

    # Normalize int arrays → float so np.nan works
    if np.issubdtype(raster_array.dtype, np.integer):
        raster_array = raster_array.astype(float)
        dtype_kind = "f"

    # Default nodata handling
    if nodata is None:
        if dtype_kind == "f":  # float
            nodata = np.nan
        elif dtype_kind in {"U", "S", "O"}:  # string-like
            nodata = np.nan  # but '' is always nodata too

    rows, cols = np.indices(raster_array.shape)

    # Base mask (exclude nodata unless keep_nodata)
    if keep_nodata:
        valid_mask = np.ones_like(rows, dtype=bool)
    else:
        if dtype_kind == "f":  # float
            valid_mask = ~np.isnan(raster_array)
        elif dtype_kind in {"U", "S", "O"}:  # string
            mask_nan = raster_array == nodata if nodata is not np.nan else np.zeros_like(raster_array, dtype=bool)
            mask_empty = raster_array == ""  # always nodata
            valid_mask = ~(mask_nan | mask_empty)
        else:  # numeric (already cast ints to float, handled above)
            valid_mask = raster_array != nodata

        if min_value is not None and np.issubdtype(raster_array.dtype, np.number):
            valid_mask &= raster_array > min_value

    valid_rows, valid_cols = np.where(valid_mask)
    if valid_rows.size == 0:
        return gpd.GeoDataFrame({'id': [], 'value': []}, geometry=[], crs="EPSG:4326")

    # AOI filtering
    if aoi is not None:
        x_c, y_c = xy(transform, valid_rows + 0.5, valid_cols + 0.5, offset="center")
        points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_c, y_c, crs=crs))
        aoi_in_raster = aoi.to_crs(crs)
        idx = points_gdf.sjoin(aoi_in_raster, how="inner", predicate="intersects").index
        valid_rows, valid_cols = valid_rows[idx], valid_cols[idx]

        if valid_rows.size == 0:
            return gpd.GeoDataFrame({'id': [], 'value': []}, geometry=[], crs="EPSG:4326")

    # Pixel bounds
    x_min, y_max = xy(transform, valid_rows, valid_cols, offset="ul")
    x_max, y_min = xy(transform, valid_rows + 1, valid_cols + 1, offset="ul")

    # Reproject to EPSG:4326
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x0, y0 = transformer.transform(x_min, y_min)
    x1, y1 = transformer.transform(x_max, y_max)

    polygons = shapely.box(x0, y0, x1, y1)

    pixel_ids = valid_rows * raster_array.shape[1] + valid_cols
    pixel_values = raster_array[valid_rows, valid_cols]

    if keep_nodata:
        mask = np.zeros_like(pixel_values, dtype=bool)

        if dtype_kind == "f":
            mask |= np.isnan(pixel_values)
            if min_value is not None:
                mask |= pixel_values <= min_value
        elif dtype_kind in {"U", "S", "O"}:
            mask |= pixel_values == ""
            if isinstance(nodata, str):
                mask |= pixel_values == nodata
        else:
            if nodata is not None:
                mask |= pixel_values == nodata
            if min_value is not None:
                mask |= pixel_values <= min_value

        pixel_values = np.where(mask, None, pixel_values)

    return gpd.GeoDataFrame(
        {"id": pixel_ids, "value": pixel_values},
        geometry=polygons,
        crs="EPSG:4326",
    )


def rasterize(
    gdf: gpd.GeoDataFrame,
    transform: Affine,
    shape: tuple[int, int],
    crs,
    value_column: Optional[str] = None,
    value_order: Optional[List[Union[int, float, str]]] = None,
    background_value: Union[int, float, str] = 0,
    all_touched: bool = True,
) -> np.ndarray:
    """
    Rasterize a GeoDataFrame into a numpy array using a given raster transform and shape.
    Supports int, float, or str rasters. Automatically handles value_order priority and
    string categorical rasters.
    """
    # Handle shape if numpy array passed
    if isinstance(shape, np.ndarray):
        shape = shape.shape

    # Drop invalid geometries
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].copy()

    # Drop rows with NaN or nulls in value_column
    if value_column:
        gdf = gdf[gdf[value_column].notna()].copy()

    # Reproject to target CRS
    gdf = gdf.to_crs(crs)

    # Simple presence raster if no value_column
    if value_column is None:
        raster = rio_rasterize(
            ((geom, 1) for geom in gdf.geometry),
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=all_touched,
            dtype=np.int32
        )
        return raster

    # Determine if value_column is numeric
    is_numeric = True
    try:
        gdf[value_column] = gdf[value_column].astype(float)
    except (ValueError, TypeError):
        is_numeric = False

    # Numeric rasterization
    if is_numeric:
        dtype = np.float32 if gdf[value_column].dtype.kind == 'f' else np.int32
        if value_order is not None:
            raster = np.zeros(shape, dtype=dtype)
            for val in value_order:
                subset = gdf[gdf[value_column] == val]
                if subset.empty:
                    continue
                tmp_raster = rio_rasterize(
                    ((geom, val) for geom in subset.geometry),
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    all_touched=all_touched,
                    dtype=dtype
                )
                mask = raster == 0
                raster[mask & (tmp_raster != 0)] = tmp_raster[mask & (tmp_raster != 0)]
        else:
            raster = rio_rasterize(
                ((geom, val) for geom, val in zip(gdf.geometry, gdf[value_column])),
                out_shape=shape,
                transform=transform,
                fill=0,
                all_touched=all_touched,
                dtype=dtype
            )
        if background_value != 0:
            raster[raster == 0] = background_value
        return raster

    # Non-numeric (string/categorical) rasterization
    if value_order is None:
        value_order = list(gdf[value_column].astype(str).unique())
    value_to_int = {v: i+1 for i, v in enumerate(value_order)}
    int_to_value = {i+1: v for i, v in enumerate(value_order)}

    placeholder_bg = -1 if not isinstance(background_value, str) else background_value
    raster_int = np.full(shape, placeholder_bg, dtype=np.int32)

    for val in value_order:
        subset = gdf[gdf[value_column] == val]
        if subset.empty:
            continue
        geom_value_pairs = ((geom, value_to_int[val]) for geom in subset.geometry)
        tmp_raster = rio_rasterize(
            geom_value_pairs,
            out_shape=shape,
            transform=transform,
            fill=placeholder_bg,
            all_touched=all_touched,
            dtype=np.int32
        )
        mask = raster_int == placeholder_bg
        raster_int[mask & (tmp_raster != placeholder_bg)] = tmp_raster[mask & (tmp_raster != placeholder_bg)]

    # Convert integer raster back to strings
    max_len = max(len(str(v)) for v in value_order)
    raster_final = np.full(shape, '', dtype=f'<U{max_len}')  # empty string for background
    for i_val, orig_val in int_to_value.items():
        raster_final[raster_int == i_val] = orig_val

    return raster_final




def buffer_mean(
    raster: np.ndarray,
    transform: Affine,
    buffer: float = 0
) -> np.ndarray:
    """
    Calculates the density of values in a raster.

    The input `raster` is expected to be in a metric (UTM) CRS.
    Density is calculated as `value / area`. If a buffer is specified,
    a moving window average is applied using a circular kernel.

    Parameters
    ----------
    raster : np.ndarray
        The input raster data array (e.g., population counts). Expected to be in a metric CRS.
    transform : Affine
        The affine transform of the raster.
    buffer : float, optional
        Radius of the buffer in meters for density calculation.
        If 0, density is calculated per pixel. If > 0, a moving average
        is computed over a circular window of this radius. Defaults to 0.

    Returns
    -------
    np.ndarray
        A numpy array representing the density raster (e.g., population density per km²).
    """
    a, b, c, d, e, f = extract_affine_params(transform)
    # Pixel width is 'a', pixel height is 'e' (typically negative).
    # Area of one pixel is |a * e|. Convert to km².
    cell_area_sq_m = abs(a * e)
    if cell_area_sq_m == 0:
        raise ValueError("Cannot calculate density: pixel area is zero. Check raster transform.")
    cell_area_sq_km = cell_area_sq_m / (1000 * 1000) # Convert m² to km²

    if buffer == 0:
        # Simple density: value per pixel area
        return raster / cell_area_sq_km
    else:
        # Convert buffer radius from meters to pixels
        # Assuming isotropic pixels (approx) for simplicity of kernel creation
        # Using abs(a) for pixel width (x-resolution)
        pixel_size = abs(a) 
        if pixel_size == 0:
            raise ValueError("Cannot calculate buffer density: pixel size is zero.")
        
        radius_pixels = buffer / pixel_size
        
        # Create a circular kernel (disk)
        # ceil to ensure the kernel covers the requested buffer distance
        kernel_radius_int = math.ceil(radius_pixels)
        kernel = disk(kernel_radius_int)
        # Sum of kernel elements for normalization
        # This will be used to get the average value over the kernel area
        kernel_sum_pixels = kernel.sum()

        if kernel_sum_pixels == 0:
             raise ValueError("Kernel has zero sum, cannot compute density. Buffer or pixel size might be too small.")

        # Apply convolution to smooth/average the raster values
        # The result will be the sum of raster values within the kernel
        convolved_raster = convolve(raster, kernel, mode='constant', cval=0.0)
        
        # The convolved_raster now contains the sum of values within the kernel for each pixel.
        # To get density, we divide this sum by the *actual area* covered by the kernel.
        # Area covered by kernel = number of pixels in kernel * area of one pixel.
        area_covered_by_kernel = kernel_sum_pixels * cell_area_sq_km
        
        if area_covered_by_kernel == 0:
            raise ValueError("Area covered by kernel is zero, cannot compute density. Buffer or pixel size might be too small.")

        pop_density = convolved_raster / area_covered_by_kernel
        return pop_density


def merge(input_paths, bounds: gpd.GeoSeries = None, method='max'):
    # Open all input GeoTIFF files
    if isinstance(input_paths[0], str):
        src = [rio.open(path) for path in input_paths]
    else:
        src = input_paths

    crs = validate_crs(src[0])

    # Prepare bounds if given
    merge_bounds = tuple(bounds.to_crs(crs).total_bounds) if bounds is not None else None

    # Merge using built-in max method
    mosaic, out_trans = rio_merge(
        src,
        bounds=merge_bounds,
        method=method,
        masked=True          # converts nodata to masked automatically
    )

    # # Handle single-band color palette
    # if (src[0].count == 1) and (src[0].colorinterp[0] == rio.enums.ColorInterp.palette):
    #     mosaic = colormap_to_rgb(mosaic, src[0].colormap(1))

    for s in src:
        s.close()

    # Get raster dimensions
    _, width, height = mosaic.shape

    # Calculate bounds using array_bounds
    # img_bounds = rio.transform.array_bounds(width, height, out_trans)
    # img_bounds = gpd.GeoSeries(shapely.geometry.box(*img_bounds), crs=crs)

    return mosaic, out_trans, crs
