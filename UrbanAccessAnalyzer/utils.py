"""Geocoding and misc filesystem/text helpers.

Geocoding goes directly against the Nominatim HTTP API (no osmnx dependency):
:func:`geocode` for point results, :func:`get_city_geometry` for full place
boundary polygons via Nominatim's ``polygon_geojson=1`` parameter.

Source: Nominatim search API, https://nominatim.org/release-docs/latest/api/Search/
"""

import re
import unicodedata
import os
from rapidfuzz import process, fuzz
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import geopandas as gpd
from typing import List, Dict, Optional, Union
import shapely
from shapely.geometry import Polygon, MultiPolygon, Point, shape
import requests

USER_AGENT = "UrbanAccessAnalyzer/1.0 (https://github.com/CityScope/UrbanAccessAnalyzer)"  # Nominatim asks for an identifying agent


def _nominatim_get(params: dict, timeout: float = 30.0, retries: int = 3):
    """GET the Nominatim search API, retrying timeouts, rate limits (429) and server errors with a growing wait."""
    import time

    last = None
    for attempt in range(retries):
        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/search", params=params,
                headers={"User-Agent": USER_AGENT}, timeout=timeout,
            )
            if response.status_code == 429 or response.status_code >= 500:
                last = RuntimeError(f"Nominatim answered HTTP {response.status_code}")
                wait = float(response.headers.get("Retry-After", 0) or 0) or 3.0 * (attempt + 1)
            else:
                response.raise_for_status()
                return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last, wait = e, 3.0 * (attempt + 1)
        if attempt < retries - 1:
            time.sleep(wait)
    raise RuntimeError(f"Nominatim did not answer after {retries} attempts: {last}")


def geocode(q, results:int=1, buffer:float=0):
    try:
        r = _nominatim_get({"q": q, "format": "json", "limit": results})
        
        data = r.json()
        
        if not data:
            return None
        
        records = []
        
        for item in data:
            lat = float(item["lat"])
            lon = float(item["lon"])
            
            records.append({
                "query": q,
                "display_name": item["display_name"],
                "lat": lat,
                "lon": lon,
                "geometry": Point(lon, lat)
            })
        
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        if buffer > 0:
            crs = gdf.crs 
            gdf = gdf.to_crs(gdf.estimate_utm_crs())
            gdf.geometry = gdf.geometry.buffer(buffer)
            gdf = gdf.to_crs(crs)
            
        return gdf
    
    except Exception as e:
        print("Error:", e)
        return None

def get_city_geometry(city_name: str, cache_dir: Optional[str] = None) -> gpd.GeoDataFrame:
    """Download a place's boundary polygon from OpenStreetMap via Nominatim.

    Timeouts, rate limits and server errors are retried (3 attempts).

    Args:
        city_name: Place name/query, e.g. ``"Berlin, Germany"``.
        cache_dir: If given, the boundary is saved in ``cache_dir/geocode/`` and read from there next time
            (no request at all, which also avoids Nominatim's rate limits). Delete the file to search again.

    Returns:
        Single-row GeoDataFrame with the place's boundary polygon
        (EPSG:4326).

    Raises:
        ValueError: If no result with a polygon geometry is found.
        RuntimeError: If Nominatim does not answer.
    """
    import hashlib

    cached = None
    if cache_dir:
        key = hashlib.sha1(city_name.strip().lower().encode()).hexdigest()[:16]
        cached = os.path.join(cache_dir, "geocode", f"{key}.geojson")
        if os.path.isfile(cached):
            return gpd.read_file(cached).to_crs(4326)

    response = _nominatim_get({"q": city_name, "format": "jsonv2", "polygon_geojson": 1, "limit": 1})
    results = response.json()
    if not results or "geojson" not in results[0]:
        raise ValueError(f"No boundary polygon found for {city_name!r}.")

    geom = shape(results[0]["geojson"])
    gdf = gpd.GeoDataFrame({"display_name": [results[0]["display_name"]]}, geometry=[geom], crs="EPSG:4326")
    if cached:
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        gdf.to_file(cached, driver="GeoJSON")
    return gdf


def get_geographic_suggestions_from_string(
    query: str,
    user_agent: str = "UrbanAccessAnalyzer",
    max_results: int = 25
) -> Dict[str, List[str]]:
    """
    Suggests all possible country codes, subdivisions, and municipalities
    for a given string using OpenStreetMap's Nominatim service.
    
    This version collects all relevant fields without skipping any.
    Counties are always included in municipalities.
    """
    geolocator = Nominatim(user_agent=user_agent, timeout=10)

    suggested_country_codes = set()
    suggested_subdivision_names = set()
    suggested_municipalities = set()

    try:
        locations = geolocator.geocode(
            query,
            addressdetails=True,
            language='en',
            exactly_one=False,
            limit=max_results
        )
        if locations:
            for location in locations:
                address = location.raw.get('address', {})

                # Country code
                country_code = address.get('country_code')
                if country_code:
                    suggested_country_codes.add(country_code.upper())

                # Collect all possible subdivisions
                for key in ['state', 'province', 'region', 'county']:
                    value = address.get(key)
                    if value:
                        suggested_subdivision_names.add(value)

                # Collect all possible municipalities
                for key in ['city', 'town', 'village', 'county']:
                    value = address.get(key)
                    if value:
                        suggested_municipalities.add(value)

    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geocoding failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return {
        'country_codes': sorted(suggested_country_codes),
        'subdivision_names': sorted(suggested_subdivision_names),
        'municipalities': sorted(suggested_municipalities),
    }

def get_geographic_suggestions_from_aoi(
    aoi: Union[Polygon, MultiPolygon, gpd.GeoDataFrame, gpd.GeoSeries],
    num_points: int = 1,
    user_agent: str = "MobilityDatabaseClient"
) -> Dict[str, List[str]]:
    """Reverse-geocode AOI geometry to suggest country, subdivision, and municipality."""
    import random

    if isinstance(aoi, (gpd.GeoDataFrame, gpd.GeoSeries)):
        if aoi.empty:
            raise ValueError("GeoDataFrame/GeoSeries is empty.")
        target_geometry = aoi.to_crs(4326).unary_union
    elif isinstance(aoi, (Polygon, MultiPolygon)):
        target_geometry = aoi
    else:
        raise TypeError("AOI must be Polygon, MultiPolygon, GeoDataFrame, or GeoSeries.")

    if target_geometry.is_empty:
        raise ValueError("AOI geometry is empty.")

    geolocator = Nominatim(user_agent=user_agent, timeout=10)
    suggested_country_codes = set()
    suggested_subdivision_names = set()
    suggested_municipalities = set()

    points_to_geocode: List[Point] = []
    min_lon, min_lat, max_lon, max_lat = target_geometry.bounds

    if num_points <= 0:
        num_points = 1
    if num_points == 1:
        points_to_geocode.append(target_geometry.representative_point())
    else:
        for _ in range(num_points):
            points_to_geocode.append(Point(random.uniform(min_lon, max_lon), random.uniform(min_lat, max_lat)))

    for i, point in enumerate(points_to_geocode):
        lat, lon = point.y, point.x
        print(f"Reverse geocoding point {i+1}/{len(points_to_geocode)}: ({lat}, {lon})")
        try:
            location = geolocator.reverse((lat, lon), language='en')
            if location and location.raw:
                address = location.raw.get('address', {})
                if cc := address.get('country_code'):
                    suggested_country_codes.add(cc.upper())
                if subdivision := address.get('state') or address.get('province') or address.get('region') or address.get('county'):
                    suggested_subdivision_names.add(subdivision)
                if municipality := address.get('city') or address.get('town') or address.get('village') or address.get('county'):
                    suggested_municipalities.add(municipality)
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"Geocoding failed for point ({lat}, {lon}): {e}")
        except Exception as e:
            print(f"Unexpected error for point ({lat}, {lon}): {e}")

    return {
        'country_codes': sorted(list(suggested_country_codes)),
        'subdivision_names': sorted(list(suggested_subdivision_names)),
        'municipalities': sorted(list(suggested_municipalities))
    }


def get_folder(path: str) -> str | None:
    """
    Returns the directory for a given path.
    - If path is a file (has an extension), returns its parent folder.
    - If path is a folder, returns the normalized folder path.
    - If path is just a filename (e.g. "file.txt"), returns None.
    """
    path = os.path.normpath(path)
    path = os.path.abspath(path)

    # Check if it's a file (has extension)
    if os.path.splitext(path)[1]:
        folder = os.path.dirname(path)
        return folder if folder else None
    else:
        return path


def normalize_text(text):
    """Lowercase + strip accents from a string"""
    text = str(text).lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def sanitize_filename(name: str) -> str:
    """Replaces spaces and invalid filename characters with an underscore."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", normalize_text(name))


def gdf_fuzzy_match(gdf, city_name, column="NAMEUNIT"):
    # Normalize input city name
    norm_city = normalize_text(city_name)

    # Normalize column
    gdf["_match_norm"] = gdf[column].astype(str).apply(normalize_text)

    # Check for exact match first
    exact = gdf[gdf["_match_norm"] == norm_city]
    if not exact.empty:
        return exact.iloc[0:1]

    # Fuzzy match using token_sort_ratio
    choices = gdf["_match_norm"].tolist()
    best_match, score, index = process.extractOne(
        norm_city, choices, scorer=fuzz.token_sort_ratio
    )

    return gdf.iloc[index : index + 1].drop(columns=["_match_norm"])


def buffer_gdf(gdf: gpd.GeoDataFrame, meters: float) -> gpd.GeoDataFrame:
    """One-polygon GeoDataFrame: ``gdf`` merged and grown by ``meters`` (same CRS as ``gdf``)."""
    merged = gpd.GeoDataFrame(geometry=[gdf.union_all()], crs=gdf.crs)
    if not meters:
        return merged
    utm = merged.estimate_utm_crs()
    return gpd.GeoDataFrame(geometry=merged.to_crs(utm).buffer(meters), crs=utm).to_crs(gdf.crs)


def select_file(path: str = "") -> str:
    """Path of a file of the user's, picked automatically where possible.

    - ``path`` given and it exists (local run, or a file already in Colab): used as is.
    - Google Colab otherwise: the upload dialog opens to choose a file from your computer.
    - Local run without a valid ``path``: raises, since there is nothing to pick from.

    Returns:
        The path of the file (uploaded files are saved in the working directory, ``/content``).
    """
    import sys

    if path and os.path.exists(path):
        return path
    if "google.colab" in sys.modules:
        from google.colab import files

        uploaded = files.upload()  # opens the file picker
        if not uploaded:
            raise ValueError("No file was selected.")
        return next(iter(uploaded))
    raise FileNotFoundError(f"File not found: {path!r}. Set the path of your file.")


def aoi_key(aoi: gpd.GeoDataFrame, *parts) -> str:
    """Short stable key of an area plus any settings (``parts``): the name of its cache entry.

    The same area (to ~1 m) and settings always give the same key, so a cache is never reused for a different
    area, even if both are saved under the same place name.
    """
    import hashlib

    geom = shapely.set_precision(aoi.to_crs(4326).union_all(), 1e-5)
    h = hashlib.sha1(shapely.to_wkb(geom))
    for part in parts:
        h.update(repr(part).encode())
    return h.hexdigest()[:12]


def output_folder(place: str, base: str = "output") -> str:
    """The folder where the results of a place are saved (``output/<place>``), created if it does not exist."""
    folder = os.path.join(base, sanitize_filename(place))
    os.makedirs(folder, exist_ok=True)
    return folder


def offer_download(path: str) -> None:
    """In Colab, download a saved map: inline outputs cannot draw the vector background, the file can."""
    import sys

    if "google.colab" not in sys.modules:
        return
    from google.colab import files

    print(f"Colab shows maps without background: open the downloaded {os.path.basename(path)} in your browser.")
    files.download(path)
