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
from typing import List, Dict, Union
import shapely
from shapely.geometry import Polygon, MultiPolygon, Point, shape
import requests

def geocode(q, results:int=1, buffer:float=0):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "json", "limit": results},
            headers={"User-Agent": "your_email@example.com"},  # Use a real identifier
            timeout=8
        )
        
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

def get_city_geometry(city_name: str) -> gpd.GeoDataFrame:
    """Download a place's boundary polygon from OpenStreetMap via Nominatim.

    Args:
        city_name: Place name/query, e.g. ``"Berlin, Germany"``.

    Returns:
        Single-row GeoDataFrame with the place's boundary polygon
        (EPSG:4326). Falls back to nothing (raises) if Nominatim returns no
        polygon geometry for the query.

    Raises:
        ValueError: If no result with a polygon geometry is found.
    """
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": city_name, "format": "jsonv2", "polygon_geojson": 1, "limit": 1},
        headers={"User-Agent": "UrbanAccessAnalyzer"},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json()
    if not results or "geojson" not in results[0]:
        raise ValueError(f"No boundary polygon found for {city_name!r}.")

    geom = shape(results[0]["geojson"])
    return gpd.GeoDataFrame({"display_name": [results[0]["display_name"]]}, geometry=[geom], crs="EPSG:4326")


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
