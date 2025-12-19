import re
import unicodedata
import os
from rapidfuzz import process, fuzz
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import geopandas as gpd
from typing import List, Dict
import osmnx as ox

def get_city_geometry(city_name: str) -> gpd.GeoDataFrame:
    """
    Download city boundary geometry from OpenStreetMap.

    Parameters
    ----------
    city_name : str
        Name of the city (e.g., "Berlin, Germany").

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the city boundary polygon in EPSG:4326.
    """
    # Query OSM for place boundary
    gdf = ox.geocode_to_gdf(city_name)

    # Ensure CRS is WGS84
    gdf = gdf.to_crs(epsg=4326)
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
