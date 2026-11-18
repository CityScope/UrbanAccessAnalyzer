"""Geocoding and misc filesystem/text helpers.

Geocoding goes directly against the Nominatim HTTP API (no osmnx dependency):
:func:`geocode` for point results, :func:`get_city_geometry` for full place
boundary polygons via Nominatim's ``polygon_geojson=1`` parameter.

Source: Nominatim search API, https://nominatim.org/release-docs/latest/api/Search/
"""

import os
from rapidfuzz import process, fuzz
import geopandas as gpd
from shapely.geometry import Point
import requests
from geohierarchy.naming import normalize_text, sanitize_filename  # noqa: F401  (one implementation, in geohierarchy)
from geohierarchy.geocoding import get_city_geometry, get_geographic_suggestions_from_aoi, get_geographic_suggestions_from_string  # noqa: F401  (one implementation, in geohierarchy)

def geocode(q, results:int=1, buffer:float=0):
    """Look a place up on OpenStreetMap Nominatim and return its boundary (or point), optionally buffered.

    Args:
        q: Place name or address.
        results: Number of results to request.
        buffer: Buffer in metres around the result.

    Returns:
        GeoDataFrame in EPSG:4326, or None if nothing was found.
    """
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






def get_folder(path: str) -> str | None:
    """
    Returns the directory for a given path.
    - If path is a file (has an extension), returns its parent folder.
    - If path is a folder, returns the normalized folder path.
    - If path is just a filename (e.g. "file.txt"), returns None.

    Args:
        path: A file or folder path.

    Returns:
        The folder, or None for a bare file name.
    """
    path = os.path.normpath(path)
    path = os.path.abspath(path)

    # Check if it's a file (has extension)
    if os.path.splitext(path)[1]:
        folder = os.path.dirname(path)
        return folder if folder else None
    else:
        return path






def gdf_fuzzy_match(gdf, city_name, column="NAMEUNIT"):
    # Normalize input city name
    """The row whose name best matches ``city_name``: an exact match after removing accents and case, else the closest
    fuzzy match.

    Args:
        gdf: Table of places.
        city_name: Name to look for.
        column: Name column.

    Returns:
        A one-row GeoDataFrame (the best match is always returned).
    """
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
