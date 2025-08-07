import re
import unicodedata
import os
from rapidfuzz import process, fuzz


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
