"""Street scores + population on H3 hexagons, and population statistics.

    hexes = h3_with_population(edges_gdf, aoi, columns=["access_score"], resolution=9, population="worldpop")
    population_by_band(hexes, "access_score")

Needs the optional ``geohierarchy`` and ``pycensus`` packages.
"""

from __future__ import annotations

from typing import Optional, Sequence

import geopandas as gpd
import pandas as pd


def h3_with_population(
    edges_gdf: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    columns: Sequence[str],
    resolution: int = 9,
    population: str = "worldpop",
    year: int = 2025,
    folder: Optional[str] = "population_data",
    kids_only: bool = False,
) -> gpd.GeoDataFrame:
    """H3 hexagons over ``aoi`` with the mean street score of each ``columns`` and the population.

    Args:
        edges_gdf: Street edges with the score ``columns``.
        aoi: Study area.
        columns: Score columns to average (weighted by street length) inside each hexagon.
        resolution: H3 resolution (9 is ~0.1 km2, 10 is ~0.015 km2).
        population: ``"worldpop"`` (anywhere in the world) or ``"census"`` (the country's own
            census; an error if it is not implemented, see :func:`pycensus.census_countries`).
        year: Year of the WorldPop data.
        folder: Where population downloads are kept.
        kids_only: Population under 18 only (WorldPop only).

    Returns:
        GeoDataFrame with ``h3``, ``columns``, ``population`` (people) and ``geometry``, one row per
        hexagon intersecting ``aoi``.
    """
    try:
        from geohierarchy import GeoHierarchy, Mean, Sum
        from geohierarchy.utils import h3_cells
        from pycensus import get_population
    except ImportError as e:
        raise ImportError("Needs the geohierarchy and pycensus packages (see the install cell).") from e

    columns = list(columns)
    aoi = aoi.to_crs(4326)
    hierarchy = GeoHierarchy()
    hierarchy.add_level("h3", h3_cells(aoi, resolution=resolution), id_col="h3")
    # Streets are lines: average their score inside each hexagon, weighted by length.
    hierarchy.add_vector_data(edges_gdf, level="h3", columns=columns, agg=Mean(), geoweight_by="length")
    # Population is a count: add it up inside each hexagon.
    people = get_population(aoi, source=population, year=year, folder=folder, kids_only=kids_only)
    hierarchy.add_vector_data(people, level="h3", columns=["population"], agg=Sum(), geoweight_by="area")

    hexes = hierarchy["h3"]
    hexes = hexes[hexes.intersects(aoi.union_all())].reset_index(drop=True)
    hexes["population"] = hexes["population"].fillna(0).round().astype(int)
    return hexes[["h3", *columns, "population", "geometry"]]


def population_by_band(
    hexes: gpd.GeoDataFrame,
    score: str,
    edges: Sequence[float] = (0, 20, 40, 60, 80, 100),
    labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """People per score band: a table with ``population`` and ``population %`` (best band last).

    Args:
        hexes: Output of :func:`h3_with_population`.
        score: Score column to split by.
        edges: Band limits; a hexagon belongs to the band ``[edge_i, edge_i+1)`` (the last includes its end).
        labels: Band names (default ``"0-20"``, ...).
    """
    edges = list(edges)
    labels = list(labels) if labels is not None else [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])]
    bands = pd.cut(hexes[score], bins=[*edges[:-1], edges[-1] + 1e-9], right=False, labels=labels, include_lowest=True)
    return population_by_group(hexes, bands.rename("band")).rename(columns={"band": score})


def population_by_group(hexes: gpd.GeoDataFrame, groups: pd.Series) -> pd.DataFrame:
    """People per group (any label per hexagon, e.g. a mode): ``population`` and ``population %``."""
    name = groups.name or "group"
    stats = hexes.groupby(groups.rename(name), observed=False)["population"].sum().reset_index()
    total = stats["population"].sum()
    stats["population %"] = (stats["population"] * 100 / total).round(1) if total else 0.0
    return stats


def rank_to_label(ranks: pd.Series, names: Sequence[str]) -> pd.Series:
    """Names for averaged ranks: rank 1 -> ``names[0]``, 2 -> ``names[1]``, ... (rounded, kept in range).

    Use it to turn the mean of a categorical score (stored as 1, 2, 3...) back into a label per hexagon.
    """
    idx = ranks.round().clip(1, len(names)).astype("Int64")
    return idx.map({i + 1: n for i, n in enumerate(names)})
