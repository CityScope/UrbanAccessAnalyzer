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
    folder: Optional[str] = "cache/population",
    kids_only: bool = False,
    empty_cells: str = "nearest_value",
    max_distance: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """H3 hexagons over ``aoi`` with the mean street score of each ``columns`` and the population.

    A hexagon with **no street** to sample has no score (``NaN``). ``empty_cells`` chooses what to do with the
    ones that have people (see :func:`geohierarchy.handle_empty_cells`):

    - ``"nearest_value"`` (default): take the scores of the nearest hexagon that has streets.
    - ``"relocate_weight"``: move their people to the nearest hexagon with streets (within 300 m by default;
      farther they are excluded and counted in ``hexes.attrs["excluded_weight"]``).
    - ``"ignore"``: leave them without score (statistics then skip them).

    Args:
        edges_gdf: Street edges with the score ``columns``.
        aoi: Study area.
        columns: Score columns to average (weighted by street length) inside each hexagon.
        resolution: H3 resolution (9 is ~0.1 km2, 10 is ~0.015 km2).
        population: ``"worldpop"`` (anywhere in the world) or ``"census"`` (the country's own
            census; an error if it is not implemented, see :func:`pycensus.census_countries`).
        year: Year of the WorldPop data.
        folder: Where population downloads (WorldPop rasters, census files) are kept and reused.
        kids_only: Population under 18 only (WorldPop only).
        empty_cells: What to do with hexagons that have no street: ``"nearest_value"``, ``"relocate_weight"`` or
            ``"ignore"`` (see above).
        max_distance: Meters, center to center. ``"nearest_value"``: only copy from hexagons at most this far
            (default: no limit). ``"relocate_weight"``: default 300.

    Returns:
        GeoDataFrame with ``h3``, ``columns``, ``population`` (people), ``filled`` (``True`` where the
        scores were taken from the nearest hexagon, ``"nearest_value"``) and ``geometry``, one row per hexagon intersecting ``aoi``.
    """
    try:
        from geohierarchy import GeoHierarchy, Mean, Sum, handle_empty_cells
        from geohierarchy.utils import h3_cells
        from pycensus import get_population
    except ImportError as e:
        raise ImportError("Needs the geohierarchy and pycensus packages (see the install cell).") from e

    columns = list(columns)
    aoi = aoi.to_crs(4326)
    hierarchy = GeoHierarchy()
    hierarchy.add_level("h3", h3_cells(aoi, resolution=resolution), id_col="h3")
    # Streets are lines: average their score inside each hexagon, weighted by length. A constant column
    # tells which hexagons have streets at all (a plain mean gives 0, not "no data", for the others).
    streets = edges_gdf[[*columns, "geometry"]].assign(_has_streets=1.0)
    hierarchy.add_vector_data(streets, level="h3", columns=[*columns, "_has_streets"], agg=Mean(), geoweight_by="length")
    # Population is a count: add it up inside each hexagon.
    people = get_population(aoi, source=population, year=year, folder=folder, kids_only=kids_only)
    hierarchy.add_vector_data(people, level="h3", columns=["population"], agg=Sum(), geoweight_by="area")

    hexes = hierarchy["h3"]
    hexes = hexes[hexes.intersects(aoi.union_all())].reset_index(drop=True)
    hexes["population"] = hexes["population"].fillna(0).round().astype(int)
    hexes.loc[~(hexes["_has_streets"] > 0.5), columns] = float("nan")  # no street: nothing to sample
    hexes = handle_empty_cells(hexes, columns, empty_cells, "population", max_distance)
    if "filled" not in hexes:
        hexes["filled"] = False
    hexes["population"] = hexes["population"].round().astype(int)
    excluded = hexes.attrs.get("excluded_weight", 0.0)
    result = hexes[["h3", *columns, "population", "filled", "geometry"]]
    result.attrs["excluded_weight"] = excluded  # people left out by "relocate_weight"
    return result


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
    idx = ranks.round().clip(1, len(names)).astype("Int64")  # NaN stays empty (no street, no value)
    return idx.map({i + 1: n for i, n in enumerate(names)})
