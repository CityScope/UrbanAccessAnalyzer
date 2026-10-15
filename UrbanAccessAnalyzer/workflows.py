"""Ready-made analyses: one call from a place name to maps and statistics.

    from UrbanAccessAnalyzer import workflows

    result = workflows.walkability("Alcoy, España", pois)   # see grading.py for the POI config
    result.map      # interactive map (also saved as a standalone .html in result.folder/maps/)
    result.stats    # people per score band

Each workflow downloads the streets and the places from OpenStreetMap, scores every street, averages the
scores into H3 hexagons, adds the population and saves everything in ``folder``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import geopandas as gpd
import pandas as pd

from . import api, grading, h3_population, osm_io, plotting, utils


@dataclass
class Result:
    """What a workflow returns."""

    folder: str  # where everything was saved
    aoi: gpd.GeoDataFrame  # study area
    edges: gpd.GeoDataFrame  # scored streets
    hexes: gpd.GeoDataFrame  # H3 hexagons: scores + population
    stats: pd.DataFrame  # people per score band (or per mode)
    map: object  # folium map; saved in ``folder/maps/``
    poi_map: object = None  # map of the POIs (walkability, green areas)
    poi_table: Optional[gpd.GeoDataFrame] = None  # the POIs found: type, name, icon


def _setup(place: Union[str, gpd.GeoDataFrame], folder: Optional[str]):
    """Study area (from a place name or a GeoDataFrame) and output folder."""
    if isinstance(place, str):
        aoi = utils.get_city_geometry(place)
        folder = folder or os.path.join("output", utils.sanitize_filename(place))
    else:
        aoi = place
        if folder is None:
            raise ValueError("folder is required when the area is a GeoDataFrame")
    os.makedirs(folder, exist_ok=True)
    return aoi, folder


def report(
    edges: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    folder: str,
    columns: Sequence[str],
    labels: dict,
    score: str,
    map_file: str,
    h3_resolution: int = 10,
    population: str = "worldpop",
    kids_only: bool = False,
    bands: Optional[Sequence[float]] = None,
    band_labels: Optional[Sequence[str]] = None,
    pois: Optional[gpd.GeoDataFrame] = None,
    categories: Optional[dict] = None,
    categorical_group: Optional[str] = None,
    **map_kwargs,
):
    """Last step of every workflow: hexagons + population, map, and statistics.

    Args:
        edges: Scored streets.
        aoi: Study area.
        folder: Output folder (``maps/<map_file>`` and ``stats.csv`` are written in it).
        columns: Score columns of ``edges`` to average per hexagon.
        labels: ``{column: name shown on hover}`` for the map, ``"population"`` first.
        score: Column that colors the map.
        map_file: File name of the map.
        h3_resolution: H3 resolution.
        population: ``"worldpop"`` or ``"census"`` (see :func:`pycensus.get_population`).
        kids_only: Under-18 population (WorldPop only).
        bands: Score limits for the statistics (default ``0, 20, ... 100``).
        band_labels: Names of the bands.
        pois: POI table drawn as emoji pins.
        categories: ``{label: color}`` when ``score`` is a text column (a mode...).
        categorical_group: Text column of the hexagons to count people by (instead of score bands).

    Returns:
        ``(hexes, stats, map)``.
    """
    hexes = h3_population.h3_with_population(edges, aoi, columns, h3_resolution, population, kids_only=kids_only)
    return finish(hexes, edges, aoi, folder, labels, score, map_file, bands, band_labels, pois, categories,
                  categorical_group, **map_kwargs)


def finish(hexes, edges, aoi, folder, labels, score, map_file, bands=None, band_labels=None, pois=None,
           categories=None, categorical_group=None, **map_kwargs):
    """Map + statistics of already computed hexagons (see :func:`report`)."""
    m = plotting.save_access_map(
        os.path.join(folder, "maps", map_file), hexes, aoi, labels, score, edges=edges, pois=pois,
        categories=categories, **map_kwargs,
    )
    if categorical_group:
        stats = h3_population.population_by_group(hexes, hexes[categorical_group])
    else:
        stats = h3_population.population_by_band(hexes, score, bands or (0, 20, 40, 60, 80, 100), band_labels)
    stats.to_csv(os.path.join(folder, "stats.csv"), index=False)
    return hexes, stats, m


def walkability(
    place: Union[str, gpd.GeoDataFrame],
    pois: Sequence[dict],
    folder: Optional[str] = None,
    download_buffer: float = 1500,
    n_steps: int = 5,
    h3_resolution: int = 10,
    population: str = "worldpop",
    map_file: str = "walkability_map.html",
    label: str = "🚶 Walkability (0-100)",
) -> Result:
    """Walkability: how close are the streets to several types of places (shops, schools, parks...).

    Args:
        place: Place name, or a GeoDataFrame with the study area.
        pois: One config per type of place, see :mod:`UrbanAccessAnalyzer.grading`
            (``name``, ``query``, ``geometry``, ``importance``, ``max_distance``, ``icon``).
        folder: Output folder (default ``output/<place>``).
        download_buffer: Extra meters around the area for streets and POIs (at least the largest ``max_distance``).
        n_steps: Distance steps per POI type (5: scores 100, 80, 60, 40, 20, then 0).
        h3_resolution: H3 resolution (10 is ~0.015 km2).
        population: ``"worldpop"`` or ``"census"``.
        map_file: File name of the map in ``folder/maps/``.
        label: Name of the score on the map.
    """
    aoi, folder = _setup(place, folder)
    aoi_download = utils.buffer_gdf(aoi, download_buffer)
    network = api.StreetNetwork.from_pbf(
        os.path.join(folder, "region.osm.pbf"), aoi=aoi_download, network_type="walk", simplify_distance=30
    )
    poi_gdfs = grading.fetch_pois(pois, aoi_download, crs=network.crs)  # if it fails, run again: Overpass can be busy
    poi_table = grading.poi_table(pois, poi_gdfs)
    poi_map = plotting.save_poi_map(os.path.join(folder, "maps", "points_of_interest.html"), aoi, poi_table)

    edges, _ = grading.grade(network, pois, poi_gdfs, n_steps=n_steps, aoi=aoi)
    edges.to_file(os.path.join(folder, "access_streets.gpkg"))

    labels = {"population": "👥 Population", "access_score": label,
              **{p["name"]: f"{p.get('icon', '📍')} {p['name']} (0-100)" for p in pois}}
    hexes, stats, m = report(
        edges, aoi, folder, [*(p["name"] for p in pois), "access_score"], labels, "access_score", map_file,
        h3_resolution, population, pois=poi_table,
    )
    return Result(folder, aoi, edges, hexes, stats, m, poi_map, poi_table)


def green_areas(
    place: Union[str, gpd.GeoDataFrame],
    folder: Optional[str] = None,
    download_buffer: float = 1500,
    max_distance: float = 1200,
    n_steps: int = 3,
    h3_resolution: int = 10,
    population: str = "worldpop",
    map_file: str = "green_areas_map.html",
    query=osm_io.green_areas,
    icon: str = "🌳",
) -> Result:
    """Access to green space (parks, gardens, forests...) on foot, and who lives near it.

    Args:
        place: Place name, or a GeoDataFrame with the study area.
        folder: Output folder (default ``output/<place>``).
        download_buffer: Extra meters around the area (at least ``max_distance``).
        max_distance: Walking distance (m) beyond which green space does not count (1200 m is ~15 min).
        n_steps: Distance steps (1200 m / 3 = 400 m, ~5 min each).
        h3_resolution: H3 resolution (10 is ~0.015 km2).
        population: ``"worldpop"`` or ``"census"``.
        map_file: File name of the map in ``folder/maps/``.
        query: Where the green areas come from (default OpenStreetMap; or your own file with
            :func:`UrbanAccessAnalyzer.grading.query_from_file`).
        icon: Emoji for the green areas on the map.
    """
    aoi, folder = _setup(place, folder)
    pois = [{"name": "green_areas", "query": query, "geometry": "Polygon", "importance": 1,
             "max_distance": max_distance, "icon": icon}]
    aoi_download = utils.buffer_gdf(aoi, download_buffer)
    network = api.StreetNetwork.from_pbf(
        os.path.join(folder, "region.osm.pbf"), aoi=aoi_download, network_type="walk", simplify_distance=30
    )
    poi_gdfs = grading.fetch_pois(pois, aoi_download, crs=network.crs)
    poi_table = grading.poi_table(pois, poi_gdfs)
    edges, _ = grading.grade(network, pois, poi_gdfs, n_steps=n_steps, aoi=aoi)
    edges.to_file(os.path.join(folder, "access_streets.gpkg"))

    minutes = [round(max_distance / n_steps * k / 80) for k in range(1, n_steps + 1)]  # ~4.8 km/h = 80 m/min
    limits = [0, 1, *(100 * k / n_steps for k in range(1, n_steps)), 100]
    band_labels = ["farther", *(f"within ~{m} min" for m in reversed(minutes))]
    labels = {"population": "👥 Population", "access_score": f"{icon} Access to green space (0-100)"}
    hexes, stats, m = report(
        edges, aoi, folder, ["access_score"], labels, "access_score", map_file, h3_resolution, population,
        bands=limits, band_labels=band_labels, pois=poi_table,
    )
    return Result(folder, aoi, edges, hexes, stats, m, None, poi_table)


def school_access(
    place: Union[str, gpd.GeoDataFrame],
    folder: Optional[str] = None,
    download_buffer: float = 0,
    distance_steps: Sequence[float] = (1000, 2500, 10000),
    modes: Sequence[str] = ("🚶 walk", "🚲 bike", "🚌 bus/car"),
    mode_colors: Sequence[str] = ("#2e7d32", "#f9a825", "#c62828"),
    query=osm_io.schools,
    icon: str = "🏫",
    h3_resolution: int = 10,
    population: str = "worldpop",
    kids_only: bool = False,
    map_file: str = "school_access_map.html",
) -> Result:
    """Which mode (walk, bike, bus/car) reaches the nearest school from every street, and who lives there.

    Args:
        place: Place name, or a GeoDataFrame with the study area.
        folder: Output folder (default ``output/<place>``).
        download_buffer: Extra meters around the area (use at least ``max(distance_steps)`` if schools
            outside the area also serve it).
        distance_steps: Distance to the nearest school (closest first), one per mode.
        modes: Name of the mode that reaches each distance (emoji + name).
        mode_colors: Color of each mode on the map.
        query: Where the schools come from (default OpenStreetMap; or your own file with
            :func:`UrbanAccessAnalyzer.grading.query_from_file`).
        icon: Emoji for the schools on the map.
        h3_resolution: H3 resolution (10 is ~0.015 km2).
        population: ``"worldpop"`` or ``"census"``.
        kids_only: Only people under 18 (WorldPop only).
        map_file: File name of the map in ``folder/maps/``.
    """
    aoi, folder = _setup(place, folder)
    aoi_download = utils.buffer_gdf(aoi, download_buffer)
    network = api.StreetNetwork.from_pbf(
        os.path.join(folder, "region.osm.pbf"), aoi=aoi_download, network_type="walk+bike+primary",
        simplify_distance=30,
    )
    school = {"name": "schools", "query": query, "geometry": "Point", "importance": 1, "max_distance": 1, "icon": icon}
    schools = grading.fetch_pois([school], aoi_download, crs=network.crs)["schools"]
    poi_table = grading.poi_table([school], {"schools": schools})

    analyzer = api.AccessibilityAnalyzer(network, api.PointsOfInterest.from_gdf(schools))
    _, edge_access = analyzer.run(
        distance_matrix=list(distance_steps), poi_score_col=None, access_score_values=list(modes),
        max_dist=130, verbose=False,
    )
    edges = analyzer.to_gdf(edge_access).rename(columns={"access_score": "mode"})
    edges.to_file(os.path.join(folder, "access_streets.gpkg"))

    # Hexagons average numbers: 1 = first mode (walk) ... n = last mode; turned back into names after.
    edges["rank"] = edges["mode"].map({mode: i for i, mode in enumerate(modes, 1)})
    hexes = h3_population.h3_with_population(edges, aoi, ["rank"], h3_resolution, population, kids_only=kids_only)
    hexes["mode"] = h3_population.rank_to_label(hexes["rank"], list(modes))
    labels = {"population": "👥 Population", "mode": "🚦 Nearest school reachable by"}
    hexes, stats, m = finish(
        hexes, edges, aoi, folder, labels, "mode", map_file, pois=poi_table,
        categories=dict(zip(modes, mode_colors)), categorical_group="mode",
    )
    return Result(folder, aoi, edges, hexes, stats, m, None, poi_table)
