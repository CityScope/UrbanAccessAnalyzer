"""Shared pytest fixtures: a small real OSM PBF extract and a matching AOI.

``sample.osm.pbf`` is a small real-world extract (a few hundred nodes/ways
around London) borrowed from pyogrio's own test fixtures, used here purely
as a fast, deterministic, network-free input for exercising the OSM ingestion
and routing pipeline end to end.
"""

import os

import geopandas as gpd
import pytest
from shapely.geometry import box

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_PBF = os.path.join(FIXTURES_DIR, "sample.osm.pbf")


@pytest.fixture(scope="session")
def sample_pbf_path() -> str:
    return SAMPLE_PBF


@pytest.fixture(scope="session")
def sample_aoi() -> gpd.GeoDataFrame:
    # Bounding box loosely covering the sample PBF's extent (~51.76-51.77N, ~0.23-0.235W).
    return gpd.GeoDataFrame({}, geometry=[box(-0.238, 51.762, -0.226, 51.771)], crs=4326)
