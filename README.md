# UrbanAccessAnalyzer

Build street networks from OpenStreetMap data and compute multi-tier
accessibility (isochrone-style access scores) to points of interest —
schools, transit stops, green space, groceries, or any other OSM feature —
per street edge, node, or aggregated to H3 hexagons.

The pipeline: an **AreaOfInterest** (geocoded or loaded from file) defines
the study area; a **StreetNetwork** is cropped and built from a local or
Geofabrik-downloaded `.osm.pbf` for a given profile (`walk`, `bike`,
`drive`, `all`, ...), with connectivity/simplification handled internally
(Polars + `scipy.sparse.csgraph`); **PointsOfInterest** are pulled from
Overpass or supplied directly; and an **AccessibilityAnalyzer** computes
per-edge/per-node access scores across configurable distance tiers, either
whole-network or in memory-bounded H3-chunked passes for large areas.

## Install

```bash
pip install -e .

# optional extras
pip install -e ".[plot]"       # matplotlib/folium/ipyleaflet mapping helpers
pip install -e ".[census]"     # WorldPop/country-level population data (pycensus)
pip install -e ".[geohierarchy]"  # aggregating street edges onto H3/other polygon layers
pip install -e ".[dev]"        # pre-commit, pytest, black, ruff
```

Requires Python >= 3.11. This project uses [uv](https://docs.astral.sh/uv/)
for dependency locking (`uv.lock`); `uv sync` works as an alternative to the
`pip install` commands above.

## Basic usage

```python
from UrbanAccessAnalyzer import (
    AreaOfInterest,
    StreetNetwork,
    PointsOfInterest,
    AccessibilityAnalyzer,
)

aoi = AreaOfInterest.from_name("Cambridge, MA", buffer=500)

network = StreetNetwork.from_pbf(
    "massachusetts.osm.pbf",   # downloaded from Geofabrik if missing
    aoi=aoi,
    network_type="walk",
    simplify_distance=30.0,
)

points = PointsOfInterest.from_overpass("schools", aoi.gdf)

analyzer = AccessibilityAnalyzer(network, points)
node_access, edge_access = analyzer.run(distance_matrix=[400, 800, 1200])

access_gdf = analyzer.to_gdf(edge_access)   # scored street-edges GeoDataFrame
access_h3 = analyzer.to_h3(resolution=9)    # aggregated to H3 hexagons
```

For a one-call convenience wrapper around the same pipeline, see
`UrbanAccessAnalyzer.api.compute_accessibility`. See `examples/` for full
notebooks (rural school access, walkability, green space, transit level of
service).

## Tests

```bash
pytest
```

Test fixtures use a small bundled `.osm.pbf` sample (`tests/fixtures/`); no
network access or API keys are required to run the test suite.
