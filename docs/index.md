# UrbanAccessAnalyzer

Measure how well each street, crossing or H3 cell of an area reaches the places people need: schools, transit stops,
shops, parks or any OpenStreetMap feature.

It works in four steps:

1. **Area** (`AreaOfInterest`): give a place name (geocoded with OpenStreetMap) or a file.
2. **Streets** (`StreetNetwork`): the walk, bike or drive network of the area, built from an `.osm.pbf` file
   (downloaded from Geofabrik when missing).
3. **Places** (`PointsOfInterest`): taken from OpenStreetMap or from your own table, each with a score from 0 to 1.
4. **Access** (`AccessibilityAnalyzer`): for each street and node, the best place reachable within each distance
   step (for example 400, 800 and 1,200 m). Results can be summed up on H3 cells.

## Install

```bash
pip install -e .                  # Python 3.11 or newer
pip install -e ".[plot]"          # maps (folium, ipyleaflet)
pip install -e ".[census]"        # population data through pycensus
pip install -e ".[geohierarchy]"  # put street results onto H3 or polygon layers
```

## Example

```python
from UrbanAccessAnalyzer import AreaOfInterest, StreetNetwork, PointsOfInterest, AccessibilityAnalyzer

aoi = AreaOfInterest.from_name("Cambridge, MA", buffer=500)
network = StreetNetwork.from_pbf("massachusetts.osm.pbf", aoi=aoi, network_type="walk", simplify_distance=30.0)
schools = PointsOfInterest.from_overpass("schools", aoi.gdf)

analyzer = AccessibilityAnalyzer(network, schools)
node_access, edge_access = analyzer.run(distance_matrix=[400, 800, 1200])
streets = analyzer.to_gdf(edge_access)      # scored streets
cells = analyzer.to_h3(resolution=9)        # the same on H3 cells
```

More worked examples are in the `examples/` notebooks (walkability, green areas, rural schools, transit level of service).

## Large areas

For a region or a country, the analyzer can run in H3 chunks so memory stays bounded; see
[`AccessibilityAnalyzer`](api/api.md).

## Population and level of service

`UrbanAccessAnalyzer.population` compares the service offered with the service the population density calls for
(rasters of offer, demand and their difference).
