"""UrbanAccessAnalyzer: street networks from OpenStreetMap and access scores to points of interest.

The four main classes, used in this order:

* :class:`AreaOfInterest` -- the study area (geocoded by name or read from a file);
* :class:`StreetNetwork` -- the walk / bike / drive network of that area, built from an ``.osm.pbf`` file;
* :class:`PointsOfInterest` -- schools, stops, shops... from OpenStreetMap or your own table, with a score;
* :class:`AccessibilityAnalyzer` -- how well each street, node or H3 cell reaches those points within given distances.
"""

from .api import (
    AreaOfInterest,
    StreetNetwork,
    PointsOfInterest,
    AccessibilityAnalyzer,
)

__all__ = [
    "AreaOfInterest",
    "StreetNetwork",
    "PointsOfInterest",
    "AccessibilityAnalyzer",
]
