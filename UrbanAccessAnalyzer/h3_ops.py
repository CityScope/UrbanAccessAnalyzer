"""H3-specific utilities: point/polygon rasterization, raster ingest, and cell resampling, in Polars.

Consolidates the previous ``h3_utils.py`` (pandas/geopandas, mature but slow)
and ``h3_polars.py`` (an incomplete polars port whose ``cells_in_geometry``
was pandas code with only the type hint changed, and which was never
imported anywhere else in the package) into one correct implementation.

Edge-to-any-polygon aggregation (the street-edges-to-H3 path previously
covered by :func:`from_df` from :class:`AccessibilityAnalyzer`) now lives in
the optional ``geohierarchy`` dependency, wired in through
:mod:`UrbanAccessAnalyzer.geometry_levels` and
:meth:`UrbanAccessAnalyzer.api.AccessibilityAnalyzer.to_level`/``to_h3`` --
that path works against *any* polygon layer, not just H3 grids. This module
keeps the remaining H3-specific pieces that have no ``geohierarchy``
equivalent: rasterizing points/polygons/rasters directly into H3 cells
(:func:`cells_in_geometry`, :func:`from_df`, :func:`from_raster`, used e.g. by
:class:`UrbanAccessAnalyzer.api.PointsOfInterest`), re-aggregating H3 data
onto a coarser resolution's parent cells (:func:`resample`), and building H3
hexagon boundary geometries (:func:`to_gdf`).

Methodology / sources:

- H3 is Uber's hexagonal hierarchical spatial index:
  https://h3geo.org/docs. Cell ids are opaque 64-bit strings; each
  resolution (0-15) subdivides the previous one into ~7 children.
- Polygon-to-cell rasterization (:func:`cells_in_geometry`) uses
  ``h3.h3shape_to_cells`` (stable "center-of-cell-inside-polygon" containment)
  and ``h3.h3shape_to_cells_experimental`` (H3's newer, still-experimental
  API: https://h3geo.org/docs/library/experimental) for the ``full``,
  ``overlap``, ``bbox_overlap`` and ``center_overlap`` containment
  strategies, since H3 itself has no stable API for those.
  H3 has no bulk/vectorized polyfill entry point, so this step is a
  Python loop over input geometries -- fine here because POI/polygon counts
  are orders of magnitude smaller than the street-graph edge counts the
  rest of the package optimizes for.
- :func:`aggregate` is fully vectorized Polars (explode + group_by/agg
  expression chains): supports ``first/last/min/max/mean/sum`` and two
  spatial-specific reducers, ``density`` (value normalized by cell area in
  m^2, via ``h3.cell_area``) and ``distribute`` (a value split evenly across
  every cell a source geometry overlaps).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Literal, Optional, Union

import h3
import math

import numpy as np
import polars as pl
import shapely

from . import geometry_ops, raster_ops

POLYGON_TYPES = ("Polygon", "MultiPolygon")
POINT_TYPES = ("Point",)
BUFFER_TYPES = ("LineString", "MultiLineString", "LinearRing", "MultiPoint")
GEOMETRY_COLLECTION_TYPE = "GeometryCollection"

ContainMode = Literal["center", "full", "overlap", "bbox_overlap", "centroid", "center_overlap"]


def cells_in_geometry(
    df: pl.DataFrame,
    resolution: int,
    geometry_col: str = "geometry",
    buffer: float = 0.0,
    contain: ContainMode = "center_overlap",
) -> pl.DataFrame:
    """Rasterize WKB geometries into lists of H3 cell ids.

    Args:
        df: Polars DataFrame with a WKB geometry column (projected CRS if
            ``buffer > 0``, EPSG:4326 otherwise -- H3 always operates in
            lon/lat).
        resolution: H3 resolution (0-15); higher is smaller/finer cells.
        geometry_col: Name of the WKB geometry column.
        buffer: Buffer distance (CRS units) applied before rasterization;
            required for line/point geometries to acquire any area.
        contain: Containment rule -- ``"center"`` (cell center inside
            geometry, stable API), ``"full"``/``"overlap"``/``"bbox_overlap"``
            (experimental API), ``"centroid"`` (rasterize each geometry's
            centroid point instead of its shape), or ``"center_overlap"``
            (try ``"center"``, fall back to ``"overlap"`` for geometries that
            match no cell, e.g. slivers smaller than one hexagon).

    Returns:
        ``df`` with an added ``h3_cells (list[str])`` column.
    """
    geoms = shapely.from_wkb(df[geometry_col].to_numpy())

    if buffer > 0:
        geoms = shapely.buffer(geoms, buffer)
    geom_types = shapely.get_type_id(geoms)  # 0=Point,1=LineString,3=Polygon,4=MultiPoint,5=MultiLineString,6=MultiPolygon,7=GeometryCollection

    if contain == "centroid":
        geoms = shapely.centroid(geoms)
        geom_types = shapely.get_type_id(geoms)

    gc_mask = geom_types == 7
    if gc_mask.any():
        geoms[gc_mask] = np.array(
            [shapely.union_all([g for g in geoms[i].geoms if not g.is_empty]) for i in np.nonzero(gc_mask)[0]]
        )
        geom_types = shapely.get_type_id(geoms)

    buffer_mask = np.isin(geom_types, [1, 4, 5])  # LineString, MultiPoint, MultiLineString
    if buffer_mask.any():
        geoms[buffer_mask] = shapely.buffer(geoms[buffer_mask], max(buffer, 1e-6))
        geom_types = shapely.get_type_id(geoms)

    h3_cells: list[list[str]] = [None] * len(geoms)
    polygon_mask = np.isin(geom_types, [3, 6])
    point_mask = geom_types == 0

    contain_mode = "center" if contain == "center_overlap" else ("center" if contain == "centroid" else contain)
    for i in np.nonzero(polygon_mask)[0]:
        shape = h3.geo_to_h3shape(geoms[i])
        if contain_mode == "center":
            cells = h3.h3shape_to_cells(shape, res=resolution)
        else:
            cells = h3.h3shape_to_cells_experimental(shape, res=resolution, contain=contain_mode)
        if contain == "center_overlap" and len(cells) == 0:
            cells = h3.h3shape_to_cells_experimental(shape, res=resolution, contain="overlap")
        h3_cells[i] = cells

    for i in np.nonzero(point_mask)[0]:
        h3_cells[i] = [h3.latlng_to_cell(geoms[i].y, geoms[i].x, res=resolution)]

    return df.with_columns(pl.Series("h3_cells", h3_cells, dtype=pl.List(pl.Utf8)))


def aggregate(
    h3_df: pl.DataFrame,
    columns: Optional[List[str]] = None,
    value_order: Optional[Union[List, Dict[str, List]]] = None,
    method: Union[str, Dict[str, str]] = "max",
    h3_column: str = "h3_cell",
) -> pl.DataFrame:
    """Explode list-valued H3 cell columns and aggregate values per cell.

    Args:
        h3_df: DataFrame with an ``h3_column`` holding either a single cell
            id per row or a ``list[str]`` of cell ids (e.g. from
            :func:`cells_in_geometry`'s ``h3_cells``).
        columns: Columns to aggregate; defaults to every column except
            ``h3_column``.
        value_order: Explicit category ordering for ``first``/``last``/etc.
            on non-numeric columns, either one list shared by all
            ``columns`` or a ``{column: order}`` dict.
        method: Aggregation method per column, one of ``first``, ``last``,
            ``min``, ``max``, ``mean``, ``sum``, ``density`` (requires an
            ``area`` column in m², normalizes by H3 cell area), or
            ``distribute`` (splits a value evenly across every cell a row's
            source geometry maps to).
        h3_column: Name of the cell-id column (renamed to ``h3_cell``
            internally if different).

    Returns:
        Polars DataFrame with one row per distinct H3 cell (``h3_cell``
        column) and one aggregated column per entry in ``columns``.
    """
    columns = columns or []
    method = method if isinstance(method, dict) else {col: method for col in (columns or [c for c in h3_df.columns if c != h3_column])}

    if h3_column != "h3_cell":
        if "h3_cell" in h3_df.columns:
            warnings.warn(f"'h3_cell' column already exists; dropping it and renaming {h3_column!r}.")
            h3_df = h3_df.drop("h3_cell")
        h3_df = h3_df.rename({h3_column: "h3_cell"})

    if "density" in method.values() and "area" not in h3_df.columns:
        # H3-specific: density normalizes by the target cell's own area, not
        # a per-row source area (geometry_ops.aggregate's generic meaning).
        # Cell area only depends on the id itself, so it's cheap to attach
        # per row before delegating to the shared aggregator.
        cell_id_col = pl.col("h3_cell").list.first() if h3_df.schema["h3_cell"] == pl.List(pl.Utf8) else pl.col("h3_cell")
        h3_df = h3_df.with_columns(
            cell_id_col.map_elements(lambda c: h3.cell_area(c, unit="m^2"), return_dtype=pl.Float64).alias("area")
        )

    return geometry_ops.aggregate(h3_df, id_column="h3_cell", columns=columns or None, value_order=value_order, method=method)


def from_df(
    df: pl.DataFrame,
    resolution: int,
    geometry_col: str = "geometry",
    columns: Optional[List[str]] = None,
    value_order: Optional[Union[List, Dict[str, List]]] = None,
    buffer: float = 0.0,
    contain: ContainMode = "center_overlap",
    method: Union[str, Dict[str, str]] = "max",
) -> pl.DataFrame:
    """Rasterize a Polars DataFrame with WKB geometry directly into aggregated H3 cells.

    Combines :func:`cells_in_geometry` and :func:`aggregate`.

    Args:
        df: DataFrame with a WKB ``geometry_col`` column and attribute columns.
        resolution: H3 resolution.
        geometry_col: Name of the WKB geometry column.
        columns: Attribute columns to aggregate.
        value_order: See :func:`aggregate`.
        buffer: Buffer distance applied before rasterization.
        contain: Containment rule, see :func:`cells_in_geometry`.
        method: Aggregation method(s), see :func:`aggregate`. If ``"density"``
            is used for any column, per-row geometry area (m²) is computed
            automatically.

    Returns:
        Polars DataFrame aggregated per H3 cell (see :func:`aggregate`).
    """
    h3_df = cells_in_geometry(df, resolution=resolution, geometry_col=geometry_col, buffer=buffer, contain=contain)
    h3_df = h3_df.drop(geometry_col)
    return aggregate(h3_df, columns=columns, value_order=value_order, method=method, h3_column="h3_cells")


def from_raster(
    raster: Union[np.ndarray, str],
    aoi=None,
    resolution: int = 10,
    contain: ContainMode = "center_overlap",
    method: Union[str, Dict[str, str]] = "distribute",
    value_order: Optional[List] = None,
    transform=None,
    crs=None,
    nodata=None,
) -> pl.DataFrame:
    """Rasterize a raster dataset into aggregated H3 cells.

    Args:
        raster: Either a file path (read via
            :func:`UrbanAccessAnalyzer.raster_ops.read_raster`) or an
            in-memory NumPy array (requires ``transform``/``crs``).
        aoi: AOI to crop to when ``raster`` is a file path.
        resolution: H3 resolution.
        contain: Containment rule, see :func:`cells_in_geometry`.
        method: Aggregation method, see :func:`aggregate`.
        value_order: Explicit ordering for categorical raster values.
        transform: Affine transform (required for array input).
        crs: CRS (required for array input).
        nodata: Nodata value to exclude.

    Returns:
        Polars DataFrame aggregated per H3 cell, from the ``value`` column.
    """
    if isinstance(raster, str):
        raster, transform, crs = raster_ops.read_raster(raster, aoi=aoi, nodata=nodata)
        vec = raster_ops.vectorize(raster, transform=transform, crs=crs, aoi=None, keep_nodata=False, nodata=nodata)
    else:
        if transform is None or crs is None:
            raise ValueError("transform and crs are required when passing a raster array directly.")
        vec = raster_ops.vectorize(raster, transform=transform, crs=crs, aoi=aoi, keep_nodata=False, nodata=None)

    vec = vec.filter(pl.col("value").is_not_null())
    order = {"value": value_order} if value_order is not None else None
    return from_df(vec, resolution=resolution, columns=["value"], value_order=order, contain=contain, method=method)


def from_raster_centroid(
    raster_path: str,
    aoi=None,
    resolution: int = 10,
    agg: Union[str, "AggregationStrategy"] = "sum",
    row_chunk: int = 500,
) -> pl.DataFrame:
    """Map each raster pixel's *center* straight to its H3 cell, in bounded-size row-bands.

    A much cheaper alternative to :func:`from_raster` for rasters whose
    pixels are comparable to or larger than the target H3 resolution's
    cells (e.g. a ~100m WorldPop pixel vs. an H3 resolution-10 cell, ~140m
    across): :func:`from_raster` -> :func:`UrbanAccessAnalyzer.raster_ops.vectorize`
    turns *every pixel* into an explicit Shapely polygon before H3-
    polyfilling it, which scales with total pixel count and can exhaust
    memory on a large AOI (confirmed via repeated real OOM kills on a
    full-metro WorldPop crop at resolution 10). This instead computes each
    pixel's center coordinate via plain affine arithmetic on NumPy arrays
    (no geometry objects at all) and maps it to an H3 cell with
    ``h3ronpy.vector.coordinates_to_cells`` -- one vectorized, Arrow-native
    call per row-band.

    For a summable ``agg`` (the population use case), a pixel is never
    mapped straight to ``resolution`` if that would be coarser than the
    pixel itself. Centroid-only assignment has a real error mode: a pixel
    whose *center* falls in cell A but whose square footprint partially
    overlaps neighbor cell B contributes 100% of its value to A and 0% to
    B. That error's *absolute* size scales with the assignment
    resolution's cell size, so assigning directly at a coarse target
    resolution (a big cell) can distort that cell's density by more than a
    couple percent. Instead, this picks the H3 resolution whose average
    cell area (:func:`h3.average_hexagon_area`) is closest to the pixel's
    own area, assigns pixels there (bounding the mismatch to that much
    smaller cell size), then aggregates up to ``resolution`` via
    :func:`resample` (``method="sum"``) -- an exact, purely index-based
    parent/child rollup with no additional approximation, so every target
    cell's total is the sum of exactly the fine cells nested under it.
    ``sum(value)`` is therefore identical whether pixels are assigned
    directly at ``resolution`` or via this finer intermediate step; only
    the *per-cell* distribution (density) accuracy improves.

    Args:
        raster_path: Path to a raster file (cropped to ``aoi`` internally
            via :func:`UrbanAccessAnalyzer.raster_ops.read_raster`, which
            uses an exact polygon mask, not just a bounding-box crop, and
            reprojected to a metric CRS so pixel area is measured in true
            m^2 rather than degrees).
        aoi: Area of interest to crop the raster to.
        resolution: Target H3 resolution.
        agg: Aggregation across pixels landing in the same cell -- a
            :class:`~geohierarchy.aggregation.AggregationStrategy` (e.g.
            ``Sum()``, ``Mean()``, ``Max()``) if the optional
            ``geohierarchy`` dependency is installed, or one of the plain
            string reducers polars' ``group_by(...).agg`` understands
            (``"sum"``, ``"mean"``, ``"max"``, ...) otherwise. Only a sum
            (the default) gets the pixel-matched-resolution treatment
            above -- summing nests exactly across resolutions, but a mean
            or max computed at a fine resolution is not equivalent to the
            same reducer computed directly at a coarser one, so non-sum
            aggregations are assigned directly at ``resolution``.
        row_chunk: Number of raster rows processed per batch. Bounds peak
            memory to one row-band's worth of pixel coordinates/H3 cells
            regardless of the overall raster size.

    Returns:
        Polars DataFrame with ``h3_cell`` and ``value`` columns, one row
        per H3 cell within ``resolution``'s grid with at least one
        contributing pixel.
    """
    import h3
    import h3ronpy
    import h3ronpy.vector as h3v
    import pyproj

    # `projected=False` (the default) is required for correctness, not just
    # convenience: `projected=True` reprojects the *data array itself*, which
    # resamples/interpolates pixel values and does not exactly preserve their
    # sum (confirmed: a real run dropped from 4,969,238 to 4,447,181 total
    # population purely from that reprojection step). Pixel area for the
    # resolution-matching below is instead computed geodesically from the
    # untouched raster's degree-sized transform, never touching `array`.
    array, transform, crs = raster_ops.read_raster(raster_path, aoi=aoi)
    height, width = array.shape[-2], array.shape[-1]
    to_wgs84 = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    agg_expr = _resolve_pixel_agg(agg)

    is_sum = isinstance(agg, str) and agg == "sum"
    try:
        from geohierarchy import Sum as _Sum

        is_sum = is_sum or isinstance(agg, _Sum)
    except ImportError:
        pass

    # `max(resolution, best_matching)` only ever picks a *finer* assign resolution
    # than requested (handling "target cells bigger than the pixel" by assigning
    # fine-then-summing-up, which `resample` can do exactly). It silently did
    # nothing for the opposite, equally real case -- a target resolution *finer*
    # than the pixel itself (e.g. a ~97m WorldPop pixel vs. an H3 res-11 cell,
    # ~46m across) -- where it degenerated to plain centroid assignment: every
    # pixel landed in whichever single fine cell contained its center, leaving
    # every *other* fine cell physically under that same pixel with zero
    # population. Those cells then failed the "has population" test used
    # elsewhere to decide which cells even appear on the map, so real,
    # populated, street-adjacent areas silently vanished. `oversample_n`
    # subdivides each such pixel into an n x n sub-point grid (population split
    # evenly, so the total is still exact) instead of collapsing it to one point.
    assign_resolution = resolution
    oversample_n = 1
    if is_sum:
        pixel_area_m2 = _pixel_area_m2(transform, crs, height, width)
        best_res = _best_matching_resolution(pixel_area_m2)
        if best_res > resolution:
            assign_resolution = best_res
        else:
            target_area_m2 = h3.average_hexagon_area(resolution, unit="m^2")
            oversample_n = max(1, math.ceil(math.sqrt(pixel_area_m2 / target_area_m2)))

    parts: List[pl.DataFrame] = []
    for row_off in range(0, height, row_chunk):
        row_end = min(row_off + row_chunk, height)
        block = array[..., row_off:row_end, :]
        finite = np.isfinite(block)
        if not finite.any():
            continue

        rows, cols = np.nonzero(finite)
        values = block[rows, cols]
        # Pixel-center coordinates in the raster's own (projected) CRS, via plain
        # affine arithmetic on NumPy arrays -- no per-pixel geometry of any kind.
        if oversample_n > 1:
            n = oversample_n
            sub = (np.arange(n) + 0.5) / n
            off_r, off_c = np.meshgrid(sub, sub, indexing="ij")
            off_r, off_c = off_r.ravel(), off_c.ravel()
            rows = np.repeat(rows, n * n) + np.tile(off_r, len(values))
            cols = np.repeat(cols, n * n) + np.tile(off_c, len(values))
            values = np.repeat(values, n * n) / (n * n)
        else:
            rows = rows + 0.5
            cols = cols + 0.5
        px = transform.a * cols + transform.b * (rows + row_off) + transform.c
        py = transform.d * cols + transform.e * (rows + row_off) + transform.f
        lon, lat = to_wgs84.transform(px, py)

        cells = h3v.coordinates_to_cells(np.asarray(lat), np.asarray(lon), assign_resolution)
        # `h3ronpy`'s output is an arro3 Array; wrapping it directly in a `pl.Series`/
        # DataFrame looks fine on the surface but triggers pathological (multi-GB,
        # thread-pool-breaking) memory blowup the moment it hits a threaded polars
        # operation like `group_by` (confirmed via a minimal 80,974-point repro:
        # >12GB and a crash raw, vs. 104MB and 8ms materialized). `.to_pylist()`
        # first avoids whatever zero-copy/foreign-buffer path causes that.
        cell_list = h3ronpy.cells_to_string(cells).to_pylist()
        part = pl.DataFrame({"h3_cell": cell_list, "value": values})
        parts.append(part.group_by("h3_cell").agg(agg_expr))

    if not parts:
        return pl.DataFrame({"h3_cell": [], "value": []}, schema={"h3_cell": pl.Utf8, "value": pl.Float64})
    fine = pl.concat(parts).group_by("h3_cell").agg(agg_expr)

    if assign_resolution == resolution:
        return fine
    return resample(fine, target_resolution=resolution, columns=["value"], method="sum")


def _pixel_area_m2(transform, crs, height: int, width: int) -> float:
    """Approximate area of one raster pixel in m^2, without reprojecting/touching the data array.

    If `crs` is already projected (metric), this is exact:
    `abs(transform.a * transform.e)`. If geographic (degrees, the common
    case for WorldPop), pixel width/height are converted at the raster
    window's center latitude via a geodesic (`pyproj.Geod`) calculation --
    accurate to a fraction of a percent for a single pixel, and exact
    enough for *choosing a matching H3 resolution*, which only needs to be
    right to within a factor of a few.
    """
    if crs is not None and crs.is_projected:
        return abs(transform.a * transform.e)

    import pyproj

    center_col, center_row = width / 2.0, height / 2.0
    lon0 = transform.a * center_col + transform.b * center_row + transform.c
    lat0 = transform.d * center_col + transform.e * center_row + transform.f
    lon1 = transform.a * (center_col + 1) + transform.b * center_row + transform.c
    lat1 = transform.d * center_col + transform.e * (center_row + 1) + transform.f

    geod = pyproj.Geod(ellps="WGS84")
    _, _, width_m = geod.inv(lon0, lat0, lon1, lat0)
    _, _, height_m = geod.inv(lon0, lat0, lon0, lat1)
    return abs(width_m * height_m)


def _best_matching_resolution(pixel_area_m2: float, max_resolution: int = 15) -> int:
    """H3 resolution whose average cell area is closest (in log-space) to `pixel_area_m2`."""
    import h3

    best_r, best_diff = 0, float("inf")
    for r in range(max_resolution + 1):
        diff = abs(np.log(h3.average_hexagon_area(r, unit="m^2")) - np.log(pixel_area_m2))
        if diff < best_diff:
            best_r, best_diff = r, diff
    return best_r


def _resolve_pixel_agg(agg: Union[str, "AggregationStrategy"]) -> pl.Expr:
    """Turn `agg` into the single polars aggregation expression `from_raster_centroid` group-by's on."""
    if isinstance(agg, str):
        return getattr(pl.col("value"), agg)().alias("value")
    # A `geohierarchy.aggregation.AggregationStrategy` (e.g. Sum()/Mean()/Max()) --
    # its `upscale_aggs` already returns exactly this kind of expression list.
    exprs = agg.upscale_aggs(["value"])
    return exprs[0]


def resample(
    df: pl.DataFrame,
    target_resolution: int,
    columns: Optional[List[str]] = None,
    value_order: Optional[Dict] = None,
    method: Union[str, Dict[str, str]] = "max",
    h3_column: str = "h3_cell",
) -> pl.DataFrame:
    """Re-aggregate H3 cell data onto a coarser resolution's parent cells.

    Args:
        df: DataFrame with an ``h3_column`` of H3 cell ids.
        target_resolution: Target (coarser) H3 resolution.
        columns: Columns to aggregate; see :func:`aggregate`.
        value_order: See :func:`aggregate`.
        method: Aggregation method(s); see :func:`aggregate`.
        h3_column: Name of the cell-id column.

    Returns:
        Polars DataFrame aggregated at ``target_resolution``.
    """
    try:
        import h3ronpy

        parent_cells = h3ronpy.cells_to_string(
            h3ronpy.change_resolution(h3ronpy.cells_parse(df[h3_column]), target_resolution)
        )
        # `pl.Series(name, values)` silently ignores `name` when `values` is an
        # arro3/Arrow array (an interop quirk) and the resulting Series ends up named
        # `""` instead -- `.with_columns()` then *adds* a bogus empty-named column
        # rather than replacing `h3_column`, leaving cells at their original
        # resolution with no error raised. `.alias(...)` after construction forces
        # the name unambiguously.
        df = df.with_columns(pl.Series(parent_cells).alias(h3_column))
    except ImportError:
        # `h3ronpy` (an optional transitive dep, e.g. via the `geohierarchy` extra) gives an
        # Arrow-native vectorized resolution change -- ~2-3x faster than the per-cell Python
        # loop below, verified to produce identical output to `h3.cell_to_parent`. Fall back
        # to the always-available `h3` package if it isn't installed.
        df = df.with_columns(
            pl.col(h3_column).map_elements(lambda c: h3.cell_to_parent(c, target_resolution), return_dtype=pl.Utf8).alias(h3_column)
        )
    return aggregate(df, columns=columns, value_order=value_order, method=method, h3_column=h3_column)


def to_gdf(df: pl.DataFrame, h3_column: str = "h3_cell"):
    """Build hexagon polygon geometries for a Polars H3 cell table.

    Args:
        df: DataFrame with an ``h3_column`` of valid H3 cell ids.
        h3_column: Name of the cell-id column.

    Returns:
        ``geopandas.GeoDataFrame`` (EPSG:4326) with the hexagon boundary
        geometry for each cell -- this is the package's I/O boundary for H3
        data (matches street-edge output, which is also handed back as a
        GeoDataFrame at the API layer).
    """
    import geopandas as gpd

    if df.is_empty():
        return gpd.GeoDataFrame(df.to_pandas(), geometry=[], crs=4326)

    valid = df[h3_column].map_elements(h3.is_valid_cell, return_dtype=pl.Boolean)
    df = df.filter(valid)
    if df.is_empty():
        raise ValueError("No valid H3 cells in dataframe.")

    # Shanghai OOM fix (2026-09-01, corrected again): two earlier attempts
    # here didn't work -- (1) chunking `df.to_pandas()` alongside geometry
    # and `pd.concat`-ing the parts back together OOM'd at the concat step
    # itself (13 built parts already at 21.5GB, then concat needs to build
    # a same-size merged result while they're still referenced); (2) a
    # single unchunked `df.to_pandas()` call also OOM'd, because `df` (the
    # polars source) and its freshly-built pandas copy are BOTH fully
    # resident for the duration of that one call -- a real, unavoidable
    # doubling for that specific operation. The actual fix: shrink the
    # *window* where both copies coexist, rather than trying to eliminate
    # the doubling itself -- extract the (cheap, single-column) h3 id
    # series BEFORE converting, convert to pandas, then `del df`
    # immediately so the polars source is freed before geometry
    # construction (`_cell_polygons`'s own internal chunking, unrelated to
    # this) even starts, instead of both large objects being simultaneously
    # alive through the whole function.
    h3_series = df[h3_column]
    pandas_df = df.to_pandas()
    del df
    polygons = _cell_polygons(h3_series)
    result = gpd.GeoDataFrame(pandas_df, geometry=polygons, crs="EPSG:4326")
    return result


# Shanghai OOM fix (2026-08-31): a single unchunked `cells_to_wkb_polygons` +
# `shapely.from_wkb` call on a real ~25.7M-cell grid transiently needs
# >16GB RSS (measured via a watchdog-guarded repro against 25,732,258 real
# Shanghai-area H3 cells: OOM-killed at a 16GB cap with no completion print) --
# h3ronpy's Arrow WKB buffer for the whole batch, its polars-numpy conversion,
# and shapely's decoded output array are all transiently resident together at
# that scale, on top of everything else already alive in the pipeline at that
# point. Processing in bounded slices and concatenating the (much cheaper --
# each element is just an 8-byte object pointer) results keeps steady-state
# memory flat regardless of input size: the same repro chunked at 2M cells/
# batch peaked at 13.1GB retaining *all* 25,732,258 final Polygon objects (vs.
# OOMing unchunked before even finishing), in 39.7s, producing byte-identical
# WKB per cell to the unchunked path (same underlying h3ronpy call, just on
# slices). Small/medium cities (well under this threshold) take exactly one
# chunk and pay zero extra overhead -- this is not a behavior change for them.
_CELL_POLYGON_CHUNK_SIZE = 2_000_000


def _cell_polygons(cells: pl.Series):
    """Hexagon boundary polygons for a Series of H3 cell ids, as a numpy object array.

    Prefers ``h3ronpy``'s Arrow-native, fully vectorized cell -> WKB polygon
    conversion (verified to produce geometries identical to the
    ``h3.cell_to_boundary`` loop below), which on a 1.6M-cell resolution-11
    metro grid is ~6x faster than the per-cell Python loop (~2s vs ~19s) and
    allocates far less transient memory, since neither the coordinate tuples
    nor the intermediate per-cell Python objects are ever materialized.
    Falls back to the always-available ``h3`` package when ``h3ronpy`` (an
    optional transitive dependency) isn't installed.

    Processes `cells` in `_CELL_POLYGON_CHUNK_SIZE`-sized slices when the
    h3ronpy path is used and the input is larger than that -- see the module-
    level note on `_CELL_POLYGON_CHUNK_SIZE` for why: a single vectorized call
    over tens of millions of cells transiently spikes far above the steady-
    state memory the same work costs when done in bounded batches.
    """
    try:
        import h3ronpy
        import h3ronpy.vector as h3ronpy_vector
    except ImportError:
        return [shapely.Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)]) for cell in cells.to_list()]

    import warnings

    def _decode(batch: pl.Series):
        wkb = h3ronpy_vector.cells_to_wkb_polygons(h3ronpy.cells_parse(batch.to_arrow()))
        with warnings.catch_warnings():
            # h3ronpy hands back a `geoarrow.wkb`-typed Arrow array; polars has no
            # registered extension type for it and warns as it falls back to the
            # plain binary storage type -- which is exactly what's wanted here,
            # since the bytes go straight into `shapely.from_wkb`.
            warnings.simplefilter("ignore", UserWarning)
            return shapely.from_wkb(pl.Series(wkb).to_numpy())

    n = len(cells)
    if n <= _CELL_POLYGON_CHUNK_SIZE:
        return _decode(cells)

    import numpy as np

    decoded = []
    for i in range(0, n, _CELL_POLYGON_CHUNK_SIZE):
        decoded.append(_decode(cells[i : i + _CELL_POLYGON_CHUNK_SIZE]))
    result = np.concatenate(decoded)
    return result
