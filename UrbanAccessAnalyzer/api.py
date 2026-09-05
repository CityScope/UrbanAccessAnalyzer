"""Public API: AreaOfInterest, StreetNetwork, PointsOfInterest, AccessibilityAnalyzer.

This is the single consolidated user-facing surface for the package,
replacing the previous ``UrbanAccess.py`` and the divergent, partially
duplicate ``objects.py`` (whose census-linked classes, ``MultiresPolygonData``
and ``AccessScore``, are dropped along with the census module -- this
package now only produces results as a street-edges GeoDataFrame or an H3
Polars table, per its scope as a general OSM accessibility engine).

Pipeline these classes wire together, all internally Polars/scipy/shapely
(see the module docstrings of :mod:`osm_io`, :mod:`graph_ops`,
:mod:`routing`, :mod:`isochrones`, :mod:`h3_ops` for the methodology of each
stage):

    AreaOfInterest -> StreetNetwork (osm_io + graph_ops)
                    -> PointsOfInterest (osm_io + scoring)
    StreetNetwork + PointsOfInterest -> AccessibilityAnalyzer (isochrones)
                                      -> .to_gdf() / .to_h3()
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import polars as pl
import shapely

from . import geometry_levels, graph_ops, h3_ops, isochrones, osm_io, routing, scoring, utils


class AreaOfInterest:
    """A geographic area of interest, backing every other class's spatial extent."""

    def __init__(self, gdf: gpd.GeoDataFrame):
        """Wrap an existing AOI GeoDataFrame.

        Prefer :meth:`from_name` or :meth:`from_file` over calling this
        directly.

        Args:
            gdf: AOI geometry, any CRS.
        """
        self.gdf = gdf

    @classmethod
    def from_name(cls, name: str, buffer: float = 0.0) -> "AreaOfInterest":
        """Geocode a place name into an AOI via Nominatim.

        Args:
            name: Place name/query, e.g. ``"Cambridge, MA"``.
            buffer: Optional buffer (meters, applied in local UTM) around the
                geocoded geometry.

        Returns:
            A new :class:`AreaOfInterest`.
        """
        gdf = utils.get_city_geometry(name)
        if buffer > 0:
            crs = gdf.crs
            gdf = gdf.to_crs(gdf.estimate_utm_crs())
            gdf["geometry"] = gdf.geometry.buffer(buffer)
            gdf = gdf.to_crs(crs)
        return cls(gdf)

    @classmethod
    def from_file(cls, path: str, bounds=None) -> "AreaOfInterest":
        """Load an AOI from any vector file format.

        Args:
            path: File path (see
                :func:`UrbanAccessAnalyzer.geometry_ops.read_geofile` for
                supported formats).
            bounds: Optional bbox filter.

        Returns:
            A new :class:`AreaOfInterest`.
        """
        from . import geometry_ops

        return cls(geometry_ops.read_geofile(path, bounds=bounds))

    def plot(self, m=None, **kwargs):
        """Draw this AOI's outline on a Folium map.

        Args:
            m: Existing ``folium.Map`` to draw onto; a new one is created if
                omitted.
            **kwargs: Passed through to
                :func:`UrbanAccessAnalyzer.plotting.general_map`.

        Returns:
            The ``folium.Map``.
        """
        from . import plotting

        return plotting.general_map(m=m, aoi=self.gdf, **kwargs)


class StreetNetwork:
    """A street network for routing, backed by Polars node/edge tables.

    Internally just two Polars DataFrames (``nodes``, ``edges``) in a
    projected metric CRS -- see :mod:`UrbanAccessAnalyzer.osm_io` and
    :mod:`UrbanAccessAnalyzer.graph_ops` for how they're built and
    manipulated. No networkx/osmnx graph object exists anywhere in this
    class.
    """

    def __init__(self, nodes: pl.DataFrame, edges: pl.DataFrame, crs: str):
        self.nodes = nodes
        self.edges = edges
        self.crs = crs

    @classmethod
    def from_pbf(
        cls,
        pbf_path: str,
        aoi: Optional[AreaOfInterest] = None,
        network_type: str = "walk",
        simplify_distance: Optional[float] = None,
        ignore_oneway: bool = False,
        crop_buffer_m: float = 500.0,
    ) -> "StreetNetwork":
        """Build a street network from a local (or Geofabrik-downloaded) PBF file.

        Args:
            pbf_path: Path to a ``.osm.pbf`` file. If it doesn't exist and
                ``aoi`` is given, it's downloaded from Geofabrik first
                (:func:`UrbanAccessAnalyzer.osm_io.download_geofabrik`).
            aoi: AOI used both for downloading (if needed) and for bbox
                prefiltering during load.
            network_type: One of ``"walk"``, ``"bike"``, ``"drive"``,
                ``"all"``, ``"walk+bike"``, ``"walk+bike+primary"``.
            simplify_distance: If given, run
                :func:`UrbanAccessAnalyzer.graph_ops.simplify` with this
                cluster distance (meters) right after loading.
            ignore_oneway: Forwarded to :func:`UrbanAccessAnalyzer.osm_io.load_pbf`
                -- if True, every kept way gets edges in both directions
                regardless of its OSM ``oneway`` tag (see that function's
                docstring). ``network_type="walk"`` already ignores oneway
                unconditionally; this only matters for other profiles
                (e.g. ``"all"``) used for walking-access purposes.
            crop_buffer_m: Metres to buffer ``aoi`` by before cropping the
                network to it (2026-09-04, explicit user request: "crop_by_
                aoi_connected here you should use the aoi with a buffer" --
                a buffer is only ever for avoiding a stop-cropping/network-
                boundary effect, never for anything population-related,
                which stays on the real unbuffered AOI elsewhere in the
                pipeline). Cropping a street network to the EXACT AOI edge
                can cut a real street mid-block right at the boundary,
                producing an artificial dead end and, in the worst case,
                stranding an otherwise-real stop/destination just outside
                the crop -- a small buffer keeps a margin of real network
                around the AOI so nothing legitimate right at the edge gets
                severed. Set to ``0`` for the old exact-AOI behavior.

        Returns:
            A new :class:`StreetNetwork`, projected to a local UTM CRS.
        """
        if not os.path.isfile(pbf_path):
            if aoi is None:
                raise ValueError("aoi is required to download a PBF when pbf_path does not exist.")
            pbf_path = osm_io.download_geofabrik(aoi.gdf, output_folder=os.path.dirname(pbf_path) or ".")

        nodes, edges = osm_io.load_pbf(
            pbf_path, aoi=aoi.gdf if aoi else None, network_type=network_type, ignore_oneway=ignore_oneway
        )
        nodes, edges, crs = graph_ops.project(nodes, edges)
        if aoi is not None:
            # 2026-09-04: this briefly built a separate wider "all roads"
            # graph here specifically to bridge a restricted `"walk"`
            # profile's connectivity gaps -- reverted the same day
            # (explicit user request: "delete this idea of excluding
            # highway and only include edges really needed for the graph.
            # Include all public roads regardless if they are walk or
            # not"). `network_type` defaults to `"all"` again (see
            # `transitlos.network.prepare_street_network`), so `nodes`/
            # `edges` already ARE the widest available graph -- no
            # separate wider-profile fallback needed; `crop_by_aoi_connected`
            # still repairs connectivity per simple polygon (falling back
            # to each polygon's largest connected component) even without
            # a wider profile to bridge from.
            # `crop_by_aoi_connected` (not plain `crop_by_aoi`): repairs any
            # connectivity fragmentation the crop introduces, per simple
            # polygon of `aoi` (a real, isolated island stays isolated; a
            # spurious cut from cropping does not). Buffered by
            # `crop_buffer_m` -- see that parameter's docstring.
            # Dissolve `aoi.gdf`'s rows into one geometry FIRST. `aoi.gdf`
            # is often several rows (e.g. one per municipality) whose
            # polygons individually fragment further into many slivers
            # (Concepcion: 11 municipality rows, ~992 constituent polygons
            # once exploded) -- if those were handed to
            # `crop_by_aoi_connected` as-is, each row/sliver would be
            # treated as an independent connectivity domain even where two
            # rows share a real border (e.g. Penco touching Concepcion,
            # Talcahuano, Tome), silently dropping real streets near every
            # such internal seam once they end up in a smaller-than-largest
            # component with no wider profile left to rescue them (bug
            # found live 2026-09-04: Penco streets missing on the map).
            # Union first so only genuinely disjoint pieces of the AOI
            # (real islands/exclaves) remain separate polygons after the
            # explode step inside `crop_by_aoi_connected`.
            crop_aoi_gdf = gpd.GeoDataFrame(geometry=[aoi.gdf.to_crs(crs).union_all()], crs=crs)
            if crop_buffer_m:
                crop_aoi_gdf["geometry"] = crop_aoi_gdf.geometry.buffer(crop_buffer_m)
            nodes, edges = graph_ops.crop_by_aoi_connected(nodes, edges, crop_aoi_gdf)
        if simplify_distance:
            nodes, edges = graph_ops.simplify(nodes, edges, cluster_distance=simplify_distance)
        return cls(nodes, edges, crs)

    def simplify(self, cluster_distance: float, protected_node_ids: Optional[list[int]] = None) -> "StreetNetwork":
        """Cluster-simplify this network's topology; see :func:`UrbanAccessAnalyzer.graph_ops.simplify`.

        Args:
            cluster_distance: Max edge length used to chain nodes into a cluster.
            protected_node_ids: Node ids to keep as singleton clusters.

        Returns:
            A new, simplified :class:`StreetNetwork`.
        """
        nodes, edges = graph_ops.simplify(self.nodes, self.edges, cluster_distance, protected_node_ids)
        return StreetNetwork(nodes, edges, self.crs)

    def crop(self, aoi: AreaOfInterest, crop_buffer_m: float = 500.0) -> "StreetNetwork":
        """Crop this network to an AOI; see :func:`UrbanAccessAnalyzer.graph_ops.crop_by_aoi_connected`.

        Args:
            aoi: Area to crop to.
            crop_buffer_m: Metres to buffer ``aoi`` by first -- see
                :meth:`from_pbf`'s ``crop_buffer_m`` docstring (a buffer is
                only ever for avoiding a network-boundary effect, never for
                anything population-related). ``0`` for the exact-AOI
                behavior.
        """
        # See the matching comment in `from_pbf`: union `aoi.gdf`'s rows
        # into one geometry before buffering/exploding, so touching rows
        # (e.g. adjoining municipalities) aren't treated as independent
        # connectivity domains.
        crop_aoi_gdf = gpd.GeoDataFrame(geometry=[aoi.gdf.to_crs(self.crs).union_all()], crs=self.crs)
        if crop_buffer_m:
            crop_aoi_gdf["geometry"] = crop_aoi_gdf.geometry.buffer(crop_buffer_m)
        nodes, edges = graph_ops.crop_by_aoi_connected(self.nodes, self.edges, crop_aoi_gdf)
        return StreetNetwork(nodes, edges, self.crs)

    def snap_points(self, points: "PointsOfInterest", max_dist: Optional[float] = None, min_edge_length: float = 1.0):
        """Snap POI points onto this network, splitting edges as needed.

        Args:
            points: Points to snap. Reprojected to this network's CRS first
                if ``points.crs`` is set and differs from it (e.g. POIs
                fetched via :meth:`PointsOfInterest.from_overpass`, which are
                always EPSG:4326, snapped onto a network in a projected UTM
                CRS).
            max_dist: Maximum snap distance.
            min_edge_length: See :func:`UrbanAccessAnalyzer.graph_ops.snap_points`.

        Returns:
            Tuple ``(StreetNetwork, points_with_node_id)`` -- the updated
            network and ``points.df`` with a new ``node_id`` column.
        """
        points = _reproject_points(points, self.crs)
        geoms = shapely.from_wkb(points.df["geometry"].to_numpy())
        nodes, edges, node_ids = graph_ops.snap_points(self.nodes, self.edges, geoms, max_dist, min_edge_length)
        df = points.df.with_columns(pl.Series("node_id", node_ids))
        return StreetNetwork(nodes, edges, self.crs), PointsOfInterest(df, crs=self.crs)

    def to_gdf(self) -> gpd.GeoDataFrame:
        """Export edges as a GeoDataFrame."""
        geom = shapely.from_wkb(self.edges["geometry_wkb"].to_numpy())
        return gpd.GeoDataFrame(self.edges.drop("geometry_wkb").to_pandas(), geometry=geom, crs=self.crs)

    def save(self, folder: str) -> None:
        """Persist nodes/edges as Parquet.

        Args:
            folder: Destination directory (created if missing). Writes
                ``nodes.parquet``, ``edges.parquet``, and a small
                ``meta.txt`` with the CRS.
        """
        os.makedirs(folder, exist_ok=True)
        self.nodes.write_parquet(os.path.join(folder, "nodes.parquet"))
        self.edges.write_parquet(os.path.join(folder, "edges.parquet"))
        with open(os.path.join(folder, "meta.txt"), "w") as f:
            f.write(self.crs)

    @classmethod
    def load(cls, folder: str) -> "StreetNetwork":
        """Load a network previously written by :meth:`save`."""
        nodes = pl.read_parquet(os.path.join(folder, "nodes.parquet"))
        edges = pl.read_parquet(os.path.join(folder, "edges.parquet"))
        with open(os.path.join(folder, "meta.txt")) as f:
            crs = f.read().strip()
        return cls(nodes, edges, crs)

    def plot(self, m=None, **kwargs):
        """Draw this network's edges on a Folium map."""
        from . import plotting

        return plotting.general_map(m=m, gdfs=[self.to_gdf().to_crs(4326)], **kwargs)


class PointsOfInterest:
    """A set of scored points of interest, backed by a Polars DataFrame with WKB geometry."""

    def __init__(self, df: pl.DataFrame, crs: Optional[str] = None):
        """Wrap an existing POI DataFrame.

        Args:
            df: POI table with a WKB ``geometry`` column, and any tag/score
                columns.
            crs: CRS of ``df``'s ``geometry`` column, e.g. ``"EPSG:4326"``.
                Optional and ``None`` by default (e.g. when a caller builds
                points directly in a street network's own projected CRS, as
                the test suite does) -- but set it whenever the geometry's
                CRS is known and may differ from the network it will be run
                against, so :meth:`StreetNetwork.snap_points` and
                :meth:`AccessibilityAnalyzer.run` can reproject correctly
                instead of silently mixing coordinate systems.
        """
        self.df = df
        self.crs = crs

    @classmethod
    def from_overpass(cls, kind: str, bounds: gpd.GeoDataFrame) -> "PointsOfInterest":
        """Fetch a named OSM POI category via Overpass.

        Args:
            kind: One of the convenience functions in
                :mod:`UrbanAccessAnalyzer.osm_io`: ``"schools"``,
                ``"healthcare"``, ``"groceries"``, ``"shops"``,
                ``"restaurants"``, ``"libraries"``, ``"pharmacies"``,
                ``"gyms"``, ``"cinemas"``, ``"bus_stops"``, ``"green_areas"``.
            bounds: AOI GeoDataFrame defining the query extent.

        Returns:
            A new :class:`PointsOfInterest`, with geometry in EPSG:4326 (see
            :func:`UrbanAccessAnalyzer.osm_io.overpass_query`).

        Raises:
            ValueError: For an unrecognized ``kind``.
        """
        fn = getattr(osm_io, kind, None)
        if fn is None:
            raise ValueError(f"Unknown POI kind {kind!r}; see UrbanAccessAnalyzer.osm_io for available queries.")
        return cls(fn(bounds), crs="EPSG:4326")

    def assign_score_by_values(self, column: str, value_priority: list, out_column: str = "poi_score") -> "PointsOfInterest":
        """Assign a ``[0, 1]`` score from a categorical column; see :func:`UrbanAccessAnalyzer.scoring.score_by_values`."""
        scores = scoring.score_by_values(self.df[column].to_list(), value_priority)
        return PointsOfInterest(self.df.with_columns(pl.Series(out_column, scores)), crs=self.crs)

    def to_h3(self, resolution: int, columns: Optional[list] = None, method: Union[str, dict] = "max") -> pl.DataFrame:
        """Rasterize these points to H3 cells; see :func:`UrbanAccessAnalyzer.h3_ops.from_df`."""
        return h3_ops.from_df(self.df, resolution=resolution, columns=columns, method=method)

    def to_gdf(self) -> gpd.GeoDataFrame:
        """Export as a GeoDataFrame, in ``self.crs`` (EPSG:4326 if unset)."""
        geom = shapely.from_wkb(self.df["geometry"].to_numpy())
        return gpd.GeoDataFrame(self.df.drop("geometry").to_pandas(), geometry=geom, crs=self.crs or 4326)


def _reproject_points(points: "PointsOfInterest", target_crs: str) -> "PointsOfInterest":
    """Reproject a :class:`PointsOfInterest`'s geometry to ``target_crs`` if needed.

    Every routing/snapping step below (:func:`UrbanAccessAnalyzer.graph_ops.snap_points`,
    :func:`UrbanAccessAnalyzer.isochrones.graph`) works on raw x/y coordinates
    with no CRS awareness of its own, so a mismatch between POI geometry
    (typically EPSG:4326 straight out of :meth:`PointsOfInterest.from_overpass`)
    and a network in a projected metric CRS would silently snap/measure
    distances against the wrong scale instead of raising. This is a no-op
    when ``points.crs`` is unset (the caller is assumed to already have
    matched CRSes, as in hand-built test fixtures) or already equal to
    ``target_crs``.

    Args:
        points: Points to (maybe) reproject.
        target_crs: Destination CRS, normally a :class:`StreetNetwork`'s ``crs``.

    Returns:
        ``points`` unchanged, or a new :class:`PointsOfInterest` with
        reprojected geometry and ``crs=target_crs``.
    """
    if points.crs is None or points.crs == target_crs:
        return points

    import pyproj

    transformer = pyproj.Transformer.from_crs(points.crs, target_crs, always_xy=True)
    geoms = shapely.from_wkb(points.df["geometry"].to_numpy())
    geoms = shapely.transform(geoms, lambda c: np.column_stack(transformer.transform(c[:, 0], c[:, 1])))
    df = points.df.with_columns(pl.Series("geometry", shapely.to_wkb(geoms), dtype=pl.Binary))
    return PointsOfInterest(df, crs=target_crs)


class AccessibilityAnalyzer:
    """Orchestrates a :class:`StreetNetwork` + :class:`PointsOfInterest` into isochrones."""

    def __init__(self, street_network: StreetNetwork, points: PointsOfInterest):
        self.street_network = street_network
        self.points = points

    def default_distance_matrix(
        self,
        distance_steps: Sequence[float],
        poi_score_col: str = "poi_score",
        score_bins: Optional[int] = None,
    ):
        """See :func:`UrbanAccessAnalyzer.isochrones.default_distance_matrix`."""
        return isochrones.default_distance_matrix(self.points.df, distance_steps, poi_score_col, score_bins=score_bins)

    def run(
        self,
        distance_matrix,
        poi_score_col: Optional[str] = "poi_score",
        access_score_values: Optional[Sequence] = None,
        min_edge_length: float = 1.0,
        max_dist: Optional[float] = None,
        undirected: bool = True,
        verbose: bool = True,
        unreached_access_score: Optional[float] = 0.0,
        chunk_h3_resolution: Optional[int] = None,
        chunk_buffer_m: float = 1000.0,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Compute the multi-tier access-score isochrone; see :func:`UrbanAccessAnalyzer.isochrones.graph`.

        Args:
            distance_matrix: See :func:`UrbanAccessAnalyzer.isochrones.distance_matrix_to_processing_order`.
            poi_score_col: Score column on ``self.points.df``. Pass ``None``
                if every POI should be treated as a single undifferentiated
                tier (then ``distance_matrix`` is just a list of distances).
            access_score_values: Explicit access-score labels, positionally
                aligned with ``distance_matrix`` (``access_score_values[i]``
                is the score awarded within ``distance_matrix[i]``). Only
                meaningful when ``distance_matrix`` is a plain list of
                distances -- ignored when it's already a labeled matrix
                DataFrame.
            min_edge_length: Minimum snap/split segment length (meters).
            max_dist: Maximum POI-to-edge snap distance (meters).
            undirected: Ignore one-way restrictions during the search.
            verbose: Print per-tier progress.
            unreached_access_score: Score given to the parts of the network no
                POI reaches. Defaults to ``0.0`` so both returned tables cover
                the *whole* network -- every node and every edge, scored ``0``
                where there is no access instead of being omitted. Set to
                ``None`` to get only the reached subset (the pre-2026-08-13
                behaviour); automatically ignored for categorical access-score
                labels, which have no meaningful zero. See
                :func:`UrbanAccessAnalyzer.isochrones.exact_edge_access`.
            chunk_h3_resolution: If set, computes node access in
                memory-bounded H3-res-N chunks instead of one whole-network
                pass -- see :func:`UrbanAccessAnalyzer.isochrones.compute_node_access_chunked`.
                Opt-in; ``None`` (default) is the original, unaffected
                behavior.
            chunk_buffer_m: Buffer (meters) forwarded when
                ``chunk_h3_resolution`` is set. Must be >= the largest
                distance in ``distance_matrix``.

        Returns:
            Tuple ``(node_access, edge_access)`` Polars DataFrames -- see
            :func:`UrbanAccessAnalyzer.isochrones.graph`.
        """
        points = _reproject_points(self.points, self.street_network.crs)
        node_access, edge_access = isochrones.graph(
            self.street_network.nodes,
            self.street_network.edges,
            points.df,
            distance_matrix,
            poi_score_col=poi_score_col,
            access_score_values=access_score_values,
            min_edge_length=min_edge_length,
            max_dist=max_dist,
            undirected=undirected,
            verbose=verbose,
            unreached_access_score=unreached_access_score,
            chunk_h3_resolution=chunk_h3_resolution,
            chunk_buffer_m=chunk_buffer_m,
            crs=self.street_network.crs,
        )
        self._last_node_access = node_access
        self._last_edge_access = edge_access
        return node_access, edge_access

    def to_gdf(self, edge_access: Optional[pl.DataFrame] = None) -> gpd.GeoDataFrame:
        """Export the last (or given) edge-access result as a street-edges GeoDataFrame.

        Args:
            edge_access: Result of :meth:`run`; defaults to the most recent call.

        Returns:
            GeoDataFrame of scored edge segments, in the street network's CRS.
        """
        edge_access = edge_access if edge_access is not None else self._last_edge_access
        geom = shapely.from_wkb(edge_access["geometry_wkb"].to_numpy())
        return gpd.GeoDataFrame(edge_access.drop("geometry_wkb").to_pandas(), geometry=geom, crs=self.street_network.crs)

    def _edge_access_wgs84(self, edge_access: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """Reproject the last (or given) edge-access result's geometry to EPSG:4326.

        Args:
            edge_access: Result of :meth:`run`; defaults to the most recent call.

        Returns:
            ``edge_access`` with ``geometry_wkb`` replaced by a WKB
            ``geometry`` column in lon/lat coordinates.
        """
        edge_access = edge_access if edge_access is not None else self._last_edge_access
        geom = shapely.from_wkb(edge_access["geometry_wkb"].to_numpy())
        transformer_crs = self.street_network.crs
        import pyproj

        to_wgs84 = pyproj.Transformer.from_crs(transformer_crs, "EPSG:4326", always_xy=True)
        geom_wgs84 = shapely.transform(geom, lambda c: np.column_stack(to_wgs84.transform(c[:, 0], c[:, 1])))
        return edge_access.drop("geometry_wkb").with_columns(
            pl.Series("geometry", shapely.to_wkb(geom_wgs84), dtype=pl.Binary)
        )

    def to_level(
        self,
        level_gdf: gpd.GeoDataFrame,
        id_col: str,
        edge_access: Optional[pl.DataFrame] = None,
        columns: Optional[list] = None,
        agg=None,
    ) -> gpd.GeoDataFrame:
        """Aggregate the last (or given) edge-access result onto an arbitrary polygon layer.

        Generalizes :meth:`to_h3` to any polygon ``GeoDataFrame`` (an H3
        grid, administrative boundaries, or any other polygon layer), via
        :func:`UrbanAccessAnalyzer.geometry_levels.edges_to_level` (which in
        turn requires the optional ``geohierarchy`` dependency).

        Args:
            level_gdf: Target polygon GeoDataFrame to aggregate onto.
            id_col: Name of the unique id column in ``level_gdf``.
            edge_access: Result of :meth:`run`; defaults to the most recent call.
            columns: Attribute columns to aggregate; defaults to
                ``["access_score"]``.
            agg: Aggregation strategy (a
                :class:`~geohierarchy.aggregation.AggregationStrategy`
                instance, or a per-column mapping); defaults to
                :class:`~geohierarchy.aggregation.Mean` for every column.

        Returns:
            A copy of ``level_gdf`` with ``id_col``, geometry, and the
            aggregated ``columns`` -- one row per polygon.
        """
        columns = columns if columns is not None else ["access_score"]
        df = self._edge_access_wgs84(edge_access)
        return geometry_levels.edges_to_level(
            df, level_gdf, id_col=id_col, columns=columns, agg=agg, geometry_col="geometry"
        )

    def to_h3(self, resolution: int, edge_access: Optional[pl.DataFrame] = None, method: str = "max") -> pl.DataFrame:
        """Rasterize the last (or given) edge-access result to H3 cells.

        Thin convenience wrapper: builds an H3 grid over the edges' extent
        via the optional ``geohierarchy`` dependency's
        :func:`~geohierarchy.h3_cells`, then delegates to :meth:`to_level`.

        Args:
            resolution: H3 resolution.
            edge_access: Result of :meth:`run`; defaults to the most recent call.
            method: Aggregation method for ``access_score``: ``"max"``,
                ``"min"``, ``"mean"``, or ``"sum"``.

        Returns:
            Polars DataFrame with an ``"h3"`` cell-id column and an
            aggregated ``access_score`` column.
        """
        try:
            from geohierarchy import h3_cells
            from geohierarchy.aggregation import Max, Mean, Min, Sum
        except ImportError as e:
            raise ImportError(
                "Polygon/H3 aggregation requires the 'geohierarchy' extra: "
                "pip install urbanaccessanalyzer[geohierarchy]"
            ) from e

        method_to_agg = {"max": Max(), "min": Min(), "mean": Mean(), "sum": Sum()}
        if method not in method_to_agg:
            raise ValueError(f"Unknown method {method!r}; expected one of {sorted(method_to_agg)}.")

        edge_access = edge_access if edge_access is not None else self._last_edge_access
        geom = shapely.from_wkb(edge_access["geometry_wkb"].to_numpy())
        transformer_crs = self.street_network.crs
        import pyproj

        to_wgs84 = pyproj.Transformer.from_crs(transformer_crs, "EPSG:4326", always_xy=True)
        geom_wgs84 = shapely.transform(geom, lambda c: np.column_stack(to_wgs84.transform(c[:, 0], c[:, 1])))
        extent = gpd.GeoSeries(geom_wgs84, crs="EPSG:4326")
        grid = h3_cells(extent, resolution)

        result_gdf = self.to_level(
            grid, id_col="h3", edge_access=edge_access, columns=["access_score"], agg=method_to_agg[method]
        )
        return pl.from_pandas(result_gdf.drop(columns="geometry"))


def compute_accessibility(
    place: str,
    poi_kind: Union[str, Sequence[str]],
    pbf_path: str,
    buffer: float = 0.0,
    network_type: str = "walk",
    distance_steps: Sequence[float] = (400, 800, 1200),
    access_score_values: Optional[Sequence] = None,
    simplify_distance: Optional[float] = 30.0,
    max_dist: Optional[float] = None,
    verbose: bool = True,
) -> tuple[gpd.GeoDataFrame, AreaOfInterest, PointsOfInterest]:
    """Run the whole AOI -> network -> POIs -> accessibility pipeline in one call.

    A "do everything" convenience wrapper around
    :class:`AreaOfInterest`/:class:`StreetNetwork`/:class:`PointsOfInterest`/
    :class:`AccessibilityAnalyzer` for the common case: one place name, one
    (or a few) OSM POI kind(s), every POI counted as an equally-weighted
    single tier (``poi_score_col=None``, mirroring the pattern used in
    ``examples/rural_schools.ipynb``). Skip this and wire the classes up
    directly for anything more custom (per-POI scoring, cached/pre-built
    networks, multiple runs against the same network, etc.).

    Args:
        place: Place name/query geocoded via :meth:`AreaOfInterest.from_name`.
        poi_kind: One or more of :meth:`PointsOfInterest.from_overpass`'s
            ``kind`` values (e.g. ``"schools"``, or
            ``["schools", "groceries", "shops"]``). When several are given,
            their POIs are pooled into a single undifferentiated set (only
            the ``geometry`` column is kept, since tag columns differ across
            kinds).
        pbf_path: Path to a local ``.osm.pbf`` file; downloaded from
            Geofabrik if missing (see :meth:`StreetNetwork.from_pbf`).
        buffer: Meters to buffer the geocoded AOI by (also used as the
            street/POI download extent, to avoid isochrone edge effects).
        network_type: Forwarded to :meth:`StreetNetwork.from_pbf`.
        distance_steps: Isochrone distance thresholds in meters, closest
            first, e.g. ``(400, 800, 1200)`` for a walking analysis.
        access_score_values: Explicit access-score label per
            ``distance_steps`` tier; defaults to ``None`` (auto-ranked,
            highest score for the closest tier) -- see
            :meth:`AccessibilityAnalyzer.run`.
        simplify_distance: Cluster-simplify distance (meters) applied to the
            street network right after loading; ``None`` skips simplification.
        max_dist: Maximum POI-to-edge snap distance (meters); defaults to no
            limit.
        verbose: Print per-tier progress during the isochrone computation.

    Returns:
        Tuple ``(access_edges_gdf, aoi, points)``: the scored street-edges
        GeoDataFrame (see :meth:`AccessibilityAnalyzer.to_gdf`), the
        :class:`AreaOfInterest` used, and the :class:`PointsOfInterest` used.
    """
    aoi = AreaOfInterest.from_name(place, buffer=buffer)
    kinds = [poi_kind] if isinstance(poi_kind, str) else list(poi_kind)
    poi_dfs = [PointsOfInterest.from_overpass(kind, aoi.gdf).df.select("geometry") for kind in kinds]
    combined = pl.concat(poi_dfs) if len(poi_dfs) > 1 else poi_dfs[0]
    # from_overpass can return way/relation polygons alongside nodes (e.g.
    # a school building footprint); collapse everything to a representative
    # point since routing/snapping needs Point geometries.
    geoms = shapely.from_wkb(combined["geometry"].to_numpy())
    geoms = np.where(shapely.get_type_id(geoms) == shapely.GeometryType.POINT, geoms, shapely.centroid(geoms))
    points = PointsOfInterest(
        combined.with_columns(pl.Series("geometry", shapely.to_wkb(geoms), dtype=pl.Binary)), crs="EPSG:4326"
    )

    network = StreetNetwork.from_pbf(
        pbf_path, aoi=aoi, network_type=network_type, simplify_distance=simplify_distance
    )

    analyzer = AccessibilityAnalyzer(network, points)
    analyzer.run(
        distance_matrix=list(distance_steps),
        poi_score_col=None,
        access_score_values=access_score_values,
        max_dist=max_dist,
        verbose=verbose,
    )
    return analyzer.to_gdf(), aoi, points
