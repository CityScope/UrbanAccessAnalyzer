"""Street-graph topology operations on Polars node/edge tables.

Replaces the previous networkx/osmnx-backed ``graph_processing.py``. The
graph is represented purely as two Polars DataFrames:

- ``nodes``: ``node_id (i64), x (f64), y (f64)`` (projected, meters).
- ``edges``: ``u (i64), v (i64), length_m (f64), geometry_wkb (binary)``,
  plus whatever extra attribute columns (``highway``, ...) the caller kept.

Geometry itself has no Polars-native representation, so it is stored as WKB
bytes and manipulated with vectorized `shapely
<https://shapely.readthedocs.io/en/stable/manual.html>`_ array functions
(``shapely.STRtree`` for nearest-neighbor/nearest-edge queries,
``shapely.line_locate_point``/``shapely.ops.substring`` for projecting points
onto edges and splitting them). This is real geometric computation, not a
graph library, so it stays even though networkx/geopandas do not.

Methodology:

- **Projection**: nodes/edges are projected to a local UTM CRS with
  ``pyproj.Transformer`` so that ``length_m`` and all distance thresholds are
  true meters, not degrees.
- **Chain contraction (simplification)**: an OSM way is frequently split into
  many short edges at every intersection-irrelevant vertex OSM happened to
  record. A node of undirected degree exactly 2 that is not "protected" (e.g.
  a snapped point of interest) carries no routing information -- it can be
  merged into a single edge spanning its two neighbors. This is done
  iteratively (each pass contracts one layer of degree-2 chains) with a
  Polars self-join, and geometries are merged with ``shapely.line_merge``.
- **Point snapping**: a query point is assigned to its nearest edge via an
  STRtree nearest-neighbor query, then projected onto that edge
  (``shapely.line_locate_point``) and the edge is split there
  (``shapely.ops.substring``) into two new edges plus a new node, exactly like
  osmnx's ``add_points_to_graph`` did previously but computed vectorized
  instead of via the WKT-string edge-splitting code in the old
  ``graph_processing.py``.

Source for the degree-2 contraction concept: standard graph-simplification
technique used by OSM routing engines (e.g. osmnx's ``simplify_graph``,
https://osmnx.readthedocs.io).
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import polars as pl
import pyproj
import shapely
from shapely import STRtree


def project(
    nodes: pl.DataFrame, edges: pl.DataFrame, crs: Optional[str] = None
) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    """Project lon/lat nodes and WKB edge geometry into a metric CRS.

    Args:
        nodes: Node table with ``node_id, lon, lat`` columns (EPSG:4326), as
            produced by :func:`UrbanAccessAnalyzer.osm_io.load_pbf`.
        edges: Edge table with a ``geometry_wkb`` column in EPSG:4326.
        crs: Target CRS (e.g. ``"EPSG:32633"``). If ``None``, the UTM zone
            covering the nodes' centroid is used.

    Returns:
        Tuple ``(nodes, edges, crs)`` where ``nodes`` gains ``x, y`` columns
        and ``edges``' ``geometry_wkb``/``length_m`` are recomputed in the
        target CRS.
    """
    if crs is None:
        lon_c, lat_c = float(nodes["lon"].mean()), float(nodes["lat"].mean())
        utm_zone = int((lon_c + 180) / 6) + 1
        hemisphere = 32600 if lat_c >= 0 else 32700
        crs = f"EPSG:{hemisphere + utm_zone}"

    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    nodes = nodes.with_columns(pl.Series("x", x), pl.Series("y", y))

    lines = shapely.from_wkb(edges["geometry_wkb"].to_numpy())
    lines_proj = shapely.transform(lines, lambda c: np.column_stack(transformer.transform(c[:, 0], c[:, 1])))
    edges = edges.with_columns(
        pl.Series("geometry_wkb", shapely.to_wkb(lines_proj), dtype=pl.Binary),
        pl.Series("length_m", shapely.length(lines_proj)),
    )
    return nodes, edges, crs


def crop_by_aoi(nodes: pl.DataFrame, edges: pl.DataFrame, aoi: gpd.GeoDataFrame | gpd.GeoSeries) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Crop a graph to edges intersecting an AOI.

    Args:
        nodes: Node table (``node_id`` + coordinate columns).
        edges: Edge table with ``u, v, geometry_wkb`` (CRS must match ``aoi``
            or be already-projected consistently with it).
        aoi: AOI geometry.

    Returns:
        Tuple ``(nodes, edges)`` filtered to the AOI: edges intersecting it,
        and nodes referenced by at least one kept edge.
    """
    aoi_geom = aoi.union_all()
    lines = shapely.from_wkb(edges["geometry_wkb"].to_numpy())
    shapely.prepare(aoi_geom)
    keep = shapely.intersects(lines, aoi_geom)
    edges = edges.filter(pl.Series(keep))
    used_ids = pl.concat([edges.select(pl.col("u").alias("node_id")), edges.select(pl.col("v").alias("node_id"))]).unique()
    nodes = nodes.join(used_ids, on="node_id", how="inner")
    return nodes, edges


def connected_component_labels(nodes: pl.DataFrame, edges: pl.DataFrame) -> pl.DataFrame:
    """Label every node with its (undirected) connected-component id.

    Uses `scipy.sparse.csgraph.connected_components` (the same
    sparse-matrix graph-algorithm family `UrbanAccessAnalyzer.routing`
    already relies on for Dijkstra, rather than adding a networkx
    dependency this package deliberately avoids -- see this module's own
    docstring). Cheap: builds one sparse adjacency matrix over `len(nodes)`
    nodes and `len(edges)` edges, no all-pairs work.

    Args:
        nodes: Node table with a `node_id` column.
        edges: Edge table with `u`, `v` columns (values must be `node_id`s
            present in `nodes`).

    Returns:
        `nodes` with an added `_component` column (`int`, arbitrary
        labeling -- only equality between two rows' labels is meaningful).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _sparse_connected_components

    node_ids = nodes["node_id"].to_numpy()
    n = len(node_ids)
    if n == 0:
        return nodes.with_columns(pl.Series("_component", [], dtype=pl.Int64))
    pos = pl.DataFrame({"node_id": node_ids, "_pos": np.arange(n)})
    # Real bug (caught by this function's own test suite): two SEPARATE
    # `.join()` calls -- one for `u`, one for `v` -- do NOT guarantee
    # matching row order with each other. A plain inner join is free to
    # reorder rows internally (hash-join bucket order), so `u_pos[i]` and
    # `v_pos[i]` could silently end up describing TWO DIFFERENT edges
    # instead of the two endpoints of edge `i` -- fabricating bogus
    # adjacency pairs and real ones going missing, both without error.
    # `maintain_order="left"` pins each join's output to `edges`' own
    # original row order, so both position arrays stay aligned edge-for-edge.
    edge_pos = (
        edges.select("u", "v")
        .join(pos.rename({"node_id": "u", "_pos": "u_pos"}), on="u", how="inner", maintain_order="left")
        .join(pos.rename({"node_id": "v", "_pos": "v_pos"}), on="v", how="inner", maintain_order="left")
    )
    u_pos = edge_pos["u_pos"].to_numpy()
    v_pos = edge_pos["v_pos"].to_numpy()
    m = min(len(u_pos), len(v_pos))
    adjacency = coo_matrix(
        (np.ones(m, dtype=np.int8), (u_pos[:m], v_pos[:m])), shape=(n, n)
    )
    _n_components, labels = _sparse_connected_components(csgraph=adjacency, directed=False)
    return nodes.with_columns(pl.Series("_component", labels))


def crop_by_aoi_connected(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    aoi: gpd.GeoDataFrame | gpd.GeoSeries,
    all_nodes: Optional[pl.DataFrame] = None,
    all_edges: Optional[pl.DataFrame] = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """`crop_by_aoi`, but guarantees a connected graph for every SIMPLE polygon of `aoi`.

    2026-09-02, explicit user request: "make sure we have a complete
    connected graph per unique polygon of the aoi... so no multipolygon,
    per simple polygon... include [a non-pedestrian road] if really needed
    for connectivity of the graph then... consider the road walkable."
    Plain `crop_by_aoi` only filters by geometric intersection -- a
    pedestrian-only (or otherwise road-type-restricted) network can come
    out of that crop genuinely fragmented into disconnected islands (e.g.
    a residential cluster whose only real link to the rest of the network
    is a road type the profile excluded), which silently makes some
    origin/destination pairs unroutable even though a real path exists on
    the ground.

    `aoi` is exploded into its individual (non-multi) polygons first --
    each is cropped and repaired independently, then the results are
    concatenated (deduplicated on `node_id`/`(u, v)` for nodes shared
    across adjacent polygons). Two real AOI polygons are not assumed to be
    connected to each other at all (they may be genuinely separate, e.g.
    two disjoint districts) -- only connectivity WITHIN each one is
    guaranteed.

    Repair strategy per polygon:
      1. Crop `nodes`/`edges` to the polygon (`crop_by_aoi`).
      2. Label connected components (`connected_component_labels`). If
         there's only one, this polygon's graph is already fine.
      3. Otherwise, if a wider-profile graph is available (`all_nodes`/
         `all_edges`, e.g. built with `network_type="all"`), crop THAT to
         the same polygon. If the wider graph is itself connected (the
         normal case -- OSM's full road network, unlike its pedestrian
         subset, is almost always contiguous within a real developed
         area), it fully replaces this polygon's edges/nodes: every road
         that was needed for connectivity is now included and is treated
         as walkable, exactly the "include it and consider it walkable"
         request, not a penalized/avoided edge.
      4. If still disconnected (no wider graph given, or the wider graph
         is ALSO fragmented -- e.g. a real island with no bridge), fall
         back to keeping only the largest connected component -- the
         standard routing-graph practice of dropping genuinely-isolated
         islands rather than leaving unroutable dead-end fragments in.

    Args:
        nodes: Node table (already the road-type-filtered, e.g. `"walk"`,
            profile), `node_id` + coordinate columns.
        edges: Edge table (same profile), `u`, `v`, `geometry_wkb` (CRS
            consistent with `aoi`).
        aoi: AOI geometry -- may be a single polygon, a MultiPolygon, or a
            GeoDataFrame/GeoSeries with several rows; every constituent
            simple polygon is handled independently.
        all_nodes: Optional wider-profile node table (e.g.
            `network_type="all"`) covering at least the same extent as
            `nodes`, used only as a source of bridging edges/nodes.
        all_edges: Optional wider-profile edge table, paired with
            `all_nodes`.

    Returns:
        Tuple `(nodes, edges)`, each polygon's contribution unioned
        together, deduplicated.
    """
    if isinstance(aoi, gpd.GeoSeries):
        aoi = gpd.GeoDataFrame(geometry=aoi)
    polygons = []
    for geom in aoi.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiPolygon":
            polygons.extend(list(geom.geoms))
        elif geom.geom_type == "Polygon":
            polygons.append(geom)
        # Any other geometry type (e.g. a GeometryCollection) is not a
        # meaningful "AOI polygon" for this purpose and is skipped rather
        # than guessed at.

    out_nodes: list[pl.DataFrame] = []
    out_edges: list[pl.DataFrame] = []
    for poly in polygons:
        poly_gdf = gpd.GeoDataFrame(geometry=[poly], crs=aoi.crs)
        poly_nodes, poly_edges = crop_by_aoi(nodes, edges, poly_gdf)
        if poly_edges.height == 0:
            continue
        labeled = connected_component_labels(poly_nodes, poly_edges)
        n_components = labeled["_component"].n_unique()

        if n_components > 1 and all_edges is not None and all_nodes is not None:
            wide_nodes, wide_edges = crop_by_aoi(all_nodes, all_edges, poly_gdf)
            if wide_edges.height > 0:
                wide_labeled = connected_component_labels(wide_nodes, wide_edges)
                # Bug fix (2026-09-04, live: Andorra's walk-only graph
                # collapsed from ~4,400 nodes to 252 once trunk/primary
                # roads were excluded by default). This used to require
                # the wider graph to be a PERFECT single component
                # (`n_unique() == 1`) before adopting it -- real OSM data
                # essentially never is (Andorra's real "all roads" crop:
                # 158 components, but the largest is 135,657 of 139,992
                # nodes -- 97%, the rest is ordinary noise like isolated
                # parking-lot loops and disconnected service tracks). That
                # strict check failed, so the wider graph was silently
                # never adopted at all, and the code fell through to the
                # largest-component fallback below applied to the
                # NARROW (walk-only) graph's own fragmented components --
                # exactly the small-village-sized island the wider
                # profile exists to bridge past. Fixed: always prefer the
                # wider graph's own largest connected component over the
                # narrow graph's, whenever a wider graph is available --
                # by construction it can only be a superset of the narrow
                # graph's connectivity (more road types = more edges),
                # never a worse choice, so there's no case where checking
                # for "perfectly single component" first was actually
                # buying extra safety.
                wide_sizes = wide_labeled.group_by("_component").len().sort("len", descending=True)
                wide_largest = wide_sizes["_component"][0]
                wide_keep_ids = set(
                    wide_labeled.filter(pl.col("_component") == wide_largest)["node_id"].to_list()
                )
                wide_nodes_kept = wide_nodes.filter(pl.col("node_id").is_in(wide_keep_ids))
                wide_edges_kept = wide_edges.filter(
                    pl.col("u").is_in(wide_keep_ids) & pl.col("v").is_in(wide_keep_ids)
                )
                if wide_nodes_kept.height >= poly_nodes.height:
                    # The wider graph's best connected piece covers at
                    # least as much as the narrow graph did -- every edge
                    # in it is now a real, used routing edge, i.e.
                    # genuinely "considered walkable", not a special-cased
                    # bridge.
                    poly_nodes, poly_edges = wide_nodes_kept, wide_edges_kept
                    n_components = 1

        if n_components > 1:
            # Fall back to the largest connected component -- standard
            # routing-graph practice; a smaller island stays out rather
            # than becoming an unroutable dead end. `mode(keep=True)`-style
            # tie-break (first-seen largest) is fine here: which of two
            # exactly-equal-size components survives is not meaningful.
            sizes = labeled.group_by("_component").len().sort("len", descending=True)
            largest = sizes["_component"][0]
            keep_ids = labeled.filter(pl.col("_component") == largest)["node_id"]
            keep_ids_set = set(keep_ids.to_list())
            poly_nodes = poly_nodes.filter(pl.col("node_id").is_in(keep_ids_set))
            poly_edges = poly_edges.filter(pl.col("u").is_in(keep_ids_set) & pl.col("v").is_in(keep_ids_set))

        out_nodes.append(poly_nodes)
        out_edges.append(poly_edges)

    if not out_nodes:
        return nodes.clear(), edges.clear()

    result_nodes = pl.concat(out_nodes, how="diagonal_relaxed").unique(subset=["node_id"])
    result_edges = pl.concat(out_edges, how="diagonal_relaxed").unique(subset=["u", "v"])
    return result_nodes, result_edges


def grid_cluster_labels(
    points: pl.DataFrame,
    cell_size: float,
    id_col: str = "node_id",
    x_col: str = "x",
    y_col: str = "y",
    protected_ids: Optional[set] = None,
) -> pl.DataFrame:
    """Two-offset-grid clustering: bounds every cluster to ~`cell_size`, no chaining.

    Overlays two square grids of `cell_size` over the point set, the second
    offset by half a cell in both axes, so every point falls into exactly
    one cell of *each* grid -- two candidate clusters. Each point joins
    whichever of its two candidate cells has the nearer mean position, then
    (since membership just changed) that mean is implicitly recomputed the
    next time a caller aggregates by the returned `_cluster` label (e.g.
    `points.join(labels).group_by("_cluster").agg(pl.col(x_col).mean(),
    ...)`) -- a single further group-by, not an iterative/converging
    k-means. The two-grid overlap (vs. a single hard grid) avoids the
    artifact where two points a meter apart, straddling one grid line,
    would otherwise never cluster together.

    Unlike chaining nearby points transitively (e.g. connected components
    over a graph of short edges), a cluster here can never span more than
    roughly `cell_size` regardless of how many points or how long a chain
    of short links connects them -- which is what "chain" clustering can
    do on a curvy street built from many short segments, collapsing a much
    longer real-world span into one point.

    Args:
        points: Table with `id_col`, `x_col`, `y_col` (projected, meters).
        cell_size: Grid cell side length (meters). Every cluster's extent
            is bounded to roughly this scale.
        id_col: Unique id column name.
        x_col: X-coordinate column name.
        y_col: Y-coordinate column name.
        protected_ids: Ids that must stay their own singleton cluster
            (never merged with another point).

    Returns:
        Polars DataFrame with `id_col` and `_cluster` (a label, not
        necessarily numeric) -- one row per input point.
    """
    protected_ids = protected_ids or set()
    protected_mask = pl.col(id_col).is_in(protected_ids)
    protected_rows = points.filter(protected_mask).select(
        id_col, (pl.lit("p") + pl.col(id_col).cast(pl.Utf8)).alias("_cluster")
    )
    free = points.filter(~protected_mask)
    if free.is_empty():
        return protected_rows

    half = cell_size / 2.0
    free = free.with_columns(
        (pl.col(x_col) / cell_size).floor().cast(pl.Int64).alias("_gxa"),
        (pl.col(y_col) / cell_size).floor().cast(pl.Int64).alias("_gya"),
        ((pl.col(x_col) - half) / cell_size).floor().cast(pl.Int64).alias("_gxb"),
        ((pl.col(y_col) - half) / cell_size).floor().cast(pl.Int64).alias("_gyb"),
    ).with_columns(
        (pl.lit("a") + pl.col("_gxa").cast(pl.Utf8) + "_" + pl.col("_gya").cast(pl.Utf8)).alias("_cell_a"),
        (pl.lit("b") + pl.col("_gxb").cast(pl.Utf8) + "_" + pl.col("_gyb").cast(pl.Utf8)).alias("_cell_b"),
    )
    mean_a = free.group_by("_cell_a").agg(pl.col(x_col).mean().alias("_mxa"), pl.col(y_col).mean().alias("_mya"))
    mean_b = free.group_by("_cell_b").agg(pl.col(x_col).mean().alias("_mxb"), pl.col(y_col).mean().alias("_myb"))
    free = free.join(mean_a, on="_cell_a").join(mean_b, on="_cell_b")
    free = free.with_columns(
        ((pl.col(x_col) - pl.col("_mxa")) ** 2 + (pl.col(y_col) - pl.col("_mya")) ** 2).alias("_da"),
        ((pl.col(x_col) - pl.col("_mxb")) ** 2 + (pl.col(y_col) - pl.col("_myb")) ** 2).alias("_db"),
    ).with_columns(
        pl.when(pl.col("_da") <= pl.col("_db")).then(pl.col("_cell_a")).otherwise(pl.col("_cell_b")).alias("_cluster")
    )
    return pl.concat([free.select(id_col, "_cluster"), protected_rows])


def simplify(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    cluster_distance: float,
    protected_node_ids: Optional[list[int]] = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cluster nearby nodes and collapse the graph onto the cluster centroids.

    Two-step, fully vectorized simplification designed for very large graphs
    (no per-edge/per-node Python loop):

    1. **Clustering**: nodes are grid-clustered (:func:`grid_cluster_labels`,
       `cell_size=cluster_distance`) -- every cluster's extent is bounded to
       roughly `cluster_distance`, unlike chaining nodes transitively through
       a sequence of short edges (which has no such bound: a long curvy
       street built from many short segments could otherwise collapse
       entirely into one point). Each cluster collapses to a single node at
       the mean coordinate of its members. ``protected_node_ids`` (e.g.
       nodes a POI was snapped to) never merge into a cluster with another
       node -- they stay singleton clusters.
    2. **Edge reduction**: every edge's endpoints are remapped to their
       cluster's representative node. Edges whose two endpoints land in the
       same cluster are dropped (internal edges). Among edges left connecting
       the same two clusters, only the straightest one is kept: the one
       minimizing ``|real_edge_length - straight_line_distance_between_cluster_centroids|``.
       The kept edge's geometry has its first/last coordinate snapped to the
       new cluster centroid (interior vertices untouched) so the linestring
       stays continuous and valid, and its length is recomputed from that
       adjusted geometry.

    Args:
        nodes: Node table (``node_id, x, y``, projected CRS).
        edges: Directed edge table (``u, v, length_m, geometry_wkb,
            edge_uid``, plus any extra attribute columns), as produced by
            :func:`UrbanAccessAnalyzer.osm_io.load_pbf` + :func:`project`.
            ``edge_uid`` must be shared between the forward/backward rows of
            the same original two-way segment so they're kept or dropped
            together.
        cluster_distance: Grid cell size used to cluster nodes (meters) --
            see :func:`grid_cluster_labels`.
        protected_node_ids: Node ids to keep as their own singleton cluster.

    Returns:
        Tuple ``(nodes, edges)``: one row per cluster in ``nodes``, and the
        reduced, geometry-corrected edge table.
    """
    protected = set(int(p) for p in (protected_node_ids or []))
    clusters = grid_cluster_labels(nodes, cluster_distance, protected_ids=protected)
    cluster_reps = (
        nodes.join(clusters, on="node_id")
        .group_by("_cluster")
        .agg(pl.col("x").mean(), pl.col("y").mean(), pl.col("node_id").min().alias("cluster_id"))
    )
    node_to_cluster = (
        clusters.join(cluster_reps.select("_cluster", "cluster_id"), on="_cluster")
        .select("node_id", "cluster_id")
    )
    new_nodes = cluster_reps.select(pl.col("cluster_id").alias("node_id"), "x", "y")

    edges = (
        edges.join(node_to_cluster.rename({"node_id": "u", "cluster_id": "new_u"}), on="u")
        .join(node_to_cluster.rename({"node_id": "v", "cluster_id": "new_v"}), on="v")
        .filter(pl.col("new_u") != pl.col("new_v"))
    )
    if edges.is_empty():
        return new_nodes, edges.drop("new_u", "new_v")

    edges = edges.join(new_nodes.rename({"node_id": "new_u", "x": "new_u_x", "y": "new_u_y"}), on="new_u")
    edges = edges.join(new_nodes.rename({"node_id": "new_v", "x": "new_v_x", "y": "new_v_y"}), on="new_v")

    straight_dist = (
        (pl.col("new_u_x") - pl.col("new_v_x")) ** 2 + (pl.col("new_u_y") - pl.col("new_v_y")) ** 2
    ).sqrt()
    edges = edges.with_columns((pl.col("length_m") - straight_dist).abs().alias("_straightness"))

    pair_key = pl.when(pl.col("new_u") < pl.col("new_v")).then(pl.col("new_u")).otherwise(pl.col("new_v")).alias("_pair_a")
    pair_key2 = pl.when(pl.col("new_u") < pl.col("new_v")).then(pl.col("new_v")).otherwise(pl.col("new_u")).alias("_pair_b")
    edges = edges.with_columns(pair_key, pair_key2)

    best_uid = (
        edges.group_by("_pair_a", "_pair_b")
        .agg(pl.col("edge_uid").sort_by("_straightness").first().alias("_best_uid"))
    )
    edges = edges.join(best_uid, on=["_pair_a", "_pair_b"]).filter(pl.col("edge_uid") == pl.col("_best_uid"))

    # Snap first/last coordinate of each kept edge's geometry to its new
    # cluster centroids; interior vertices are left untouched.
    lines = shapely.from_wkb(edges["geometry_wkb"].to_numpy())
    coords, index = shapely.get_coordinates(lines, return_index=True)
    first_mask = np.concatenate(([True], index[1:] != index[:-1]))
    last_mask = np.concatenate((index[:-1] != index[1:], [True]))
    coords[first_mask, 0] = edges["new_u_x"].to_numpy()
    coords[first_mask, 1] = edges["new_u_y"].to_numpy()
    coords[last_mask, 0] = edges["new_v_x"].to_numpy()
    coords[last_mask, 1] = edges["new_v_y"].to_numpy()
    lines = shapely.transform(lines, lambda c: c)  # copy before in-place set_coordinates
    shapely.set_coordinates(lines, coords)

    edges = edges.with_columns(
        pl.Series("geometry_wkb", shapely.to_wkb(lines), dtype=pl.Binary),
        pl.Series("length_m", shapely.length(lines)),
        pl.col("new_u").alias("u"),
        pl.col("new_v").alias("v"),
    ).drop("new_u", "new_v", "new_u_x", "new_u_y", "new_v_x", "new_v_y", "_straightness", "_pair_a", "_pair_b", "_best_uid")

    used_ids = pl.concat([edges.select(pl.col("u").alias("node_id")), edges.select(pl.col("v").alias("node_id"))]).unique()
    new_nodes = new_nodes.join(used_ids, on="node_id", how="inner")

    return new_nodes, edges


def snap_points(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    point_geoms: np.ndarray,
    max_dist: Optional[float] = None,
    min_edge_length: float = 1.0,
) -> tuple[pl.DataFrame, pl.DataFrame, list[Optional[int]]]:
    """Snap external points onto the nearest graph edge, splitting it.

    Args:
        nodes: Node table (``node_id, x, y`` at minimum, projected CRS
            matching ``edges``).
        edges: Directed edge table (``u, v, length_m, geometry_wkb``, plus
            extra attribute columns).
        point_geoms: Array of shapely ``Point`` geometries, in the same CRS
            as ``nodes``/``edges``.
        max_dist: Maximum snap distance; points farther than this from every
            edge are not snapped (returned as ``None``).
        min_edge_length: If the projected point falls within this distance of
            an existing edge endpoint, it is snapped to that endpoint instead
            of creating a near-duplicate node.

    Returns:
        Tuple ``(nodes, edges, point_node_ids)``: updated node/edge tables
        with new nodes/edges inserted at each split, and a list the same
        length as ``point_geoms`` giving the node id each point was snapped
        to (``None`` where no edge was within ``max_dist``).
    """
    if len(point_geoms) == 0:
        return nodes, edges, []

    lines = shapely.from_wkb(edges["geometry_wkb"].to_numpy())
    tree = STRtree(lines)
    nearest_idx = tree.nearest(point_geoms)
    nearest_lines = lines[nearest_idx]

    if max_dist is not None:
        dists = shapely.distance(point_geoms, nearest_lines)
        within = dists <= max_dist
    else:
        within = np.ones(len(point_geoms), dtype=bool)

    proj_dist = shapely.line_locate_point(nearest_lines, point_geoms)
    edge_len = shapely.length(nearest_lines)

    next_node_id = int(nodes["node_id"].max()) + 1
    edges = edges.with_row_index("_row")
    new_edge_rows: list[dict] = []
    drop_edge_rows: set[int] = set()
    point_node_ids: list[Optional[int]] = [None] * len(point_geoms)

    extra_cols = [c for c in edges.columns if c not in ("u", "v", "length_m", "geometry_wkb", "_row")]
    row_lookup = {r["_row"]: r for r in edges.iter_rows(named=True)}

    for i, (row_idx, d, elen, is_within) in enumerate(zip(nearest_idx, proj_dist, edge_len, within)):
        if not is_within:
            continue
        edge_row = row_lookup[int(row_idx)]

        if d <= min_edge_length:
            point_node_ids[i] = edge_row["u"]
            continue
        if (elen - d) <= min_edge_length:
            point_node_ids[i] = edge_row["v"]
            continue

        line = shapely.from_wkb(edge_row["geometry_wkb"])
        part1 = shapely.ops.substring(line, 0, d)
        part2 = shapely.ops.substring(line, d, elen)
        new_id = next_node_id
        next_node_id += 1

        new_edge_rows.append(
            {"u": edge_row["u"], "v": new_id, "length_m": float(d), "geometry_wkb": shapely.to_wkb(part1),
             **{c: edge_row[c] for c in extra_cols}}
        )
        new_edge_rows.append(
            {"u": new_id, "v": edge_row["v"], "length_m": float(elen - d), "geometry_wkb": shapely.to_wkb(part2),
             **{c: edge_row[c] for c in extra_cols}}
        )
        drop_edge_rows.add(int(row_idx))
        point_node_ids[i] = new_id

        pt = shapely.line_interpolate_point(line, d)
        nodes = pl.concat(
            [nodes, pl.DataFrame({"node_id": [new_id], "x": [pt.x], "y": [pt.y]}).select(
                [c for c in nodes.columns if c in ("node_id", "x", "y")]
            )],
            how="diagonal_relaxed",
        )

    if new_edge_rows:
        edges = edges.filter(~pl.col("_row").is_in(list(drop_edge_rows))).drop("_row")
        edges = pl.concat([edges, pl.DataFrame(new_edge_rows, schema=edges.schema)], how="vertical")
    else:
        edges = edges.drop("_row")

    return nodes, edges, point_node_ids


def nearest_nodes(nodes: pl.DataFrame, point_geoms: np.ndarray, max_dist: Optional[float] = None) -> list[Optional[int]]:
    """Find the nearest graph node to each query point.

    Args:
        nodes: Node table (``node_id, x, y``), projected CRS.
        point_geoms: Array of shapely ``Point`` geometries in the same CRS.
        max_dist: Maximum search distance; points with no node within this
            distance return ``None``.

    Returns:
        List (same length as ``point_geoms``) of nearest ``node_id`` or
        ``None``.
    """
    node_points = shapely.points(nodes["x"].to_numpy(), nodes["y"].to_numpy())
    tree = STRtree(node_points)
    idx = tree.nearest(point_geoms)
    node_ids = nodes["node_id"].to_numpy()
    result: list[Optional[int]] = []
    for i, pt in zip(idx, point_geoms):
        if max_dist is not None and shapely.distance(node_points[i], pt) > max_dist:
            result.append(None)
        else:
            result.append(int(node_ids[i]))
    return result
