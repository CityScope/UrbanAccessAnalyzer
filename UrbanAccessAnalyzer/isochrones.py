"""Multi-tier accessibility isochrones on top of the Polars routing core.

An isochrone here is not just "reachable within N meters" — points of
interest (POIs) can have different *access scores* (e.g. a large well-staffed
school scores higher than a one-room schoolhouse), and each score tier
reaches a different distance before its contribution to accessibility drops
off. This module computes, for every street node and every point along every
edge, the best access score reachable within that tier's distance budget from
any POI, using :mod:`UrbanAccessAnalyzer.routing` (scipy CSR Dijkstra) as the
distance engine and Polars for all tabular orchestration.

Methodology:

1. A *distance matrix* maps ``(poi_score tier, distance threshold) ->
   access_score``. :func:`default_distance_matrix` builds a simple diagonal
   matrix; callers may also supply their own (e.g. calibrated with
   :mod:`UrbanAccessAnalyzer.scoring`).
2. :func:`distance_matrix_to_processing_order` collapses that matrix to one
   row per distinct ``access_score``, carrying the *maximum* distance at
   which each POI-score tier still earns that access_score, and ranks
   access_score values from best (rank 0) to worst.
3. For each row, :func:`compute_node_access` runs
   ``routing.multi_source_distances`` from the POIs in that tier, at that
   tier's radius, and keeps, per node, the largest "remaining distance"
   (``radius - shortest_dist``) seen for that access_score across all rows
   that share it. A node's final access score is the best (lowest rank) tier
   for which it has positive remaining distance — ties broken by whichever
   tier has more remaining distance, matching the original "keep the larger
   remaining distance" reconciliation rule.
4. Edges get split at the exact point where the best-reaching tier gives out
   and a worse tier picks up: for each tier from best to worst, the portion
   of an edge within ``remaining_dist`` of either endpoint that is not
   already claimed by a better tier is assigned to that tier and its
   sub-segment geometry cut with ``shapely.ops.substring`` (this replaces
   the old WKT-string / pandas ``__exact_isochrones`` pass in
   ``graph_processing.py`` with a vectorized NumPy pass over the small
   ``n_edges x n_tiers`` matrix).
5. :func:`buffers` computes the pure Euclidean (no street network) equivalent
   for cases where a network isn't available or wanted: nested tier buffers
   around POIs with `shapely.difference` used to keep only the outermost
   ring not already covered by a better tier.

This package is scoped to general OSM-derived accessibility, so there is no
transit/GTFS-specific code here — a GTFS feed's stops are just another POI
input with precomputed access scores.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Sequence, Union

import geopandas as gpd
import h3
import numpy as np
import polars as pl
import pyproj
import shapely
from tqdm import tqdm

from . import graph_ops, routing

logger = logging.getLogger(__name__)


def default_distance_matrix(
    poi: pl.DataFrame,
    distance_steps: Sequence[float],
    poi_score_col: str = "poi_score",
    score_bins: Optional[int] = None,
) -> tuple[pl.DataFrame, list[float]]:
    """Build a diagonal poi-score x distance access-score matrix.

    Higher poi_score and shorter distance both push the resulting
    access_score towards 1.0; poi_score == 0 always yields access_score 0.

    Args:
        poi: DataFrame with a ``poi_score_col`` column of POI quality/score
            values in some numeric range.
        distance_steps: Distance thresholds (e.g. ``[200, 400, 800, 1200]``
            meters), best (shortest) first or any order — sorted internally.
        poi_score_col: Column in ``poi`` holding the per-POI score.
        score_bins: Optional number of quantile buckets used to collapse
            *how many distinct rank positions* continuous ``poi_score``
            values are spread across, before computing each row's access
            score. Defaults to ``None``, which preserves the exact current
            behavior: each distinct ``poi_score`` gets its own rank and thus
            (generally) its own access-score tier. When POI scores are
            continuous floats (e.g. a computed score in ``[0, 1]``), this can
            produce dozens of near-duplicate tiers -- each requiring a full
            network search in :func:`compute_node_access` /
            :func:`exact_edge_access` -- for little accuracy benefit, since
            nearly-identical scores yield nearly-identical access outcomes
            anyway. Setting ``score_bins`` (``20``-``50`` is a reasonable
            range) quantile-buckets the distinct nonzero scores into at most
            that many rank groups (scores in the same bucket share one rank
            and therefore one access-score value per distance step, so they
            collapse into the same downstream tier), which can turn a step
            that used to build ~90 tiers into ~10-20. The ``poi_score``
            column of the returned matrix still contains the *original,
            unbucketed* score values -- so it still joins correctly against
            a ``points``/``poi`` table with continuous scores -- only the
            *value assigned* to nearby scores is quantized, at the cost of a
            small approximation error in the resulting access scores for
            POIs whose scores fall in the same bucket. This parameter is
            opt-in and ``score_bins=None`` leaves today's behavior bit-for-bit
            unchanged.

            Measured tradeoff (real Cambridge, MA street network + GTFS
            stops, 866 distinct continuous ``stop_score`` values, distance
            steps ``(400, 800, 1200)``, error measured on the ``[0, 1]``
            per-node ``access_score`` scale vs. the ``score_bins=None``
            baseline): ``score_bins=50`` -> mean abs error 0.0045, max error
            0.010, ~2.9x speedup; ``score_bins=100`` -> mean abs error
            0.0028, max error 0.005, ~2.6x speedup. See
            ``transitlos.level_of_service.compute_level_of_service``'s
            docstring for the full table across more bin counts -- that
            function defaults to ``score_bins=50`` for transitLOS's public
            pipeline (this library-level default stays ``None``).

            Note: prior to a fix, the bucket-rank normalization here
            re-ranked binned scores down to small contiguous integers while
            the access-score formula's denominator (``max_idx``) stayed at
            the *original* (unbucketed) scale -- this made every binned
            access_score collapse toward the same value regardless of bin
            count, with errors of 0.3-0.9 on realistic data instead of the
            small quantization error described above. Bucket ranks now keep
            their original-scale mean rank position instead of being
            re-ranked to contiguous small integers.

    Returns:
        Tuple ``(distance_matrix, access_score_values)``:

        - ``distance_matrix``: one row per distinct poi_score (excluding 0),
          one column per distance step plus a ``poi_score`` column, values
          are access scores in ``[0, 1]``.
        - ``access_score_values``: sorted distinct access score values found
          in the matrix, best first.
    """
    poi_scores = poi[poi_score_col].drop_nulls().unique().to_numpy()
    poi_scores = np.unique(np.append(poi_scores, 0.0))
    poi_scores = np.sort(poi_scores)[::-1]
    distance_steps = sorted(distance_steps)

    n_sq, n_dist = len(poi_scores), len(distance_steps)

    # `rank[i]` is normally just `i` (each distinct poi_score gets its own
    # rank, 0 = best). With `score_bins` set, nonzero scores are grouped by
    # quantile into at most `score_bins` buckets and every score in a bucket
    # shares the same rank -- collapsing many near-identical continuous
    # scores onto a handful of ranks/tiers -- while `poi_scores` itself keeps
    # every original distinct value so the matrix still joins exactly
    # against the caller's raw per-POI scores.
    nonzero_mask = poi_scores != 0
    n_nonzero = int(nonzero_mask.sum())
    rank = np.arange(n_sq, dtype=float)
    if score_bins is not None and n_nonzero > score_bins:
        nonzero_scores = poi_scores[nonzero_mask]  # descending, all distinct
        ascending = nonzero_scores[::-1]
        quantile_edges = np.unique(np.quantile(ascending, np.linspace(0, 1, score_bins + 1)))
        n_buckets = len(quantile_edges) - 1
        bucket_idx_ascending = np.clip(np.searchsorted(quantile_edges, ascending, side="right") - 1, 0, n_buckets - 1)
        # Higher score -> lower (better) bucket rank; use each bucket's mean
        # rank position among the original per-score ranks so spacing in the
        # access-score formula below stays proportional to bucket population.
        orig_rank_ascending = np.arange(n_nonzero - 1, -1, -1, dtype=float)  # descending scores get rank n-1..0 reversed to align w/ ascending order
        bucket_mean_rank = np.array([orig_rank_ascending[bucket_idx_ascending == b].mean() for b in range(n_buckets)])
        rank_ascending = bucket_mean_rank[bucket_idx_ascending]
        rank_nonzero_descending = rank_ascending[::-1]
        # IMPORTANT: keep `rank_nonzero_descending` at its original
        # full-resolution scale (0..n_nonzero-1, just quantized to each
        # bucket's mean position) rather than re-ranking it down to small
        # contiguous bucket indices (0..n_buckets-1). `max_idx` below is
        # computed from the *unbinned* rank scale (dominated by the zero-score
        # entry, which always keeps its full-resolution rank), so plugging in
        # small contiguous bucket ranks against that large-scale max_idx used
        # to make every binned access_score collapse toward 1.0 (values were
        # divided by a denominator ~n_sq while the numerator only ranged over
        # ~n_buckets), producing errors of 0.3-0.9 on real/synthetic score
        # distributions -- not the intended small quantization error. Keeping
        # the mean-rank's original scale preserves proportional spacing.
        rank[nonzero_mask] = rank_nonzero_descending.astype(float)

    max_idx = rank.max() + n_dist - 1 if n_sq else 1

    rows = []
    for i, sq in enumerate(poi_scores):
        row = {"poi_score": float(sq)}
        for j, dist in enumerate(distance_steps):
            value = 0.0 if sq == 0 else max(1.0 - ((rank[i] + j) / max_idx), 0.0)
            row[str(dist)] = round(value, 3)
        rows.append(row)

    matrix = pl.DataFrame(rows).filter(pl.col("poi_score").abs() > 1e-9)
    access_score_values = sorted(
        {v for row in rows for k, v in row.items() if k != "poi_score"}, reverse=True
    )
    return matrix, access_score_values


def distance_matrix_to_processing_order(
    distance_matrix: Union[pl.DataFrame, Sequence[float]],
    access_score_values: Optional[Sequence] = None,
) -> pl.DataFrame:
    """Collapse a distance matrix into one row per distinct access_score.

    Args:
        distance_matrix: Either a Polars DataFrame (as returned by
            :func:`default_distance_matrix`, with a ``poi_score`` column and
            one column per distance) or a flat sequence of distances (in
            which case every POI is treated as a single undifferentiated
            tier, ``poi_score = 1``, one access_score per distance).
        access_score_values: Explicit access-score labels. When
            ``distance_matrix`` is a plain list of distances, must be the
            same length and positionally aligned (``access_score_values[i]``
            is awarded within ``distance_matrix[i]``); rank is then assigned
            by first-appearance order in ``access_score_values`` (index 0 =
            best/rank 0). When ``distance_matrix`` is a DataFrame, used only
            to control rank order for otherwise-inferred access_score
            values (must align with the matrix's own distinct values).

    Returns:
        Polars DataFrame with columns ``poi_score (list[f64])``,
        ``distance (f64)``, ``access_score``, ``rank (i64, 0=best)``,
        sorted best-rank-first.
    """
    if not isinstance(distance_matrix, pl.DataFrame):
        distances = [float(d) for d in distance_matrix]
        labels = list(access_score_values) if access_score_values is not None else list(range(len(distances), 0, -1))
        if len(labels) != len(distances):
            raise ValueError("access_score_values must be the same length as distance_matrix")
        rank_by_label = {v: i for i, v in enumerate(dict.fromkeys(labels))}
        order = pl.DataFrame({"poi_score": [[1.0]] * len(distances), "distance": distances, "access_score": labels})
        order = order.with_columns(pl.col("access_score").replace_strict(rank_by_label, default=len(rank_by_label)).cast(pl.Int64).alias("rank"))
        return order.sort("rank")

    dist_cols = [c for c in distance_matrix.columns if c != "poi_score"]
    melted = distance_matrix.unpivot(index="poi_score", on=dist_cols, variable_name="distance", value_name="access_score")
    melted = melted.with_columns(pl.col("distance").cast(pl.Float64))

    if access_score_values is None:
        access_score_values = sorted(melted["access_score"].unique().to_list(), reverse=True)
    rank_map = {v: i for i, v in enumerate(access_score_values)}

    order = (
        melted.group_by("poi_score", "access_score")
        .agg(pl.col("distance").max())
        .group_by("access_score", "distance")
        .agg(pl.col("poi_score").alias("poi_score"))
        .with_columns(pl.col("access_score").replace_strict(rank_map, default=len(access_score_values)).cast(pl.Int64).alias("rank"))
        .sort(["rank", "distance"], descending=[True, True])
    )
    return order.select("poi_score", "distance", "access_score", "rank").sort("rank")


def compute_node_access(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    points: pl.DataFrame,
    process_order: pl.DataFrame,
    poi_score_col: Optional[str] = None,
    undirected: bool = True,
    verbose: bool = True,
) -> tuple[pl.DataFrame, dict[float, pl.DataFrame]]:
    """Compute the best reachable access score for every graph node.

    Args:
        nodes: Node table (``node_id`` at minimum).
        edges: Directed edge table (``u, v, length_m``).
        points: POI table with a ``node_id`` column (already snapped onto
            the graph, e.g. via :func:`UrbanAccessAnalyzer.graph_ops.snap_points`)
            and, if ``poi_score_col`` is given, a matching score column.
        process_order: Output of :func:`distance_matrix_to_processing_order`.
        poi_score_col: Column in ``points`` with per-POI score values
            matching the ``poi_score`` groups in ``process_order``. If
            ``None``, every point is treated as tier ``1.0``.
        undirected: Whether to ignore edge direction during the search.
        verbose: Print progress per tier.

    Note:
        The sparse CSR adjacency matrix (:func:`UrbanAccessAnalyzer.routing.build_csr`)
        is built once, before the per-tier loop, and reused for every tier's
        Dijkstra call -- the graph topology never changes between tiers, only
        the source-node set and radius do. This is a pure performance
        optimization with no effect on results.

    Returns:
        Tuple ``(node_access, tier_tables)``:

        - ``node_access``: columns ``node_id, access_score, rank,
          remaining_dist`` — one row per node with the best (lowest rank)
          access score it can reach and how much distance budget is left at
          that tier's radius. ``access_score`` is a "higher is better" score,
          so a node with no row here has no computed access at all; if you
          left-join this against a full node table for a dense result, fill
          those nulls with ``0`` (the worst score), not left as null --
          mirroring :func:`UrbanAccessAnalyzer.scoring.score_by_values`'s
          convention. ``remaining_dist``, by contrast, is a "lower is
          better"-style distance quantity and should stay ``null`` for
          unreached nodes rather than being filled with anything.
        - ``tier_tables``: dict mapping each distinct ``access_score`` value
          to a ``node_id, remaining_dist`` DataFrame, needed by
          :func:`exact_edge_access` to reproduce the same per-tier reach when
          splitting edges.
    """
    if poi_score_col is None:
        points = points.with_columns(pl.lit(1.0).alias("__poi_score"))
        poi_score_col = "__poi_score"

    # The graph topology (nodes/edges) is identical across every tier -- only
    # the source-node set and radius change -- so the CSR adjacency matrix is
    # built exactly once here and reused for every tier's Dijkstra call,
    # instead of rebuilding it from scratch per tier (which used to dominate
    # runtime when there were many distinct tiers, e.g. from continuous POI
    # scores producing dozens of near-duplicate tiers).
    csr = routing.build_csr(edges, nodes)

    tier_tables: dict[float, pl.DataFrame] = {}
    rows = list(process_order.iter_rows(named=True))
    pbar = tqdm(
        rows,
        desc="[isochrones] node access",
        unit="tier",
        disable=not verbose,
        mininterval=1.0,
    )
    for row in pbar:
        score_group = row["poi_score"] if isinstance(row["poi_score"], list) else [row["poi_score"]]
        source_ids = (
            points.filter(pl.col(poi_score_col).is_in(score_group))["node_id"].drop_nulls().unique().to_list()
        )
        if not source_ids:
            continue
        pbar.set_description(
            f"[isochrones] tier access_score={row['access_score']} distance={row['distance']}"
        )
        logger.debug(
            "[isochrones] tier access_score=%s distance=%s n_sources=%s",
            row["access_score"], row["distance"], len(source_ids),
        )

        reach = routing.multi_source_distances(
            nodes, edges, source_ids, radius=row["distance"], directed=not undirected, csr=csr
        )
        if reach.is_empty():
            continue
        reach = reach.select("node_id", (row["distance"] - pl.col("dist")).alias("remaining_dist"))

        existing = tier_tables.get(row["access_score"])
        if existing is None:
            tier_tables[row["access_score"]] = reach
        else:
            tier_tables[row["access_score"]] = (
                pl.concat([existing, reach]).group_by("node_id").agg(pl.col("remaining_dist").max())
            )

    if not tier_tables:
        return pl.DataFrame(schema={"node_id": pl.Int64, "access_score": pl.Float64, "rank": pl.Int64, "remaining_dist": pl.Float64}), {}

    rank_by_score = process_order.select("access_score", "rank").unique().to_dict(as_series=False)
    rank_map = dict(zip(rank_by_score["access_score"], rank_by_score["rank"]))

    frames = [
        table.with_columns(pl.lit(score).alias("access_score"), pl.lit(rank_map[score]).alias("rank"))
        for score, table in tier_tables.items()
    ]
    stacked = pl.concat(frames, how="vertical").filter(pl.col("remaining_dist") > 0)
    node_access = (
        stacked.sort(["node_id", "rank", "remaining_dist"], descending=[False, False, True])
        .group_by("node_id", maintain_order=True)
        .first()
    )
    return node_access, tier_tables


def compute_node_access_chunked(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    points: pl.DataFrame,
    process_order: pl.DataFrame,
    crs: str,
    poi_score_col: Optional[str] = None,
    undirected: bool = True,
    verbose: bool = True,
    chunk_h3_resolution: int = 4,
    buffer_m: float = 1000.0,
) -> tuple[pl.DataFrame, dict[float, pl.DataFrame]]:
    """Memory-bounded :func:`compute_node_access`, chunked by H3 res-N cell.

    ``compute_node_access`` builds one sparse CSR adjacency matrix for the
    *entire* input graph and keeps it (plus every tier's reach table) in
    memory for the whole run -- fine for an ordinary metro, but this scales
    with total graph size regardless of how "local" any given isochrone
    search actually is, and has caused real OOM crashes on oversized AOIs
    (e.g. Boston's accidentally state-scale AOI, Shanghai's population
    scale). This function instead partitions the graph into H3 res-N cells
    (``chunk_h3_resolution``, coarse by default -- res 4 is ~1,770 km^2 per
    cell), clips ``nodes``/``edges`` to each cell **expanded by
    ``buffer_m``**, and calls the existing, *unmodified*
    :func:`compute_node_access` on just that clipped subgraph -- so the only
    new logic here is graph partitioning and result merging; the actual
    Dijkstra/tiering logic is exactly the unchunked code path, just run on
    smaller inputs, one cell at a time.

    Buffer correctness (read before changing the default): the buffer must
    be at least as large as the longest real search radius any tier in
    ``process_order`` can reach (``process_order["distance"].max()``), or a
    stop/POI near a cell's edge will have its isochrone truncated at the
    chunk boundary instead of reaching its true extent -- a silent
    correctness bug, not a crash, so this is checked and warned on (not
    silently allowed) rather than asserted, since a caller intentionally
    accepting a smaller buffer for a coarser approximation is plausible but
    should never happen by accident. The default (1000m) matches the
    project's own initial proposal but is NOT automatically correct for
    every study -- e.g. `CS_transitLOS`'s `StudyParams.walk_distance_steps`
    reaches 2000m, so studies using that config must pass
    ``buffer_m >= 2000`` explicitly.

    A node that falls within one cell's buffer-only zone (not its own core
    res-N cell) may get computed redundantly by that neighboring cell too;
    results are merged by keeping, per ``node_id``, the best (lowest rank,
    then largest remaining_dist) row across every chunk that computed it --
    the same "overlap and take the best" reconciliation
    :func:`compute_node_access` itself already uses across tiers within one
    chunk. This makes chunking a pure performance/memory strategy with no
    effect on results: a node inside a cell's buffer zone will, by
    construction, also be computed by its own home cell (whose buffer covers
    the reverse direction), and merging always picks that better-or-equal
    result -- see ``tests/test_isochrones_chunked.py`` for a chunked vs.
    unchunked equivalence test.

    Args:
        nodes: Full node table, projected (``node_id, x, y``). Each node's
            lon/lat (needed to assign it to an H3 cell, since H3 always
            operates in lon/lat) is derived by inverse-projecting ``x, y``
            back through ``crs``, so no separate lon/lat columns are
            required even after :func:`UrbanAccessAnalyzer.graph_ops.simplify`
            (which does not retain them).
        edges: Full directed edge table, projected, matching ``nodes``.
        points: POI table, see :func:`compute_node_access`.
        process_order: Output of :func:`distance_matrix_to_processing_order`.
        crs: The projected CRS ``nodes``/``edges`` are in (e.g. as returned
            by :func:`UrbanAccessAnalyzer.graph_ops.project`) -- used to
            reproject each H3 cell's lon/lat boundary into the graph's own
            CRS for clipping.
        poi_score_col: See :func:`compute_node_access`.
        undirected: See :func:`compute_node_access`.
        verbose: Print a progress bar over chunks (and disables the
            per-chunk, per-tier progress bar to avoid nested bar spam).
        chunk_h3_resolution: H3 resolution defining the chunk grid. Default
            4 (~1,770 km^2/cell) -- comfortably larger than any single
            city's dense core. Must be explicitly chosen per study/city, not
            hardcoded by a caller.
        buffer_m: Buffer distance (meters) added around each cell before
            clipping. Default 1000m. Must be >= the longest distance in
            ``process_order`` (checked, see above) or results near chunk
            edges will be truncated/wrong.

    Returns:
        Same shape as :func:`compute_node_access`: ``(node_access,
        tier_tables)``, plus one extra column on ``node_access``,
        ``h3_chunk_cell`` -- the res-``chunk_h3_resolution`` H3 cell the
        node itself (not a buffer neighbor's cell) falls in, for downstream
        chunk-labeled consumers (e.g. chunked tile generation). Unlike
        :func:`compute_node_access`, the returned ``tier_tables`` are
        per-chunk-merged-by-best only for nodes; per-tier remaining_dist
        tables are unioned across chunks with a max-reduction (same rule
        :func:`compute_node_access` itself uses to merge duplicate node_ids
        within one tier), so :func:`exact_edge_access` can still be called
        against the *original, unchunked* ``edges`` table with this
        function's ``tier_tables`` output.
    """
    if process_order.height and buffer_m < float(process_order["distance"].max()):
        warnings.warn(
            f"compute_node_access_chunked: buffer_m={buffer_m} is smaller than "
            f"the largest isochrone tier distance in process_order "
            f"({float(process_order['distance'].max())}m). Nodes near a chunk "
            "boundary will have their isochrone search truncated at the "
            "buffer edge instead of reaching this tier's true radius -- this "
            "silently produces WRONG (under-reaching) access scores near "
            "chunk edges. Pass buffer_m >= process_order['distance'].max() "
            "unless this approximation is intentional.",
            stacklevel=2,
        )
    to_lonlat = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lons, lats = to_lonlat.transform(nodes["x"].to_numpy(), nodes["y"].to_numpy())
    own_cells = np.array([h3.latlng_to_cell(lat, lon, chunk_h3_resolution) for lat, lon in zip(lats, lons)])
    nodes_tagged = nodes.with_columns(pl.Series("h3_chunk_cell", own_cells))

    cells = sorted(set(own_cells.tolist()))
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    node_frames: list[pl.DataFrame] = []
    tier_frames: dict[float, list[pl.DataFrame]] = {}

    for cell in tqdm(cells, desc="[isochrones] chunk", unit="cell", disable=not verbose, mininterval=1.0):
        boundary = h3.cell_to_boundary(cell)  # list[(lat, lon)]
        xs, ys = transformer.transform([lon for _, lon in boundary], [lat for lat, _ in boundary])
        cell_poly = shapely.Polygon(zip(xs, ys))
        buffered = shapely.buffer(cell_poly, buffer_m)
        aoi = gpd.GeoSeries([buffered], crs=crs)

        sub_nodes, sub_edges = graph_ops.crop_by_aoi(nodes_tagged, edges, aoi)
        if sub_nodes.is_empty() or sub_edges.is_empty():
            continue
        sub_points = points.filter(pl.col("node_id").is_in(sub_nodes["node_id"].implode()))
        if sub_points.is_empty():
            continue

        chunk_access, chunk_tier_tables = compute_node_access(
            sub_nodes.drop("h3_chunk_cell"), sub_edges, sub_points, process_order,
            poi_score_col=poi_score_col, undirected=undirected, verbose=False,
        )
        if chunk_access.height:
            chunk_access = chunk_access.join(
                sub_nodes.select("node_id", "h3_chunk_cell").filter(pl.col("h3_chunk_cell") == cell),
                on="node_id", how="inner",
            )
            node_frames.append(chunk_access)
        for score, table in chunk_tier_tables.items():
            tier_frames.setdefault(score, []).append(table)

    tier_tables = {
        score: pl.concat(frames, how="vertical").group_by("node_id").agg(pl.col("remaining_dist").max())
        for score, frames in tier_frames.items()
    }

    if not node_frames:
        empty_schema = {"node_id": pl.Int64, "access_score": pl.Float64, "rank": pl.Int64, "remaining_dist": pl.Float64, "h3_chunk_cell": pl.Utf8}
        return pl.DataFrame(schema=empty_schema), tier_tables

    combined = pl.concat(node_frames, how="vertical")
    node_access = (
        combined.sort(["node_id", "rank", "remaining_dist"], descending=[False, False, True])
        .group_by("node_id", maintain_order=True)
        .first()
    )
    return node_access, tier_tables


def _numeric_access_scores(process_order: pl.DataFrame) -> bool:
    """Whether ``access_score`` is a numeric ("0 means nothing") quantity.

    The "an unreached place scores 0, not null" rule only makes sense for a
    numeric access score, where ``0`` is a real, meaningful value ("no useful
    access here"). Callers may also label tiers with arbitrary categorical
    values (``"walk"``/``"bike"``/``"bus/car"`` -- see
    ``AccessibilityAnalyzer.run``'s ``access_score_values``), and there is no
    such thing as a "zero" category, so unreached rows are left out /
    ``null`` exactly as before for those.
    """
    return process_order.schema["access_score"].is_numeric()


def exact_edge_access(
    edges: pl.DataFrame,
    process_order: pl.DataFrame,
    tier_tables: dict[float, pl.DataFrame],
    min_edge_length: float = 0.0,
    unreached_access_score: Optional[float] = 0.0,
) -> pl.DataFrame:
    """Split edges at exact tier boundaries and assign an access_score to each piece.

    For each tier from best to worst, the portion of every edge within that
    tier's remaining distance of either endpoint (and not already claimed by
    a better tier) is cut out with ``shapely.ops.substring`` and labeled with
    that tier's ``access_score``. This reproduces sub-edge isochrone boundary
    precision without node-level snapping artifacts.

    Args:
        edges: Directed edge table (``u, v, length_m, geometry_wkb``, plus
            any extra columns, which are copied onto every output piece).
        process_order: Output of :func:`distance_matrix_to_processing_order`
            (used for tier ranking, best first).
        tier_tables: Per-tier ``node_id, remaining_dist`` tables from
            :func:`compute_node_access`.
        min_edge_length: Minimum sub-segment length to keep; shorter
            fragments are dropped (their access_score effectively inherited
            by the adjacent kept piece).
        unreached_access_score: Access score given to the parts of the
            network no tier reaches -- both whole edges outside every
            isochrone and the un-covered middle of a partially-covered edge.
            Defaults to ``0.0``: on a "higher is better" access score, ``0``
            is a real, meaningful value ("this street exists, and has no
            useful access"), whereas *omitting* the edge entirely (the
            pre-2026-08-13 behaviour) silently shrank the analysed network
            down to a halo around the POIs and made every downstream
            consumer see a network cropped to POI range rather than to the
            AOI. Pass ``None`` to restore that old drop-the-edge behaviour.
            Ignored (treated as ``None``) when ``access_score`` is
            categorical rather than numeric -- see
            :func:`_numeric_access_scores`.

    Returns:
        Polars DataFrame with the same columns as ``edges`` plus
        ``access_score (f64)``, one row per kept sub-segment. With
        ``unreached_access_score`` set (the default) every input edge is
        represented, in full, by one or more output pieces; with it set to
        ``None``, edges no tier reaches are dropped.

    Performance note:
        Per-tier, per-edge remaining-distance lookups (``rem_u``/``rem_v``)
        are vectorized with NumPy (``np.searchsorted`` against a sorted
        array of node ids, instead of a Python-level ``dict.get`` loop over
        every edge for every tier). The final per-edge segment cut still
        loops in Python because ``shapely.ops.substring`` has no vectorized
        (array-in, array-out) form, but that loop is restricted to only the
        edges with nonzero available length in the current tier (via
        ``np.nonzero``) instead of iterating over every edge on every tier
        regardless of whether it has anything left to cut.
    """
    tiers = (
        process_order.select("access_score", "rank").unique().sort("rank")["access_score"].to_list()
    )
    extra_cols = [c for c in edges.columns if c not in ("u", "v", "length_m", "geometry_wkb")]

    n = edges.height
    length = edges["length_m"].to_numpy().astype(float)
    dist_u = np.zeros(n)
    dist_v = np.zeros(n)
    lines = shapely.from_wkb(edges["geometry_wkb"].to_numpy())
    u_ids = edges["u"].to_numpy().astype(np.int64)
    v_ids = edges["v"].to_numpy().astype(np.int64)

    # Map node ids -> dense positions once, via sorted-array binary search,
    # so per-tier remaining-distance lookups are pure NumPy fancy indexing
    # instead of per-edge dict.get() calls.
    all_node_ids = np.unique(np.concatenate([u_ids, v_ids])) if n else np.array([], dtype=np.int64)
    u_pos = np.searchsorted(all_node_ids, u_ids)
    v_pos = np.searchsorted(all_node_ids, v_ids)

    out_rows: list[dict] = []
    edges_named = list(edges.select(extra_cols).iter_rows(named=True)) if extra_cols else [{}] * n

    for tier in tiers:
        table = tier_tables.get(tier)
        if table is None:
            continue
        table_node_ids = table["node_id"].to_numpy().astype(np.int64)
        table_rem = table["remaining_dist"].to_numpy().astype(float)
        pos = np.searchsorted(all_node_ids, table_node_ids)
        valid = (pos < len(all_node_ids)) & (all_node_ids[pos] == table_node_ids)
        rem_dense = np.zeros(len(all_node_ids))
        rem_dense[pos[valid]] = table_rem[valid]

        rem_u = rem_dense[u_pos]
        rem_v = rem_dense[v_pos]

        avail_u = np.clip(rem_u - dist_u, 0, None)
        avail_v = np.clip(rem_v - dist_v, 0, None)
        remaining_len = np.clip(length - dist_u - dist_v, 0, None)
        avail_u = np.minimum(avail_u, remaining_len)
        avail_v = np.minimum(avail_v, np.clip(remaining_len - avail_u, 0, None))

        # Only visit edges that actually have something to cut this tier
        # (instead of every edge on every tier), but keep the interleaved
        # u-then-v-per-edge row order of the original implementation --
        # Polars' row-oriented DataFrame-from-dicts schema inference can
        # otherwise misinfer a column's type (e.g. treat `v` as all-null)
        # when many same-shaped rows are grouped together instead of mixed.
        touched = np.nonzero((avail_u > min_edge_length) | (avail_v > min_edge_length))[0]
        for i in touched:
            if avail_u[i] > min_edge_length:
                start, end = dist_u[i], dist_u[i] + avail_u[i]
                seg = shapely.ops.substring(lines[i], start, end)
                out_rows.append({"u": int(u_ids[i]), "v": None, "length_m": float(end - start),
                                  "geometry_wkb": shapely.to_wkb(seg), "access_score": tier, **edges_named[i]})
            if avail_v[i] > min_edge_length:
                start, end = length[i] - dist_v[i] - avail_v[i], length[i] - dist_v[i]
                seg = shapely.ops.substring(lines[i], start, end)
                out_rows.append({"u": None, "v": int(v_ids[i]), "length_m": float(end - start),
                                  "geometry_wkb": shapely.to_wkb(seg), "access_score": tier, **edges_named[i]})

        dist_u = np.clip(dist_u + avail_u, 0, length)
        dist_v = np.clip(dist_v + avail_v, 0, length)

    # Whatever no tier reached is still part of the street network, and on a
    # "higher is better" score its honest value is 0, not "missing". Emit the
    # leftover middle of every partially-covered edge and the whole of every
    # untouched edge as one more piece scored `unreached_access_score`, so the
    # result covers the *entire* input network (see the arg's docstring).
    fill_unreached = unreached_access_score is not None and _numeric_access_scores(process_order) and bool(n)
    untouched_frame = None
    if fill_unreached:
        leftover = np.clip(length - dist_u - dist_v, 0, None)
        # On a real city the overwhelming majority of unreached edges were not
        # touched by *any* tier, so their leftover piece is the whole original
        # edge: no `substring` call, no per-row Python dict (which at metro
        # scale means millions of dicts and gigabytes of peak RSS). Those are
        # taken straight off the input frame as one vectorized slice, and only
        # the genuinely partially-covered edges go through the Python loop.
        untouched = (dist_u <= 0) & (dist_v <= 0) & (leftover > min_edge_length)
        partial = (~untouched) & (leftover > min_edge_length)
        if untouched.any():
            untouched_frame = edges.filter(pl.Series(untouched)).with_columns(
                pl.lit(unreached_access_score).alias("access_score")
            )
        for i in np.nonzero(partial)[0]:
            start, end = dist_u[i], length[i] - dist_v[i]
            seg = shapely.ops.substring(lines[i], start, end)
            out_rows.append(
                {
                    # Keep the real endpoint id on whichever side of the piece
                    # is still the edge's own endpoint, `None` on a cut side.
                    "u": int(u_ids[i]) if dist_u[i] <= 0 else None,
                    "v": int(v_ids[i]) if dist_v[i] <= 0 else None,
                    "length_m": float(end - start),
                    "geometry_wkb": shapely.to_wkb(seg),
                    "access_score": unreached_access_score,
                    **edges_named[i],
                }
            )

    out_schema = {
        **{c: edges.schema[c] for c in ["u", "v", "length_m", "geometry_wkb"]},
        **{c: edges.schema[c] for c in extra_cols},
        "access_score": process_order.schema["access_score"],
    }
    if untouched_frame is not None:
        untouched_frame = untouched_frame.select(list(out_schema)).cast(out_schema)  # type: ignore[arg-type]
    if not out_rows:
        return untouched_frame if untouched_frame is not None else pl.DataFrame(schema=out_schema)
    # Explicit schema instead of relying on pl.DataFrame(list[dict]) row-oriented
    # type inference: `u`/`v` alternate between real ints and None across rows
    # (a u-side piece has v=None and vice versa), and with a long enough run of
    # one-sided rows (beyond polars' default `infer_schema_length`), the
    # inferred dtype for the all-None column becomes Null, which then raises
    # `polars.ComputeError: could not append value ... make sure all rows have
    # the same schema` the moment a real int shows up in a later row. Passing
    # `schema=` makes construction independent of row order/interleaving.
    result = pl.DataFrame(out_rows, schema=out_schema)
    if untouched_frame is not None:
        result = pl.concat([result, untouched_frame], how="vertical")
    return result


def _densify_node_access(
    nodes: pl.DataFrame,
    node_access: pl.DataFrame,
    process_order: pl.DataFrame,
    unreached_access_score: float,
) -> pl.DataFrame:
    """Give every graph node a row, scoring unreached ones ``unreached_access_score``.

    :func:`compute_node_access` only emits the nodes some tier actually
    reached. A node that exists in the network but no POI can reach has a
    perfectly well-defined access score -- ``0``, "no useful access" -- and
    leaving it out (or ``null``) makes every downstream consumer either drop
    a real piece of the city or propagate a null through arithmetic that
    should have seen a zero. ``rank`` gets one worse than the worst real tier
    and ``remaining_dist`` stays ``null``, since a *distance* has no
    meaningful zero for an unreached node (see :func:`compute_node_access`'s
    own note).
    """
    if "node_id" not in nodes.columns:
        return node_access
    worst_rank = int(process_order["rank"].max() or 0) + 1
    dense = nodes.select("node_id").unique().join(node_access, on="node_id", how="left")
    return dense.with_columns(
        pl.col("access_score").fill_null(unreached_access_score),
        pl.col("rank").fill_null(worst_rank),
    ).select(node_access.columns)


def graph(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    points: pl.DataFrame,
    distance_matrix: Union[pl.DataFrame, Sequence[float]],
    poi_score_col: Optional[str] = None,
    access_score_values: Optional[Sequence] = None,
    min_edge_length: float = 0.0,
    max_dist: Optional[float] = None,
    undirected: bool = True,
    verbose: bool = True,
    unreached_access_score: Optional[float] = 0.0,
    chunk_h3_resolution: Optional[int] = None,
    chunk_buffer_m: float = 1000.0,
    crs: Optional[str] = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute a full multi-tier access-score isochrone over a street graph.

    Args:
        nodes: Projected node table (``node_id, x, y``).
        edges: Projected directed edge table (``u, v, length_m,
            geometry_wkb``).
        points: POI table with a ``geometry`` (shapely Point array) column
            (or an already-present ``node_id`` column if points are
            pre-snapped) and, optionally, a score column.
        distance_matrix: See :func:`distance_matrix_to_processing_order`.
        poi_score_col: Column in ``points`` with per-POI scores.
        access_score_values: Explicit access-score labels (see
            :func:`distance_matrix_to_processing_order`).
        min_edge_length: Minimum edge/sub-segment length used both when
            snapping points (:func:`UrbanAccessAnalyzer.graph_ops.snap_points`)
            and when splitting edges at tier boundaries.
        max_dist: Maximum snap distance for points onto the graph.
        undirected: Whether to ignore one-way restrictions during the search.
        verbose: Print per-tier progress.
        unreached_access_score: Score for network the isochrones never reach.
            Defaults to ``0.0``, which makes *both* returned tables dense over
            the whole input network: every node gets a row and every edge is
            represented in full, with ``access_score = 0`` where there is no
            access rather than the row being absent/``null``. ``remaining_dist``
            is a distance, not a "0 means nothing" score, so it deliberately
            stays ``null`` for unreached nodes. Pass ``None`` for the old
            sparse behaviour; ignored for categorical access scores.
        chunk_h3_resolution: If set, runs the memory-bounded
            :func:`compute_node_access_chunked` instead of
            :func:`compute_node_access` -- opt-in, additive path; ``None``
            (default) is the original, unaffected behavior. See that
            function's docstring for the H3-chunked isochrone strategy.
            Requires ``crs`` to also be given.
        chunk_buffer_m: Forwarded to :func:`compute_node_access_chunked` as
            ``buffer_m`` when ``chunk_h3_resolution`` is set. Default 1000m
            -- must be >= the largest distance in ``distance_matrix`` or
            results near chunk edges will be truncated (see that function's
            docstring).
        crs: Projected CRS of ``nodes``/``edges``, required (and only used)
            when ``chunk_h3_resolution`` is set.

    Returns:
        Tuple ``(node_access, edge_access)`` — see
        :func:`compute_node_access` and :func:`exact_edge_access`.
    """
    if "node_id" not in points.columns:
        point_geoms = shapely.from_wkb(points["geometry"].to_numpy()) if points["geometry"].dtype == pl.Binary else points["geometry"].to_numpy()
        nodes, edges, snapped_ids = graph_ops.snap_points(nodes, edges, point_geoms, max_dist=max_dist, min_edge_length=min_edge_length or 1.0)
        points = points.with_columns(pl.Series("node_id", snapped_ids))

    process_order = distance_matrix_to_processing_order(distance_matrix, access_score_values)
    if chunk_h3_resolution is not None:
        if crs is None:
            raise ValueError("isochrones.graph: crs is required when chunk_h3_resolution is set")
        node_access, tier_tables = compute_node_access_chunked(
            nodes, edges, points, process_order, crs,
            poi_score_col=poi_score_col, undirected=undirected, verbose=verbose,
            chunk_h3_resolution=chunk_h3_resolution, buffer_m=chunk_buffer_m,
        )
    else:
        node_access, tier_tables = compute_node_access(nodes, edges, points, process_order, poi_score_col, undirected, verbose)
    edge_access = exact_edge_access(
        edges, process_order, tier_tables, min_edge_length, unreached_access_score=unreached_access_score
    )
    if unreached_access_score is not None and _numeric_access_scores(process_order):
        node_access = _densify_node_access(nodes, node_access, process_order, unreached_access_score)
    return node_access, edge_access


def buffers(
    poi: pl.DataFrame,
    distance_matrix: Union[pl.DataFrame, Sequence[float]],
    poi_score_col: Optional[str] = None,
    access_score_values: Optional[Sequence] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """Compute Euclidean (no street network) buffer-ring access-score zones.

    Args:
        poi: DataFrame with a WKB ``geometry`` column (projected CRS, meters)
            and, optionally, a score column.
        distance_matrix: See :func:`distance_matrix_to_processing_order`.
        poi_score_col: Column in ``poi`` with per-POI scores.
        access_score_values: Explicit access-score labels.

    Returns:
        Polars DataFrame with columns ``access_score, rank, geometry_wkb`` —
        one non-overlapping ring per tier, best tier's ring drawn first and
        subtracted from worse tiers so no area is double-counted.
    """
    if poi_score_col is None:
        poi = poi.with_columns(pl.lit(1.0).alias("__poi_score"))
        poi_score_col = "__poi_score"

    process_order = distance_matrix_to_processing_order(distance_matrix, access_score_values)
    geoms = shapely.from_wkb(poi["geometry"].to_numpy())
    scores = poi[poi_score_col].to_numpy()

    tier_geoms: dict[float, list] = {}
    order_rows = list(process_order.iter_rows(named=True))
    for row in tqdm(order_rows, desc="[isochrones] buffers", unit="tier", disable=not verbose, mininterval=1.0):
        score_group = row["poi_score"] if isinstance(row["poi_score"], list) else [row["poi_score"]]
        mask = np.isin(scores, score_group)
        if not mask.any():
            continue
        merged = shapely.unary_union(geoms[mask])
        tier_geoms.setdefault(row["access_score"], []).append(shapely.buffer(merged, row["distance"], quad_segs=4))

    tiers_ranked = process_order.select("access_score", "rank").unique().sort("rank")
    tier_values = tiers_ranked["access_score"].to_list()
    rows, covered = [], None
    for access_score in tqdm(tier_values, desc="[isochrones] buffer rings", unit="tier", disable=not verbose, mininterval=1.0):
        parts = tier_geoms.get(access_score)
        if not parts:
            continue
        geom = shapely.unary_union(parts)
        ring = geom if covered is None else shapely.difference(geom, covered)
        covered = geom if covered is None else shapely.unary_union([covered, geom])
        rows.append({"access_score": access_score, "geometry_wkb": shapely.to_wkb(ring)})

    return pl.DataFrame(rows)
