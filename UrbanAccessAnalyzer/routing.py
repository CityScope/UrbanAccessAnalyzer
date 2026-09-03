"""Shortest-path routing core: multi-source Dijkstra for isochrones.

Polars has no graph primitives, and implementing Dijkstra as iterative
join/group-by relaxation rounds in Polars would need one full-table pass per
network hop (a 15-minute, ~1200 m walk isochrone over ~100 m edges needs
20-40+ hop rounds through real, non-grid street topology), which is both slow
and only Bellman-Ford-equivalent, not true Dijkstra. Instead this module
converts the Polars edge table into a SciPy sparse CSR adjacency matrix once
and delegates the actual shortest-path search to
:func:`scipy.sparse.csgraph.dijkstra`, a compiled, well-tested Fibonacci-heap
Dijkstra implementation that natively supports multiple sources and a distance
cutoff in a single call. SciPy is a NumPy/Polars-adjacent numerical library,
not a graph-modeling framework like networkx, so using it here does not
reintroduce the dependency this refactor removes. Polars is used for all data
prep and result assembly around that one call.

Methodology:
    Multi-source Dijkstra computes, for a set of source nodes, the shortest
    (weighted) distance from *any* source to every reachable node, which is
    exactly the node set of a street-network isochrone: "all points reachable
    within N meters of travel from these origins." This is equivalent to
    adding a virtual super-source node connected to every real source with
    zero-weight edges and running single-source Dijkstra from it; SciPy's
    ``indices`` parameter does this internally without materializing the
    virtual node.

Source: SciPy docs,
https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.dijkstra.html
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def build_csr(edges: pl.DataFrame, nodes: pl.DataFrame) -> tuple[csr_matrix, dict, np.ndarray]:
    """Build a SciPy CSR adjacency matrix from Polars node/edge tables.

    Args:
        edges: DataFrame with directed edge columns ``u``, ``v``,
            ``length_m`` (or another weight column selected by the caller
            before calling this function; the weight column must be named
            ``length_m``).
        nodes: DataFrame with a ``node_id`` column listing every node that
            should exist in the matrix (including isolated ones), so index
            positions are stable across repeated calls with the same graph.

    Returns:
        A tuple ``(csgraph, node_id_to_index, index_to_node_id)``:

        - ``csgraph``: ``scipy.sparse.csr_matrix`` of shape
          ``(n_nodes, n_nodes)`` with edge weights.
        - ``node_id_to_index``: dict mapping OSM/graph ``node_id`` to matrix
          row/column index.
        - ``index_to_node_id``: numpy array such that
          ``index_to_node_id[i]`` is the ``node_id`` at matrix index ``i``.
    """
    node_ids = nodes["node_id"].to_numpy()
    index_to_node_id = node_ids
    node_id_to_index = {int(nid): i for i, nid in enumerate(node_ids)}

    n = len(node_ids)
    u_idx = edges["u"].replace_strict(node_id_to_index, default=None).to_numpy()
    v_idx = edges["v"].replace_strict(node_id_to_index, default=None).to_numpy()
    weights = edges["length_m"].to_numpy()

    valid = ~(np.isnan(u_idx.astype(float)) | np.isnan(v_idx.astype(float)))
    u_idx, v_idx, weights = u_idx[valid].astype(np.int64), v_idx[valid].astype(np.int64), weights[valid]

    csgraph = csr_matrix((weights, (u_idx, v_idx)), shape=(n, n))
    return csgraph, node_id_to_index, index_to_node_id


def multi_source_distances(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    source_node_ids: Sequence[int],
    radius: float,
    directed: bool = True,
    csr: Optional[tuple] = None,
) -> pl.DataFrame:
    """Compute shortest-path distance from the nearest source to every reachable node.

    Args:
        nodes: Node table with a ``node_id`` column.
        edges: Directed edge table with ``u``, ``v``, ``length_m`` columns
            (as produced by :func:`UrbanAccessAnalyzer.osm_io.load_pbf` or
            :mod:`UrbanAccessAnalyzer.graph_ops`).
        source_node_ids: Node ids to search from simultaneously (e.g. street
            nodes nearest to a set of POIs).
        radius: Maximum distance (in the same units as ``length_m``, normally
            meters) to search out to; nodes farther than this are dropped
            from the result, not returned with ``inf``.
        directed: If ``False``, edges are treated as traversable in both
            directions regardless of the ``oneway`` flag already encoded in
            ``edges`` (i.e. the CSR matrix is symmetrized). Street networks
            from :mod:`osm_io` already contain both directions for two-way
            ways, so this is normally left ``True``; set ``False`` to ignore
            one-way restrictions (e.g. for a pure walking network where
            one-way tags shouldn't apply).
        csr: Optional pre-built ``(csgraph, node_id_to_index,
            index_to_node_id)`` tuple, as returned by :func:`build_csr`, to
            reuse across repeated calls against the same ``nodes``/``edges``
            (the graph topology) with different ``source_node_ids``/
            ``radius`` (e.g. one call per access-score tier). When given,
            ``nodes`` and ``edges`` are not re-read to build the matrix --
            only used as documentation of which graph the caller intends;
            pass the same ``nodes``/``edges`` used to build ``csr``. When
            ``None`` (default), the CSR is built fresh from ``nodes``/
            ``edges`` on every call, exactly matching prior behavior.

    Returns:
        Polars DataFrame with columns ``node_id (i64)``, ``source_id (i64)``
        (the nearest source that reaches this node), ``dist (f64)``. Contains
        one row per node reachable within ``radius`` of at least one source,
        including the sources themselves at ``dist = 0``. ``dist`` is a
        "lower is better" quantity, so unreached nodes are simply absent
        rather than given a sentinel value; if a caller left-joins this
        against a full node table, unreached nodes correctly come out as
        ``null`` (representing an effectively infinite distance) -- never
        fill them with 0, which would mean "already there."
    """
    if csr is not None:
        csgraph, node_id_to_index, index_to_node_id = csr
    else:
        csgraph, node_id_to_index, index_to_node_id = build_csr(edges, nodes)

    source_indices = np.array(
        [node_id_to_index[int(sid)] for sid in source_node_ids if int(sid) in node_id_to_index],
        dtype=np.int64,
    )
    if len(source_indices) == 0:
        return pl.DataFrame(schema={"node_id": pl.Int64, "source_id": pl.Int64, "dist": pl.Float64})

    dist_matrix, predecessors, sources_arr = dijkstra(
        csgraph,
        directed=directed,
        indices=source_indices,
        limit=radius,
        return_predecessors=True,
        min_only=True,
    )
    # With min_only=True, scipy returns 1-D arrays: the best distance to each
    # node across all sources, and which source index achieved it.
    reachable = np.isfinite(dist_matrix)
    reached_idx = np.nonzero(reachable)[0]
    if len(reached_idx) == 0:
        return pl.DataFrame(schema={"node_id": pl.Int64, "source_id": pl.Int64, "dist": pl.Float64})

    nearest_source_index = sources_arr[reached_idx]
    return pl.DataFrame(
        {
            "node_id": index_to_node_id[reached_idx],
            "source_id": index_to_node_id[nearest_source_index],
            "dist": dist_matrix[reached_idx],
        }
    )


def multi_source_distances_per_source(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    source_node_ids: Sequence[int],
    radius: float,
    directed: bool = True,
) -> pl.DataFrame:
    """Like :func:`multi_source_distances`, but keeps every (source, node) pair.

    Useful when the caller needs to know all sources within range of a node,
    not just the nearest one (e.g. counting how many POIs are reachable).
    More expensive: runs one Dijkstra sweep per source instead of a single
    ``min_only`` sweep.

    Args:
        nodes: Node table with a ``node_id`` column.
        edges: Directed edge table with ``u``, ``v``, ``length_m`` columns.
        source_node_ids: Node ids to search from.
        radius: Maximum search distance.
        directed: Whether to respect edge direction.

    Returns:
        Polars DataFrame with columns ``node_id``, ``source_id``, ``dist``,
        one row per (source, reachable node) pair.
    """
    csgraph, node_id_to_index, index_to_node_id = build_csr(edges, nodes)
    source_indices = np.array(
        [node_id_to_index[int(sid)] for sid in source_node_ids if int(sid) in node_id_to_index],
        dtype=np.int64,
    )
    if len(source_indices) == 0:
        return pl.DataFrame(schema={"node_id": pl.Int64, "source_id": pl.Int64, "dist": pl.Float64})

    dist_matrix = dijkstra(csgraph, directed=directed, indices=source_indices, limit=radius)

    frames = []
    for row, src_idx in enumerate(source_indices):
        reached = np.nonzero(np.isfinite(dist_matrix[row]))[0]
        if len(reached) == 0:
            continue
        frames.append(
            pl.DataFrame(
                {
                    "node_id": index_to_node_id[reached],
                    "source_id": np.full(len(reached), index_to_node_id[src_idx]),
                    "dist": dist_matrix[row, reached],
                }
            )
        )
    if not frames:
        return pl.DataFrame(schema={"node_id": pl.Int64, "source_id": pl.Int64, "dist": pl.Float64})
    return pl.concat(frames, how="vertical")


def induced_edges(edges: pl.DataFrame, node_distances: pl.DataFrame) -> pl.DataFrame:
    """Filter an edge table down to edges whose endpoints are both reachable.

    Args:
        edges: Full directed edge table with ``u``, ``v`` columns.
        node_distances: Output of :func:`multi_source_distances` (or similar),
            with a ``node_id`` column of reachable nodes.

    Returns:
        Subset of ``edges`` where both ``u`` and ``v`` are present in
        ``node_distances``, with ``u_dist``/``v_dist`` columns joined in from
        ``node_distances.dist``.
    """
    dist_lookup = node_distances.select("node_id", pl.col("dist"))
    return (
        edges.join(dist_lookup.rename({"node_id": "u", "dist": "u_dist"}), on="u", how="inner")
        .join(dist_lookup.rename({"node_id": "v", "dist": "v_dist"}), on="v", how="inner")
    )
