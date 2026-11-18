"""Street network -> CSR graph, point snapping, and the edge <-> H3 cell records used by fast multi-source searches.

`CsrNetwork` takes the node/edge tables of a `StreetNetwork` (projected metres; `StreetNetwork.save` folders load with `CsrNetwork.load`, read-only) and gives
  * `indptr / adj / w`        undirected CSR graph (float32 metres), ready for compiled searches (transitlos.flex.core.multilabel_search),
  * `snap(lat, lon)`          each point becomes two sources: the two ends of the nearest edge, offset by the snap distance plus the position along the edge
                              (what splitting the edge at the point would give),
  * `edge_cell_records(res)`  for every H3 cell the streets touch, the edges that touch it with the stretch [a, b] of each edge inside the cell. The
                              distance until an edge touches a cell is then min(d_u + a, d_v + L - b) over those edges (+ the cell radius, added by the caller).
A station is snapped to the nearest point of the nearest edge and becomes two sources (the two edge ends, offset by the snap distance plus the
position along the edge), which is what splitting the edge at the stop would give. A cell's distance is the distance until an edge touches it
(entry/exit positions of the edge inside the cell) + the cell radius (added in `core.process_cells`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl
import shapely
from pyproj import Transformer

SAMPLE_M = 10.0   # spacing of the points used to find which cells an edge crosses (cells are ~130 m across)


class CsrNetwork:
    """A street network in compressed sparse row form, for fast shortest-path searches from many stations at once.

    Built from node and edge tables (``node_id``; ``u``, ``v``, ``length_m``); edges are made undirected and unique.
    """
    def __init__(self, nodes: pl.DataFrame, edges: pl.DataFrame, crs: str):
        self.crs = crs
        ids = nodes["node_id"].to_numpy()
        self.node_index = {int(i): k for k, i in enumerate(ids)}   # only used for small lookups; bulk work uses searchsorted
        order = np.argsort(ids)
        self._ids_sorted, self._ids_pos = ids[order], order
        u = self._pos(edges["u"].to_numpy()); v = self._pos(edges["v"].to_numpy())
        L = edges["length_m"].to_numpy().astype(np.float32)
        # undirected unique edges (the table may hold both directions)
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        key = lo.astype(np.int64) * len(ids) + hi
        _, first = np.unique(key, return_index=True)
        self.eu, self.ev, self.el = lo[first], hi[first], L[first]
        self.geom = shapely.from_wkb(edges["geometry_wkb"].to_numpy()[first])
        self.n = len(ids)
        self.xy = np.column_stack([nodes["x"].to_numpy(), nodes["y"].to_numpy()])
        # CSR with both directions
        src = np.r_[self.eu, self.ev]; dst = np.r_[self.ev, self.eu]; w = np.r_[self.el, self.el]
        o = np.argsort(src, kind="stable")
        self.indptr = np.r_[0, np.cumsum(np.bincount(src, minlength=self.n))].astype(np.int32)
        self.adj, self.w = dst[o].astype(np.int32), w[o].astype(np.float32)

    def _pos(self, node_ids):
        return self._ids_pos[np.searchsorted(self._ids_sorted, node_ids)].astype(np.int32)

    @classmethod
    def load(cls, folder) -> "CsrNetwork":
        """Load a network saved as ``nodes.parquet``, ``edges.parquet`` and ``meta.txt`` (the CRS) in ``folder``.

        Args:
            folder: The network folder.

        Returns:
            The network.
        """
        folder = Path(folder)
        crs = (folder / "meta.txt").read_text().strip().splitlines()[0]
        return cls(pl.read_parquet(folder / "nodes.parquet"), pl.read_parquet(folder / "edges.parquet"), crs)

    # -- stations ------------------------------------------------------------------------------------------------------
    def snap(self, lat, lon, max_snap_m: float = 300.0):
        """Sources for stations at (lat, lon): returns (src_node, src_off, src_station, snapped_mask).

        Args:
            lat: Station latitudes (EPSG:4326).
            lon: Station longitudes (EPSG:4326).
            max_snap_m: Stations farther than this from every edge are not snapped.

        Returns:
            ``(src_node, src_off, src_station, snapped_mask)``: the two end nodes of each snapped station's edge, the
            walking distance from the station to each, the station index of each source, and which stations were
            snapped.
        """
        x, y = Transformer.from_crs(4326, self.crs, always_xy=True).transform(np.asarray(lon), np.asarray(lat))
        pts = shapely.points(x, y)
        tree = shapely.STRtree(self.geom)
        ei = tree.nearest(pts)
        near = self.geom[ei]
        dist = shapely.distance(near, pts)
        pos = shapely.line_locate_point(near, pts)
        ok = dist <= max_snap_m
        L = self.el[ei]
        pos = np.clip(pos, 0, L)
        st = np.flatnonzero(ok)
        src_node = np.r_[self.eu[ei[st]], self.ev[ei[st]]].astype(np.int32)
        src_off = np.r_[dist[st] + pos[st], dist[st] + (L[st] - pos[st])].astype(np.float32)
        src_station = np.r_[st, st].astype(np.int32)
        return src_node, src_off, src_station, ok

    # -- cells ---------------------------------------------------------------------------------------------------------
    def edge_cell_records(self, resolution: int = 10, edge_mask=None):
        """Records of (cell, edge, entry a, exit b): cells = uint64 H3 ids, sorted by cell. Returns (cells_unique, rec_off, ru, rv, ra, rb, rl).
        `edge_mask` (bool per undirected edge, e.g. "an end of the edge was reached by the search") restricts the work to those edges: rural networks have
        millions of edges nobody can reach.

        Args:
            resolution: H3 resolution of the cells.
            edge_mask: Optional boolean per undirected edge: only those edges are sampled.
        """
        from h3ronpy.vector import coordinates_to_cells   # pip install h3ronpy (extra: UrbanAccessAnalyzer[flex])

        sel = np.arange(len(self.el)) if edge_mask is None else np.flatnonzero(edge_mask)
        if len(sel) == 0:
            z = np.zeros(0, dtype=np.int32)
            return np.zeros(0, dtype=np.uint64), np.zeros(1, dtype=np.int32), z, z, z.astype(np.float32), z.astype(np.float32), z.astype(np.float32)
        L = np.maximum(self.el[sel].astype(np.float64), 1e-3)
        n_s = np.maximum(np.ceil(L / SAMPLE_M).astype(np.int64), 1) + 1
        ei = np.repeat(np.arange(len(L)), n_s)
        start = np.r_[0, np.cumsum(n_s)[:-1]]
        k = np.arange(len(ei)) - np.repeat(start, n_s)
        pos = k / np.repeat(n_s - 1, n_s) * L[ei]
        pts = shapely.line_interpolate_point(self.geom[sel][ei], pos)
        c = shapely.get_coordinates(pts)
        lon, lat = Transformer.from_crs(self.crs, 4326, always_xy=True).transform(c[:, 0], c[:, 1])
        cells = np.asarray(coordinates_to_cells(np.asarray(lat), np.asarray(lon), resolution)).astype(np.uint64)
        # (cell, edge) -> min / max position
        o = np.lexsort((ei, cells))
        cells, ei, pos = cells[o], ei[o], pos[o]
        brk = np.flatnonzero(np.r_[True, (cells[1:] != cells[:-1]) | (ei[1:] != ei[:-1]), True])
        a = np.minimum.reduceat(pos, brk[:-1]); b = np.maximum.reduceat(pos, brk[:-1])
        rc, re_ = cells[brk[:-1]], sel[ei[brk[:-1]]]   # back to indexes of the full edge table
        uc, cstart = np.unique(rc, return_index=True)
        rec_off = np.r_[cstart, len(rc)].astype(np.int32)
        return uc, rec_off, self.eu[re_], self.ev[re_], a.astype(np.float32), b.astype(np.float32), self.el[re_]
