import numpy as np
import polars as pl
import pytest
import shapely

pytest.importorskip("h3ronpy")
from UrbanAccessAnalyzer.csr_network import CsrNetwork

CRS = "EPSG:32618"
X0, Y0 = 327000.0, 4690000.0


def line_network(n=6, step=100.0):
    nodes = pl.DataFrame({"node_id": list(range(n)), "x": [X0 + i * step for i in range(n)], "y": [Y0] * n})
    geom = [shapely.to_wkb(shapely.LineString([(X0 + i * step, Y0), (X0 + (i + 1) * step, Y0)])) for i in range(n - 1)]
    edges = pl.DataFrame({"u": list(range(n - 1)), "v": list(range(1, n)), "length_m": [step] * (n - 1), "geometry_wkb": geom})
    return CsrNetwork(nodes, edges, CRS)


def test_csr_is_undirected():
    net = line_network()
    assert net.n == 6 and len(net.adj) == 10
    assert set(net.adj[net.indptr[2]:net.indptr[3]]) == {1, 3}


def test_snap_gives_two_sources_with_offsets():
    net = line_network()
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(CRS, 4326, always_xy=True).transform(X0 + 130.0, Y0 + 5.0)   # 5 m off the street, 30 m past node 1
    sn, so, ss, ok = net.snap(np.array([lat]), np.array([lon]))
    assert ok[0] and sorted(sn) == [1, 2]
    off = dict(zip(sn, so))
    assert off[1] == pytest.approx(35.0, abs=0.5) and off[2] == pytest.approx(75.0, abs=0.5)      # snap 5 m + 30 m / + 70 m


def test_edge_cell_records_cover_the_street_and_respect_the_mask():
    net = line_network()
    cells, off, ru, rv, ra, rb, rl = net.edge_cell_records(10)
    assert len(cells) >= 3 and off[-1] == len(ru) and (rb >= ra).all()
    c2, off2, *_ = net.edge_cell_records(10, edge_mask=np.array([True, False, False, False, False]))
    assert 0 < off2[-1] < off[-1]
