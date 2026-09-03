import geopandas as gpd
import polars as pl
import pytest
import shapely
from shapely.geometry import Point, box

from UrbanAccessAnalyzer import geometry_ops, h3_ops


def test_cells_in_geometry_point():
    pts = shapely.points([-0.23, -0.231], [51.766, 51.7661])
    df = pl.DataFrame({"geometry": shapely.to_wkb(pts)})
    result = h3_ops.cells_in_geometry(df, resolution=10)

    assert "h3_cells" in result.columns
    assert result["h3_cells"].list.len().min() >= 1


def test_from_df_sum_preserves_total():
    pts = shapely.points([-0.23, -0.231, -0.232], [51.766, 51.7661, 51.7662])
    df = pl.DataFrame({"geometry": shapely.to_wkb(pts), "value": [1.0, 2.0, 3.0]})

    result = h3_ops.from_df(df, resolution=10, columns=["value"], method="sum")
    assert result["value"].sum() == pytest.approx(6.0)


def test_from_df_density_preserves_total():
    pts = shapely.points([-0.23, -0.2301, -0.2302], [51.766, 51.7661, 51.7662])
    df = pl.DataFrame({"geometry": shapely.to_wkb(pts), "value": [1.0, 2.0, 3.0]})

    result = h3_ops.from_df(df, resolution=8, columns=["value"], method="density")
    assert result["value"].sum() == pytest.approx(6.0, rel=1e-6)


def test_to_gdf_roundtrip():
    pts = shapely.points([-0.23], [51.766])
    df = pl.DataFrame({"geometry": shapely.to_wkb(pts), "value": [5.0]})
    h3_df = h3_ops.from_df(df, resolution=9, columns=["value"], method="sum")

    gdf = h3_ops.to_gdf(h3_df, h3_column="h3_cell")
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs.to_epsg() == 4326
    assert len(gdf) == h3_df.height


def test_cell_polygons_chunked_matches_unchunked(monkeypatch):
    """Shanghai OOM fix: chunked `_cell_polygons` must produce byte-identical
    geometry to a single unchunked call, for a real cell set spanning several
    chunk boundaries. Forces a tiny `_CELL_POLYGON_CHUNK_SIZE` (real 2M is
    impractical in a unit test) so a modest cell count still exercises the
    multi-chunk `np.concatenate` path this fix added."""
    import h3

    # A real neighborhood of valid resolution-9 H3 cells (a disk around a
    # real point), large enough to span multiple 3-cell chunks.
    center = h3.latlng_to_cell(31.23, 121.47, 9)  # central Shanghai
    cells = pl.Series(sorted(h3.grid_disk(center, 3)))  # 37 cells
    assert len(cells) > 9  # comfortably spans >1 chunk at size 3 below

    unchunked = h3_ops._cell_polygons(cells)

    monkeypatch.setattr(h3_ops, "_CELL_POLYGON_CHUNK_SIZE", 3)
    chunked = h3_ops._cell_polygons(cells)

    assert len(chunked) == len(unchunked) == len(cells)
    for a, b in zip(chunked, unchunked):
        assert a.equals_exact(b, tolerance=0)


def test_geometry_ops_aggregate_distribute_preserves_total():
    # row0 (value=10) splits evenly across ids [0, 1] -> 5 each; row1 (value=5) goes wholly to id 1.
    df = pl.DataFrame({"id": [[0, 1], [1]], "value": [10.0, 5.0]})
    result = geometry_ops.aggregate(df, id_column="id", columns=["value"], method="distribute")

    assert result["value"].sum() == pytest.approx(15.0)  # total input value (10 + 5), none lost or duplicated
    assert result.filter(pl.col("id") == 0)["value"][0] == pytest.approx(5.0)
    assert result.filter(pl.col("id") == 1)["value"][0] == pytest.approx(10.0)  # 5 (split) + 5 (whole)


def test_resample_gdf_sum():
    src = gpd.GeoDataFrame(
        {"val": [1.0, 2.0, 3.0]},
        geometry=[Point(0.1, 0.1), Point(0.6, 0.1), Point(1.5, 1.5)],
        crs=4326,
    )
    dst = gpd.GeoDataFrame({"cell_id": [0, 1]}, geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)], crs=4326)

    result = geometry_ops.resample_gdf(src, dst, columns=["val"], method="sum", id_column="cell_id")
    assert result.loc[0, "val"] == pytest.approx(3.0)
    assert result.loc[1, "val"] == pytest.approx(3.0)
