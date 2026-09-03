import geopandas as gpd
import polars as pl
import pytest
import shapely

from UrbanAccessAnalyzer import api


@pytest.fixture(scope="module")
def street_network(sample_pbf_path, sample_aoi):
    aoi = api.AreaOfInterest(sample_aoi)
    return api.StreetNetwork.from_pbf(sample_pbf_path, aoi=aoi, network_type="all", simplify_distance=10.0)


def test_street_network_from_pbf(street_network):
    assert street_network.nodes.height > 0
    assert street_network.edges.height > 0
    assert street_network.crs.startswith("EPSG:")


def test_street_network_to_gdf_roundtrip(street_network):
    gdf = street_network.to_gdf()
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == street_network.edges.height


def test_street_network_save_load_roundtrip(street_network, tmp_path):
    folder = tmp_path / "network"
    street_network.save(str(folder))
    loaded = api.StreetNetwork.load(str(folder))

    assert loaded.crs == street_network.crs
    assert loaded.nodes.shape == street_network.nodes.shape
    assert loaded.edges.shape == street_network.edges.shape


def test_points_of_interest_assign_score_by_values():
    df = pl.DataFrame({"geometry": shapely.to_wkb(shapely.points([0.0], [0.0])), "kind": ["school"]})
    poi = api.PointsOfInterest(df).assign_score_by_values("kind", ["school"])
    assert poi.df["poi_score"][0] == 1.0


def test_accessibility_analyzer_end_to_end(street_network):
    xs = street_network.nodes["x"].to_numpy()[:3]
    ys = street_network.nodes["y"].to_numpy()[:3]
    poi_df = pl.DataFrame({"geometry": shapely.to_wkb(shapely.points(xs, ys))})
    poi = api.PointsOfInterest(poi_df)

    analyzer = api.AccessibilityAnalyzer(street_network, poi)
    node_access, edge_access = analyzer.run(
        distance_matrix=[100, 250, 500],
        poi_score_col=None,
        access_score_values=["walk", "bike", "bus/car"],
        min_edge_length=1.0,
        max_dist=100,
        verbose=False,
    )

    assert node_access.height > 0
    assert edge_access.height > 0
    assert set(node_access["access_score"].unique().to_list()) <= {"walk", "bike", "bus/car"}

    gdf = analyzer.to_gdf()
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == edge_access.height

    h3_df = analyzer.to_h3(resolution=13)
    assert h3_df.height > 0
    assert "access_score" in h3_df.columns


def test_compute_accessibility_pools_multiple_poi_kinds(monkeypatch, sample_pbf_path, sample_aoi):
    # Stub out the network-dependent geocoding/Overpass calls so this stays a
    # fast, deterministic, network-free test -- exercises compute_accessibility's
    # own wiring (AOI -> network -> pooled POIs -> analyzer.run -> to_gdf).
    monkeypatch.setattr(
        api.AreaOfInterest, "from_name", classmethod(lambda cls, name, buffer=0.0: cls(sample_aoi))
    )

    def fake_from_overpass(cls, kind, bounds):
        xs, ys = [-0.235, -0.230], [51.765, 51.767]
        df = pl.DataFrame({"geometry": shapely.to_wkb(shapely.points(xs, ys)), "kind": [kind, kind]})
        return cls(df)

    monkeypatch.setattr(api.PointsOfInterest, "from_overpass", classmethod(fake_from_overpass))

    gdf, aoi, points = api.compute_accessibility(
        place="ignored",
        poi_kind=["schools", "shops"],
        pbf_path=sample_pbf_path,
        network_type="all",
        distance_steps=[250, 500],
        simplify_distance=10.0,
        verbose=False,
    )

    assert isinstance(aoi, api.AreaOfInterest)
    assert points.df.height == 4  # 2 POIs from each of the 2 pooled kinds
    assert points.df.columns == ["geometry"]  # only geometry kept when pooling
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert "access_score" in gdf.columns
