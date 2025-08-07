import pandas as pd
import geopandas as gpd
import polars as pl
import osmnx as ox
import networkx as nx
from shapely.geometry import Point
import numpy as np
from sklearn.cluster import AgglomerativeClustering

import warnings

"TODO: Compute length in polars"
"TODO: node elevations does not work"

NODE_COLS = ["highway"]
EDGE_COLS = [
    "osmid",
    "highway",
    "oneway",
    "reversed",
    "name",
    "maxspeed",
    "bridge",
    "lanes",
    "ref",
    "junction",
    "access",
    "width",
    "service",
    "tunnel",
    "area",
]


def add_node_elevations_open_api(G):
    orig_template = ox.settings.elevation_url_template
    ox.settings.elevation_url_template = (
        "https://api.open-elevation.com/api/v1/lookup?locations={locations}"
    )
    crs = G.graph["crs"]
    G = ox.projection.project_graph(G, to_latlong=True)
    G = ox.add_node_elevations_google(G, batch_size=250)
    ox.settings.elevation_url_template = orig_template
    G = ox.projection.project_graph(G, to_crs=crs)
    return G


def graph_to_polars(G):
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    nodes_gdf[NODE_COLS] = nodes_gdf[NODE_COLS].astype(str)
    edges_gdf[EDGE_COLS] = edges_gdf[EDGE_COLS].astype(str)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Geometry column does not contain geometry.",
        )

        edges = edges_gdf.reset_index()[
            ["u", "v", "key"] + EDGE_COLS + ["length", "geometry"]
        ]
        edges["geometry"] = edges["geometry"].to_wkt()
        edges["geometry"] = edges["geometry"].astype(str)
        edges_pl = pl.from_pandas(edges)

        nodes = nodes_gdf.reset_index()[["osmid", "x", "y"] + NODE_COLS + ["geometry"]]
        nodes["geometry"] = nodes["geometry"].to_wkt()
        nodes["geometry"] = nodes["geometry"].astype(str)
        nodes_pl = pl.from_pandas(nodes)
    return nodes_pl, edges_pl, nodes_gdf.crs, G.graph


def polars_to_graph(nodes_pl, edges_pl, crs, graph_attrs, compute_length: bool = False):
    if compute_length:
        edges_gdf = edges_pl.to_pandas()[["u", "v", "key"] + EDGE_COLS + ["geometry"]]
    else:
        edges_gdf = edges_pl.to_pandas()[
            ["u", "v", "key"] + EDGE_COLS + ["length", "geometry"]
        ]

    edges_gdf["u"] = edges_gdf["u"].astype(int)
    edges_gdf["v"] = edges_gdf["v"].astype(int)
    edges_gdf["key"] = edges_gdf["key"].astype(int)
    edges_gdf = edges_gdf.set_index(["u", "v", "key"])
    edges_gdf = gpd.GeoDataFrame(
        edges_gdf, geometry=gpd.GeoSeries.from_wkt(edges_gdf["geometry"]), crs=crs
    )
    if compute_length:
        edges_gdf["length"] = edges_gdf.geometry.length

    nodes_gdf = nodes_pl.to_pandas()[["osmid", "x", "y"] + NODE_COLS + ["geometry"]]
    nodes_gdf["osmid"] = nodes_gdf["osmid"].astype(int)
    nodes_gdf = gpd.GeoDataFrame(
        nodes_gdf, geometry=gpd.points_from_xy(nodes_gdf["x"], nodes_gdf["y"]), crs=crs
    )
    nodes_gdf = nodes_gdf.set_index("osmid")

    G = ox.graph_from_gdfs(
        gdf_nodes=nodes_gdf, gdf_edges=edges_gdf, graph_attrs=graph_attrs
    )

    return G


def __connected_node_groups(nodes_pl, edges_pl, max_dist: float | None = None):
    edges_pl = edges_pl.with_columns(
        pl.concat_list(pl.col("u"), pl.col("v")).sort().alias("node_list")
    )

    edges_pl = (
        pl.concat(
            [edges_pl, edges_pl.rename({"u": "v", "v": "u"}).select(edges_pl.columns)]
        )
        .unique(["u", "v"])
        .lazy()
    )

    edges_pl = (
        edges_pl.group_by("u")
        .agg(pl.col("v").alias("node_list"), pl.col("v"))
        .explode("v")
        .group_by("v")
        .agg(pl.col("node_list").flatten(), pl.col("u"))
        .with_columns(
            pl.col("node_list")
            .list.concat(pl.col("u"))
            .list.sort()
            .list.unique()
            .alias("node_list")
        )
        .with_columns(pl.col("node_list").alias("osmid"))
        .explode("osmid")
    )

    prev_count = None

    while True:
        # Collapse node lists by osmid
        edges_pl = (
            edges_pl.group_by("osmid")
            .agg(pl.col("node_list").flatten().sort().unique())
            .with_columns(
                [
                    pl.col("node_list").alias(
                        "osmid"
                    )  # Use merged node list to redefine osmid
                ]
            )
            .explode("osmid")
            .unique()
        )

        current_count = edges_pl.select(pl.col("osmid").n_unique()).collect()[0, 0]

        if prev_count == current_count:
            break
        prev_count = current_count

    edges_pl = edges_pl.collect()

    if max_dist is None:
        edges_pl = (
            edges_pl.group_by("osmid")
            .agg(pl.col("node_list").flatten().min().alias("osmid_group"))
            .join(nodes_pl.select("osmid", "x", "y"), on="osmid", how="left")
            .group_by("osmid_group")
            .agg(
                pl.col("x").mean(),
                pl.col("y").mean(),
                pl.col("osmid"),
                pl.col("osmid").min().alias("new_osmid"),
            )
            .explode("osmid")
        )
    else:

        def cluster(x, y, max_dist):
            coords = np.column_stack((x, y))
            return list(
                AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=max_dist,
                    metric="euclidean",
                    linkage="complete",
                )
                .fit(coords)
                .labels_
            )

        edges_pl = (
            edges_pl.group_by("osmid")
            .agg(pl.col("node_list").flatten().min().alias("osmid_group"))
            .join(nodes_pl.select("osmid", "x", "y"), on="osmid", how="left")
            .group_by("osmid_group")
            .agg(pl.col("x"), pl.col("y"), pl.col("osmid"))
            .with_columns(
                (
                    pl.when(pl.col("x").len() > 2)
                    .then(
                        pl.struct(["x", "y"]).map_elements(
                            lambda row: cluster(row["x"], row["y"], max_dist),
                            return_dtype=list[int],
                        )
                    )
                    .otherwise([0, 0])
                ).alias("cluster_id")
            )
            .explode(["osmid", "x", "y", "cluster_id"])
            .group_by("osmid_group", "cluster_id")
            .agg(
                pl.col("x").mean(),
                pl.col("y").mean(),
                pl.col("osmid"),
                pl.col("osmid").min().alias("new_osmid"),
            )
            .drop("cluster_id")
            .explode("osmid")
        )

    return edges_pl


def __remove_small_edges(nodes_pl, edges_pl, min_edge_length, crs):
    delete_edges = edges_pl.filter(
        (pl.col("length") <= min_edge_length) & (pl.col("u") != pl.col("v"))
    )

    delete_edges = __connected_node_groups(
        nodes_pl, delete_edges, max_dist=min_edge_length * 1.5
    )

    edges_pl = (
        edges_pl.join(
            delete_edges.rename(
                {
                    "osmid": "u",
                    "new_osmid": "new_u",
                    "x": "new_u_x",
                    "y": "new_u_y",
                    "osmid_group": "osmid_group_u",
                }
            ),
            on="u",
            how="left",
        )
        .join(
            delete_edges.rename(
                {
                    "osmid": "v",
                    "new_osmid": "new_v",
                    "x": "new_v_x",
                    "y": "new_v_y",
                    "osmid_group": "osmid_group_v",
                }
            ),
            on="v",
            how="left",
        )
        .filter(
            pl.col("new_u").is_null()
            | pl.col("new_v").is_null()
            | (pl.col("new_u") != pl.col("new_v"))
            | ((pl.col("new_u") == pl.col("new_v")) & (pl.col("u") == pl.col("v")))
        )
        .with_columns(
            pl.when(pl.col("new_u_x").is_not_null())
            .then(
                # Build new LINESTRING with replacement first point
                "LINESTRING ("
                + pl.col("new_u_x").cast(pl.Utf8)
                + " "
                + pl.col("new_u_y").cast(pl.Utf8)
                + ", "
                + pl.col("geometry").str.extract(r"^LINESTRING\s*\([^,]+,\s*(.*)\)$", 1)
                + ")"
            )
            .otherwise(pl.col("geometry"))
            .alias("geometry"),
            (
                pl.when(pl.col("new_u").is_not_null())
                .then(pl.col("new_u"))
                .otherwise(pl.col("u"))
            ).alias("u"),
            (
                pl.when(pl.col("new_u").is_not_null())
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
            ).alias("changed_u"),
        )
        .with_columns(
            (
                pl.when(pl.col("new_v_x").is_not_null())
                .then(
                    # Extract part before the last point
                    pl.col("geometry").str.extract(
                        r"^(LINESTRING\s*\(.*),\s*[^, )]+ [^, )]+\)$", 1
                    )
                    + ", "
                    + pl.col("new_v_x").cast(pl.Utf8)
                    + " "
                    + pl.col("new_v_y").cast(pl.Utf8)
                    + ")"
                )
                .otherwise(pl.col("geometry"))
            ).alias("geometry"),
            (
                pl.when(pl.col("new_v").is_not_null())
                .then(pl.col("new_v"))
                .otherwise(pl.col("v"))
            ).alias("v"),
            (
                pl.when(pl.col("new_v").is_not_null())
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
            ).alias("changed_v"),
        )
        .with_columns((pl.col("u").cum_count().over(["u", "v"]) - 1).alias("key"))
        .drop("new_u_x", "new_u_y", "new_u", "new_v_x", "new_v_y", "new_v")
    )

    nodes_pl = (
        nodes_pl.join(
            delete_edges.rename({"x": "new_x", "y": "new_y"}), on="osmid", how="left"
        )
        .with_columns(
            (
                pl.when(pl.col("new_osmid").is_not_null())
                .then(pl.col("new_x"))
                .otherwise(pl.col("x"))
            ).alias("x"),
            (
                pl.when(pl.col("new_osmid").is_not_null())
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
            ).alias("grouped_node"),
            (
                pl.when(pl.col("new_osmid").is_not_null())
                .then(pl.col("new_y"))
                .otherwise(pl.col("y"))
            ).alias("y"),
            (
                pl.when(pl.col("new_osmid").is_not_null())
                .then(pl.col("new_osmid"))
                .otherwise(pl.col("osmid"))
            ).alias("osmid"),
        )
        .unique("osmid")
        .drop("new_x", "new_y", "new_osmid")
    )

    edges_gdf = edges_pl.to_pandas()
    edges_gdf["u"] = edges_gdf["u"].astype(int)
    edges_gdf["v"] = edges_gdf["v"].astype(int)
    edges_gdf["key"] = edges_gdf["key"].astype(int)
    edges_gdf = gpd.GeoDataFrame(
        edges_gdf, geometry=gpd.GeoSeries.from_wkt(edges_gdf["geometry"]), crs=crs
    )
    edges_gdf["length"] = edges_gdf.geometry.length

    df_with_group = edges_gdf[
        edges_gdf["osmid_group_u"].notna()
        & edges_gdf["osmid_group_v"].notna()
        & (edges_gdf["osmid_group_u"] == edges_gdf["osmid_group_v"])
    ]
    df_without_group = edges_gdf[
        edges_gdf["osmid_group_u"].isna()
        | edges_gdf["osmid_group_v"].isna()
        | (edges_gdf["osmid_group_u"] != edges_gdf["osmid_group_v"])
    ]

    # Keep the one with the smallest 'length' for each (u, v)
    df_with_group = df_with_group.sort_values("length").drop_duplicates(
        subset=["u", "v"], keep="first"
    )
    df_with_group["key"] = 0
    edges_gdf = pd.concat([df_with_group, df_without_group], ignore_index=True)

    edges_gdf = edges_gdf[["u", "v", "key"] + EDGE_COLS + ["length", "geometry"]]
    edges_gdf = edges_gdf.set_index(["u", "v", "key"])

    nodes_gdf = nodes_pl.to_pandas()[["osmid", "x", "y"] + NODE_COLS + ["geometry"]]
    nodes_gdf["osmid"] = nodes_gdf["osmid"].astype(int)
    nodes_gdf = gpd.GeoDataFrame(
        nodes_gdf, geometry=gpd.points_from_xy(nodes_gdf["x"], nodes_gdf["y"]), crs=crs
    )
    nodes_gdf = nodes_gdf.set_index("osmid")

    return nodes_gdf, edges_gdf


def __remove_near_edges(edges_gdf, max_dist):
    def cluster(x, y, max_dist):
        coords = np.column_stack((x, y))
        return list(
            AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=max_dist,
                metric="euclidean",
                linkage="complete",
            )
            .fit(coords)
            .labels_
        )

    edges_gdf["center"] = edges_gdf.geometry.interpolate(edges_gdf["length"] / 2)
    edges_gdf["center_x"] = edges_gdf["center"].x
    edges_gdf["center_y"] = edges_gdf["center"].y
    edges_gdf = edges_gdf.reset_index()
    edges_groups = (
        edges_gdf.groupby(["u", "v"])
        .agg(
            key=("key", list),
            center_x=("center_x", list),
            center_y=("center_y", list),
            n_keys=("key", "count"),
        )
        .reset_index()
    )
    edges_gdf = edges_gdf.drop(columns=["center", "center_x", "center_y"])

    edges_groups["cluster_id"] = [[0]] * len(edges_groups)

    mask = edges_groups["n_keys"] == 2
    edges_groups.loc[mask, "cluster_id"] = edges_groups[mask].apply(
        lambda row: [0, 0]
        if (
            (
                (row["center_x"][0] - row["center_x"][1]) ** 2
                + (row["center_y"][0] - row["center_y"][1]) ** 2
            )
            < (max_dist**2)
        )
        else [0, 1],
        axis=1,
    )

    mask = edges_groups["n_keys"] > 2
    edges_groups.loc[mask, "cluster_id"] = edges_groups[mask].apply(
        lambda row: cluster(row["center_x"], row["center_y"], max_dist=max_dist), axis=1
    )
    edges_groups = edges_groups.drop(columns=["center_x", "center_y"]).explode(
        ["key", "cluster_id"]
    )

    edges_gdf = edges_gdf.merge(
        edges_groups[["u", "v", "key", "cluster_id"]], on=["u", "v", "key"], how="left"
    )
    edges_gdf = edges_gdf.sort_values(
        ["u", "v", "cluster_id", "length"]
    ).drop_duplicates(["u", "v", "cluster_id"], keep="first")
    edges_gdf = edges_gdf.drop(columns="key").rename(columns={"cluster_id": "key"})
    edges_gdf["key"] = edges_gdf["key"].astype(int)
    edges_gdf = edges_gdf.set_index(["u", "v", "key"])
    return edges_gdf


def simplify_graph(
    G,
    min_edge_length=0,
    min_edge_separation=0,
    loops: bool = True,
    multi: bool = True,
    undirected: bool = False,
):
    nodes_pl, edges_pl, crs, graph_attrs = graph_to_polars(G)
    if not loops:
        edges_pl = edges_pl.filter(pl.col("u") != pl.col("v"))

    if not multi:
        if undirected:
            edges_pl = (
                edges_pl.with_columns(
                    pl.concat_list([pl.col("u"), pl.col("v")])
                    .list.sort()
                    .alias("sorted_nodes")
                )
                .sort(by=["sorted_nodes", "length"])
                .unique(subset=["sorted_nodes"], keep="first")
                .with_columns(pl.lit(0).alias("key"))
                .drop("sorted_nodes")
            )
        else:
            edges_pl = (
                edges_pl.sort(by=["u", "v", "length"])
                .unique(subset=["u", "v"], keep="first")
                .with_columns(pl.lit(0).alias("key"))
            )

    elif undirected:
        edges_pl = (
            edges_pl.with_columns(
                pl.concat_list([pl.col("u"), pl.col("v")])
                .list.sort()
                .alias("sorted_nodes")
            )
            .unique(subset=["sorted_nodes", "length", "maxspeed"])
            .with_columns(
                pl.col("sorted_nodes").list.get(0).alias("new_u"),
                pl.col("sorted_nodes").list.get(1).alias("new_v"),
                (pl.col("sorted_nodes").cum_count().over(["sorted_nodes"]) - 1).alias(
                    "key"
                ),
            )
            .drop("sorted_nodes")
            .with_columns(
                (
                    pl.when(pl.col("u") == pl.col("new_u"))
                    .then(pl.col("geometry"))
                    .otherwise(
                        pl.lit("LINESTRING (")
                        + pl.col("geometry")
                        .str.replace("LINESTRING \\(", "")
                        .str.replace("\\)$", "")
                        .str.split(", ")
                        .list.eval(pl.element().str.strip_chars())
                        .list.reverse()
                        .list.join(", ")
                        + pl.lit(")")
                    )
                ).alias("geometry"),
                pl.col("new_u").alias("u"),
                pl.col("new_v").alias("v"),
            )
            .drop(["new_u", "new_v"])
        )

    if min_edge_length > 0:
        nodes_gdf, edges_gdf = __remove_small_edges(
            nodes_pl, edges_pl, min_edge_length=min_edge_length, crs=crs
        )
        G = ox.graph_from_gdfs(
            gdf_nodes=nodes_gdf, gdf_edges=edges_gdf, graph_attrs=graph_attrs
        )
        G = simplify_graph(
            G,
            min_edge_length=0,
            min_edge_separation=min_edge_separation,
            loops=loops,
            multi=multi,
            undirected=undirected,
        )
    else:
        if multi and (min_edge_separation > 0):
            edges_gdf = edges_pl.to_pandas()
            edges_gdf["u"] = edges_gdf["u"].astype(int)
            edges_gdf["v"] = edges_gdf["v"].astype(int)
            edges_gdf["key"] = edges_gdf["key"].astype(int)
            edges_gdf = gpd.GeoDataFrame(
                edges_gdf,
                geometry=gpd.GeoSeries.from_wkt(edges_gdf["geometry"]),
                crs=crs,
            )
            edges_gdf = edges_gdf[
                ["u", "v", "key"] + EDGE_COLS + ["length", "geometry"]
            ]
            edges_gdf = edges_gdf.set_index(["u", "v", "key"])

            nodes_gdf = nodes_pl.to_pandas()[
                ["osmid", "x", "y"] + NODE_COLS + ["geometry"]
            ]
            nodes_gdf["osmid"] = nodes_gdf["osmid"].astype(int)
            nodes_gdf = gpd.GeoDataFrame(
                nodes_gdf,
                geometry=gpd.points_from_xy(nodes_gdf["x"], nodes_gdf["y"]),
                crs=crs,
            )
            nodes_gdf = nodes_gdf.set_index("osmid")

            edges_gdf = __remove_near_edges(edges_gdf, max_dist=min_edge_separation)

            G = ox.graph_from_gdfs(
                gdf_nodes=nodes_gdf, gdf_edges=edges_gdf, graph_attrs=graph_attrs
            )
        else:
            G = polars_to_graph(nodes_pl, edges_pl, crs, graph_attrs)

    return G


def nodes_to_points(nodes, G):
    # Get point geometries for the given nodes
    point_geometries = [
        (node, Point((G.nodes[node]["x"], G.nodes[node]["y"])))
        for node in nodes
        if node in G.nodes
    ]
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(point_geometries, columns=["osmid", "geometry"])
    # Get the CRS from the graph
    crs = G.graph["crs"]
    # Set the coordinate reference system (CRS) from the graph
    gdf.set_crs(crs, inplace=True)
    return gdf


def nearest_nodes(
    geometries: gpd.GeoDataFrame | gpd.GeoSeries, G, max_dist: float | None = None
):
    nodes = ox.graph_to_gdfs(G, edges=False)
    geom = geometries.to_crs(nodes.crs).copy()

    # Find nearest nodes
    indices = nodes.sindex.nearest(
        geom.geometry, max_distance=max_dist, return_all=False
    )
    # Create a DataFrame to store nearest node results
    geom["index"] = list(geom.index)
    geom = geom.reset_index(drop=True)
    geom.loc[indices[0], "node_id"] = nodes.iloc[indices[1]].index
    geom = geom.drop_duplicates("index").reset_index(drop=True)
    return list(geom["node_id"].astype(int))


def nearest_edges(
    geometries: gpd.GeoDataFrame | gpd.GeoSeries, G, max_dist: float | None = None
):
    edges = ox.graph_to_gdfs(G, nodes=False)
    geom = geometries.geometry.to_crs(edges.crs)
    indices = edges.sindex.nearest(geom, max_distance=max_dist)
    geom["index"] = list(geom.index)
    geom = geom.reset_index(drop=True)
    geom.loc[indices[0], "edge_id"] = edges.iloc[indices[1]].index
    geom = geom.drop_duplicates("index").reset_index(drop=True)
    return list(geom["edge_id"])


def __polars_linestring_to_points(df, id_col=["u", "v", "key"], length: bool = False):
    df = df.lazy()
    df = (
        df.with_columns(
            [
                # Remove 'LINESTRING(' and ')' and split into point strings
                pl.col("geometry")
                .str.replace_all(r"LINESTRING\s*\(", "")
                .str.replace_all(r"\)", "")
                .str.split(", ")
                .alias("point_list")
            ]
        )
        .explode("point_list")
        .with_columns(
            [
                # pt_sequence based on position after explode
                pl.col("point_list").cum_count().over(id_col).alias("pt_sequence"),
                # Split each point into x and y
                pl.col("point_list").str.split(" "),
            ]
        )
        .with_columns(
            [
                pl.col("point_list").list.get(0).cast(pl.Float64).alias("pt_x"),
                pl.col("point_list").list.get(1).cast(pl.Float64).alias("pt_y"),
            ]
        )
    ).drop("point_list", "geometry")

    if length:
        # Compute Euclidean distance between consecutive points per edge
        # First, create lagged x and y
        df = (
            df.sort(id_col, "pt_sequence")
            .with_columns(
                pl.col("pt_x").shift(1).over(id_col).alias("prev_x"),
                pl.col("pt_y").shift(1).over(id_col).alias("prev_y"),
            )
            .with_columns(
                (
                    (pl.col("pt_x") - pl.col("prev_x")) ** 2
                    + (pl.col("pt_y") - pl.col("prev_y")) ** 2
                )
                .sqrt()
                .alias("length")
            )
            .drop("prev_x", "prev_y")
            .with_columns(pl.col("length").cum_sum().over(id_col).alias("length"))
        )

    return df.collect()


def __split_at_edges(nodes_gdf, edges_gdf, new_edges_gdf):
    edges_gdf = edges_gdf[~edges_gdf.index.isin(list(new_edges_gdf.index))]

    new_edges_gdf = new_edges_gdf.reset_index()
    new_edges_gdf["edge_index"] = new_edges_gdf.index

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Geometry column does not contain geometry.",
        )
        edge_selection = new_edges_gdf[
            ["u", "v", "key", "edge_index"] + EDGE_COLS + ["length", "geometry"]
        ].copy()
        edge_selection["geometry"] = edge_selection["geometry"].to_wkt()
        edge_selection["geometry"] = edge_selection["geometry"].astype(str)
        edge_selection[EDGE_COLS] = edge_selection[EDGE_COLS].astype(str)
        edge_selection_pl = pl.from_pandas(edge_selection)

    new_edges = new_edges_gdf[
        ["u", "v", "key", "edge_index"]
        + EDGE_COLS
        + ["length", "geometry"]
        + ["new_node_id", "point"]
    ].copy()
    new_edges[EDGE_COLS] = new_edges[EDGE_COLS].astype(str)
    new_edges["pt_x"] = new_edges["point"].get_coordinates()["x"]
    new_edges["pt_y"] = new_edges["point"].get_coordinates()["y"]
    new_edges = new_edges.drop(columns=["point", "geometry"])
    new_edges_pl = pl.from_pandas(new_edges)

    new_nodes_gdf = gpd.GeoDataFrame(
        new_edges_gdf[NODE_COLS + ["new_node_id"]],
        geometry=new_edges_gdf["point"],
        crs=nodes_gdf.crs,
    )
    new_nodes_gdf["x"] = new_nodes_gdf["geometry"].get_coordinates()["x"]
    new_nodes_gdf["y"] = new_nodes_gdf["geometry"].get_coordinates()["y"]
    new_nodes_gdf = new_nodes_gdf.rename(columns={"new_node_id": "osmid"})
    for c in NODE_COLS:
        if c not in new_nodes_gdf.columns:
            new_nodes_gdf[c] = None

    new_nodes_gdf = new_nodes_gdf[["osmid", "x", "y"] + NODE_COLS + ["geometry"]]
    new_nodes_gdf[NODE_COLS] = new_nodes_gdf[NODE_COLS].astype(str)
    new_nodes_gdf = new_nodes_gdf.drop_duplicates("osmid")
    new_nodes_gdf = new_nodes_gdf.set_index("osmid")

    edge_selection_pl = __polars_linestring_to_points(
        edge_selection_pl, id_col="edge_index", length=True
    )
    edge_selection_pl = edge_selection_pl.with_columns(
        pl.lit(None).cast(int).alias("new_node_id")
    )
    edge_selection_pl = edge_selection_pl.with_columns(
        pl.col("pt_sequence").cast(int).alias("pt_sequence")
    )

    new_edges_pl = new_edges_pl.with_columns(
        pl.lit(None).cast(int).alias("pt_sequence")
    )

    columns = (
        ["u", "v", "key", "edge_index"]
        + EDGE_COLS
        + ["length", "pt_x", "pt_y", "pt_sequence", "new_node_id"]
    )
    new_edges_pl = new_edges_pl.select(columns)
    edge_selection_pl = edge_selection_pl.select(columns)
    edge_selection_pl = pl.concat([edge_selection_pl, new_edges_pl, new_edges_pl])

    edge_selection_pl = (
        edge_selection_pl.sort(["edge_index", "length"])  # Ensure proper order
        .with_columns(
            # Compute cumulative sum of nulls per edge_index
            (
                pl.col("pt_sequence")
                .is_null()
                .cast(pl.Int8)
                .cum_sum()
                .over("edge_index")
                / 2
            )
            .floor()
            .cast(pl.Int8)
            .alias("new_point_count")
        )
        .with_columns(
            # Concatenate to form something like "edge123_2"
            pl.concat_str(
                [
                    pl.col("edge_index").cast(pl.Utf8),
                    pl.lit("_"),
                    pl.col("new_point_count").cast(pl.Utf8),
                ]
            ).alias("edge_index"),
            pl.concat_str(
                [
                    pl.col("pt_x").cast(pl.Utf8),
                    pl.lit(" "),
                    pl.col("pt_y").cast(pl.Utf8),
                ]
            ).alias("points"),
        )
        .group_by("edge_index")
        .agg(
            pl.col("new_node_id").sort_by("length").first().alias("new_u"),
            pl.col("new_node_id").sort_by("length").last().alias("new_v"),
            pl.col(["u", "v", "key"] + EDGE_COLS).first(),
            pl.col("points").sort_by("length"),
        )
        .with_columns(
            pl.concat_str(
                [pl.lit("LINESTRING("), pl.col("points").list.join(", "), pl.lit(")")]
            ).alias("geometry"),
            (
                pl.when(pl.col("new_u").is_not_null())
                .then(pl.col("new_u"))
                .otherwise(pl.col("u"))
            ).alias("u"),
            (
                pl.when(pl.col("new_v").is_not_null())
                .then(pl.col("new_v"))
                .otherwise(pl.col("v"))
            ).alias("v"),
        )
        .drop("points", "new_u", "new_v")
        .with_columns((pl.col("u").cum_count().over(["u", "v"]) - 1).alias("key"))
    ).drop("edge_index")

    new_edges_gdf = edge_selection_pl.to_pandas()
    new_edges_gdf = gpd.GeoDataFrame(
        new_edges_gdf,
        geometry=gpd.GeoSeries.from_wkt(new_edges_gdf["geometry"]),
        crs=edges_gdf.crs,
    )
    new_edges_gdf["length"] = new_edges_gdf.length
    new_edges_gdf = new_edges_gdf[
        ["u", "v", "key"] + EDGE_COLS + ["length", "geometry"]
    ]
    new_edges_gdf["u"] = new_edges_gdf["u"].astype(int)
    new_edges_gdf["v"] = new_edges_gdf["v"].astype(int)
    new_edges_gdf["key"] = new_edges_gdf["key"].astype(int)
    new_edges_gdf = new_edges_gdf.set_index(["u", "v", "key"])

    edges_gdf = pd.concat([new_edges_gdf, edges_gdf])

    nodes_gdf = pd.concat([nodes_gdf, new_nodes_gdf.drop_duplicates()])

    return nodes_gdf, edges_gdf


def add_points_to_graph(
    points: gpd.GeoDataFrame | gpd.GeoSeries,
    G,
    max_dist: float | None = None,
    min_edge_length: float = 0,
):
    if len(points) == 0:
        return G, []

    graph_attrs = G.graph
    points = points.to_crs(G.graph["crs"]).copy()

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    nodes_gdf[NODE_COLS] = nodes_gdf[NODE_COLS].astype(str)
    edges_gdf[EDGE_COLS] = edges_gdf[EDGE_COLS].astype(str)

    nearest_indices = edges_gdf.sindex.nearest(points.geometry, max_distance=max_dist)
    new_edges_gdf = edges_gdf.iloc[nearest_indices[1, :]]
    new_edges_gdf = new_edges_gdf.reset_index()
    new_edges_gdf["edge_index"] = new_edges_gdf.index
    points = points.iloc[nearest_indices[0, :]]

    new_edges_gdf["projected_dist"] = new_edges_gdf.project(
        points.geometry, align=False
    )

    if ("id" in points.columns) and (points["id"].dtype == int):
        new_edges_gdf["new_node_id"] = list(points["id"])

    # Remove points too close to edge limits (less than min_dist)
    new_edges_gdf = new_edges_gdf.loc[new_edges_gdf["projected_dist"] > min_edge_length]
    new_edges_gdf = new_edges_gdf.loc[
        (new_edges_gdf.geometry.length - new_edges_gdf["projected_dist"])
        > min_edge_length
    ]

    new_edges_gdf = new_edges_gdf.reset_index(drop=True)

    if len(new_edges_gdf) == 0:
        if max_dist is None:
            return G, nearest_nodes(points, G)  # This is not the most efficient way
        else:
            return G, nearest_nodes(
                points, G, max_dist=min_edge_length + max_dist + 0.01
            )  # This is not the most efficient way

    if ("id" in points.columns) and (points["id"].dtype == int):
        if any(points["id"].isin(nodes_gdf.index)):
            warnings.warn(
                "Some of the ids in points column 'id' are in nodes 'osmid'. Using default ids."
            )
            min_id = nodes_gdf.index.max() + 1
            new_edges_gdf["new_node_id"] = min_id + points.index
    else:
        min_id = nodes_gdf.index.max() + 1
        new_edges_gdf["new_node_id"] = min_id + points.index

    new_edges_gdf["point_edge_id"] = round(
        new_edges_gdf["projected_dist"] / min_edge_length
    ).astype(int)
    new_edges_gdf["projected_dist"] = new_edges_gdf.groupby(
        ["edge_index", "point_edge_id"]
    )["projected_dist"].transform("mean")
    new_edges_gdf = new_edges_gdf.drop_duplicates(
        ["edge_index", "projected_dist"]
    ).sort_values(["edge_index", "projected_dist"])

    new_edges_gdf["diff"] = new_edges_gdf.groupby("edge_index")["projected_dist"].diff()
    new_edges_gdf.loc[new_edges_gdf["diff"] < min_edge_length, "point_edge_id"] -= 1
    new_edges_gdf["projected_dist"] = new_edges_gdf.groupby(
        ["edge_index", "point_edge_id"]
    )["projected_dist"].transform("mean")
    new_edges_gdf = new_edges_gdf.drop_duplicates(["edge_index", "projected_dist"])

    new_edges_gdf["point"] = new_edges_gdf.interpolate(new_edges_gdf["projected_dist"])
    new_edges_gdf["length"] = new_edges_gdf["projected_dist"]

    new_edges_gdf["u"] = new_edges_gdf["u"].astype(int)
    new_edges_gdf["v"] = new_edges_gdf["v"].astype(int)
    new_edges_gdf["key"] = new_edges_gdf["key"].astype(int)
    new_edges_gdf = new_edges_gdf.set_index(["u", "v", "key"])

    nodes_gdf, edges_gdf = __split_at_edges(nodes_gdf, edges_gdf, new_edges_gdf)

    G = ox.graph_from_gdfs(
        gdf_nodes=nodes_gdf, gdf_edges=edges_gdf, graph_attrs=graph_attrs
    )

    if max_dist is None:
        return G, nearest_nodes(points, G)  # This is not the most efficient way
    else:
        return G, nearest_nodes(
            points, G, max_dist=min_edge_length + max_dist + 0.01
        )  # This is not the most efficient way


def __multi_ego_graph(
    G,
    n,
    radius: float = 1,
    center: bool = True,
    undirected: bool = False,
    distance: str = "length",
):
    """Returns induced subgraph of neighbors centered at node n within
    a given radius.

    Parameters
    ----------
    G : graph
      A NetworkX Graph or DiGraph

    n : node
      A single node or multiple

    radius : number, optional
      Include all neighbors of distance<=radius from n.

    center : bool, optional
      If False, do not include center node in graph

    undirected : bool, optional
      If True use both in- and out-neighbors of directed graphs.

    distance : key, optional
      Use specified edge data key as distance.  For example, setting
      distance='weight' will use the edge weight to measure the
      distance from the node n.

    Notes
    -----
    For directed graphs D this produces the "out" neighborhood
    or successors.  If you want the neighborhood of predecessors
    first reverse the graph with D.reverse().  If you want both
    directions use the keyword argument undirected=True.

    Node, edge, and graph attributes are copied to the returned subgraph.
    """
    if undirected:
        if isinstance(distance, str):
            sp, _ = nx.multi_source_dijkstra(
                G.to_undirected(), n, cutoff=radius, weight=distance
            )
        else:
            sp = dict(
                nx.multi_source_dijkstra_path_length(
                    G.to_undirected(), n, cutoff=radius
                )
            )
    else:
        if isinstance(distance, str):
            sp, _ = nx.multi_source_dijkstra(G, n, cutoff=radius, weight=distance)
        else:
            sp = dict(nx.multi_source_dijkstra_path_length(G, n, cutoff=radius))

    H = G.subgraph(sp).copy()
    nx.set_node_attributes(H, sp, "dist_to_center")
    if not center:
        H.remove_node(n)

    return H


def crop_graph_by_iso_nodes(
    G=None,
    node_ids=[],
    border_node_ids=[],
    min_edge_length: float = 0,
    undirected: bool = False,
    outbound: bool = True,
    nodes_gdf=None,
    edges_gdf=None,
    graph_attrs=None,
):
    if G is None:
        if (nodes_gdf is None) or (edges_gdf is None) or (graph_attrs is None):
            raise Exception("Eather provide G or nodes_gdf, edges_gdf and graph_attrs")
    else:
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
        graph_attrs = G.graph

    if len(node_ids) == 0:
        return nx.MultiDiGraph(attr=graph_attrs)

    crs = graph_attrs["crs"]

    nodes_gdf = nodes_gdf.loc[node_ids + border_node_ids]

    edges_gdf = edges_gdf.reset_index()

    if undirected:
        edges_gdf = edges_gdf[
            (edges_gdf["u"].isin(node_ids) & edges_gdf["v"].isin(node_ids))
            | (edges_gdf["u"].isin(node_ids) & edges_gdf["v"].isin(border_node_ids))
            | (edges_gdf["u"].isin(border_node_ids) & edges_gdf["v"].isin(node_ids))
            | (
                (
                    edges_gdf["u"].isin(border_node_ids)
                    & edges_gdf["v"].isin(border_node_ids)
                )
                & (edges_gdf["length"] < (2 * min_edge_length))
            )
        ]
    elif outbound:
        edges_gdf = edges_gdf[
            (edges_gdf["u"].isin(node_ids) & edges_gdf["v"].isin(node_ids))
            | (edges_gdf["u"].isin(node_ids) & edges_gdf["v"].isin(border_node_ids))
            | (
                (
                    edges_gdf["u"].isin(border_node_ids)
                    & edges_gdf["v"].isin(border_node_ids)
                )
                & (edges_gdf["length"] < (2 * min_edge_length))
            )
        ]
    else:
        edges_gdf = edges_gdf[
            (edges_gdf["u"].isin(node_ids) & edges_gdf["v"].isin(node_ids))
            | (edges_gdf["u"].isin(border_node_ids) & edges_gdf["v"].isin(node_ids))
            | (
                (
                    edges_gdf["u"].isin(border_node_ids)
                    & edges_gdf["v"].isin(border_node_ids)
                )
                & (edges_gdf["length"] < (2 * min_edge_length))
            )
        ]

    edges_gdf = edges_gdf.set_index(["u", "v", "key"])

    nodes_gdf = nodes_gdf.to_crs(crs)
    edges_gdf = edges_gdf.to_crs(crs)

    return ox.graph_from_gdfs(nodes_gdf, edges_gdf, graph_attrs=graph_attrs)


def isochrone(
    G,
    nodes,
    radius,
    distance_column="length",
    min_edge_length: float = 0.001,
    undirected: bool = False,
    exact: bool = True,
    outbound: bool = True,
    crop_graph: bool = True,
):
    H = __multi_ego_graph(
        G, nodes, radius, center=True, undirected=undirected, distance=distance_column
    )

    nodes_iso_gdf = ox.graph_to_gdfs(H, edges=False)
    node_ids = list(nodes_iso_gdf.index)
    nodes_iso_gdf = nodes_iso_gdf.reset_index()

    if not exact:
        if crop_graph:
            return H
        else:
            return G, node_ids

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    edges_border_gdf = edges_gdf.reset_index().copy()
    if undirected or outbound:
        edges_border_gdf = edges_border_gdf.merge(
            nodes_iso_gdf[["osmid", "dist_to_center"]].rename(
                columns={"osmid": "u", "dist_to_center": "remaining_dist_u"}
            ),
            on="u",
            how="left",
        )
        if outbound and (not undirected):
            edges_border_gdf["remaining_dist_v"] = None

    if undirected or (not outbound):
        edges_border_gdf = edges_border_gdf.merge(
            nodes_iso_gdf[["osmid", "dist_to_center"]].rename(
                columns={"osmid": "v", "dist_to_center": "remaining_dist_v"}
            ),
            on="v",
            how="left",
        )
        if (not outbound) and (not undirected):
            edges_border_gdf["remaining_dist_u"] = None

    edges_border_gdf["remaining_dist_u"] = radius - edges_border_gdf["remaining_dist_u"]
    edges_border_gdf["remaining_dist_v"] = radius - edges_border_gdf["remaining_dist_v"]
    edges_border_gdf["remaining_dist_u"] = edges_border_gdf["remaining_dist_u"].fillna(
        0
    )
    edges_border_gdf["remaining_dist_v"] = edges_border_gdf["remaining_dist_v"].fillna(
        0
    )
    edges_border_gdf = edges_border_gdf[
        (edges_border_gdf["remaining_dist_u"] > 0)
        | (edges_border_gdf["remaining_dist_v"] > 0)
    ]
    edges_border_gdf = edges_border_gdf[
        (edges_border_gdf["remaining_dist_u"] < edges_border_gdf["length"])
        & (edges_border_gdf["remaining_dist_v"] < edges_border_gdf["length"])
    ]

    edges_border_gdf = edges_border_gdf[
        (
            (edges_border_gdf["remaining_dist_u"] > 0)
            & (edges_border_gdf["remaining_dist_v"] > 0)
            & (
                (
                    edges_border_gdf["remaining_dist_u"]
                    + edges_border_gdf["remaining_dist_v"]
                )
                < (edges_border_gdf["length"] - min_edge_length)
            )
        )
        | (
            (edges_border_gdf["remaining_dist_u"] == 0)
            | (edges_border_gdf["remaining_dist_v"] == 0)
        )
    ]

    border_node_ids = []
    border_node_ids += list(
        edges_border_gdf.loc[
            (edges_border_gdf["remaining_dist_u"] <= min_edge_length)
            & (edges_border_gdf["remaining_dist_u"] > 0),
            "u",
        ]
    )
    border_node_ids += list(
        edges_border_gdf.loc[
            (
                (edges_border_gdf["length"] - edges_border_gdf["remaining_dist_u"])
                <= min_edge_length
            )
            & (edges_border_gdf["remaining_dist_u"] > 0),
            "v",
        ]
    )
    border_node_ids += list(
        edges_border_gdf.loc[
            (edges_border_gdf["remaining_dist_v"] <= min_edge_length)
            & (edges_border_gdf["remaining_dist_v"] > 0),
            "v",
        ]
    )
    border_node_ids += list(
        edges_border_gdf.loc[
            (
                (edges_border_gdf["length"] - edges_border_gdf["remaining_dist_v"])
                <= min_edge_length
            )
            & (edges_border_gdf["remaining_dist_v"] > 0),
            "u",
        ]
    )

    edges_border_gdf = edges_border_gdf[
        (edges_border_gdf["remaining_dist_u"] > min_edge_length)
        | (edges_border_gdf["remaining_dist_v"] > min_edge_length)
    ]
    edges_border_gdf = edges_border_gdf[
        (
            (edges_border_gdf["length"] - edges_border_gdf["remaining_dist_u"])
            > min_edge_length
        )
        | (
            (edges_border_gdf["length"] - edges_border_gdf["remaining_dist_v"])
            > min_edge_length
        )
    ]

    # Compute interpolation distance
    edges_border_gdf["projected_dist_u"] = edges_border_gdf["remaining_dist_u"]
    edges_border_gdf["projected_dist_v"] = (
        edges_border_gdf["length"] - edges_border_gdf["remaining_dist_v"]
    )

    edges_border_gdf = edges_border_gdf.drop(
        columns=[
            "remaining_dist_u",
            "remaining_dist_v",
            "remaining_dist_u",
            "remaining_dist_v",
        ]
    )

    edges_border_gdf = pd.melt(
        edges_border_gdf,
        id_vars=[
            col
            for col in edges_border_gdf.columns
            if col not in ["projected_dist_u", "projected_dist_v"]
        ],
        value_vars=["projected_dist_u", "projected_dist_v"],
        var_name="source",
        value_name="projected_dist",
    ).drop(columns=["source"])

    edges_border_gdf = edges_border_gdf[
        (edges_border_gdf["projected_dist"] > min_edge_length)
        & (
            (edges_border_gdf["length"] - edges_border_gdf["projected_dist"])
            > min_edge_length
        )
    ]

    edges_border_gdf["point"] = edges_border_gdf.interpolate(
        edges_border_gdf["projected_dist"]
    )
    edges_border_gdf["length"] = edges_border_gdf["projected_dist"]

    min_id = nodes_gdf.index.max() + 1
    new_border_node_ids = list(min_id + np.arange(0, len(edges_border_gdf)))
    edges_border_gdf["new_node_id"] = new_border_node_ids
    border_node_ids += new_border_node_ids

    edges_border_gdf["u"] = edges_border_gdf["u"].astype(int)
    edges_border_gdf["v"] = edges_border_gdf["v"].astype(int)
    edges_border_gdf["key"] = edges_border_gdf["key"].astype(int)
    edges_border_gdf = edges_border_gdf.set_index(["u", "v", "key"])

    if len(edges_border_gdf) == 0:
        if crop_graph:
            return H
        else:
            return G, node_ids, []

    nodes_gdf, edges_gdf = __split_at_edges(nodes_gdf, edges_gdf, edges_border_gdf)
    if crop_graph:
        nodes_gdf = nodes_gdf.loc[node_ids + border_node_ids]
        if undirected:
            edges_gdf = edges_gdf[
                (
                    (
                        edges_gdf.reset_index()["u"].isin(node_ids)
                        & edges_gdf.reset_index()["v"].isin(node_ids)
                    )
                    | (
                        edges_gdf.reset_index()["u"].isin(node_ids)
                        | edges_gdf.reset_index()["v"].isin(border_node_ids)
                    )
                    | (
                        edges_gdf.reset_index()["u"].isin(border_node_ids)
                        | edges_gdf.reset_index()["v"].isin(node_ids)
                    )
                )
            ]
        elif outbound:
            edges_gdf = edges_gdf[
                (
                    (
                        edges_gdf.reset_index()["u"].isin(node_ids)
                        & edges_gdf.reset_index()["v"].isin(node_ids)
                    )
                    | (
                        edges_gdf.reset_index()["u"].isin(node_ids)
                        | edges_gdf.reset_index()["v"].isin(border_node_ids)
                    )
                )
            ]
        else:
            edges_gdf = edges_gdf[
                (
                    (
                        edges_gdf.reset_index()["u"].isin(node_ids)
                        & edges_gdf.reset_index()["v"].isin(node_ids)
                    )
                    | (
                        edges_gdf.reset_index()["u"].isin(border_node_ids)
                        | edges_gdf.reset_index()["v"].isin(node_ids)
                    )
                )
            ]

    border_node_ids = list(set(border_node_ids) - set(node_ids))

    if crop_graph:
        G = crop_graph_by_iso_nodes(
            G=None,
            node_ids=node_ids,
            border_node_ids=border_node_ids,
            min_edge_length=min_edge_length,
            undirected=undirected,
            outbound=outbound,
            nodes_gdf=nodes_gdf,
            edges_gdf=edges_gdf,
            graph_attrs=G.graph,
        )
        return G
    else:
        G = ox.graph_from_gdfs(
            gdf_nodes=nodes_gdf, gdf_edges=edges_gdf, graph_attrs=G.graph
        )
        return G, node_ids, border_node_ids
