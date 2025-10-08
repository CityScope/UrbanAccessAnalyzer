import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
from . import graph_processing
import string
from itertools import repeat
from tqdm import tqdm


def __distance_matrix_to_processing_order(distance_matrix, level_of_services):
    if isinstance(distance_matrix, list):
        distances = distance_matrix
        # Generate labels A, B, C... for each row
        if level_of_services is None:
            labels = list(string.ascii_uppercase[: len(distances)])
        else:
            labels = list(level_of_services)

        # Build DataFrame
        ls_process_order_df = pd.DataFrame(
            {
                "ls_id": range(1, len(distances) + 1),
                "distance": distances,
                "service_quality": 1,
                "ls": labels,
            }
        )
        ls_process_order_df = ls_process_order_df.sort_values(
            ["ls_id", "distance"], ascending=False
        )
        return ls_process_order_df

    if "service_quality" not in distance_matrix:
        raise Exception("Column service_quality should always exist in distance_matrix")

    if level_of_services is None:
        level_of_services = distance_matrix.drop(columns=["service_quality"]).to_numpy()
        level_of_services = list(np.unique(level_of_services))

    # Melt the dataframe to long format for easier processing
    melted = distance_matrix.melt(
        id_vars="service_quality", var_name="distance", value_name="ls"
    )

    # For each service_quality and value, find the max distance
    ls_process_order_df = (
        melted.groupby(["service_quality", "ls"])["distance"].max().reset_index()
    )
    ls_process_order_df["ls_id"] = ls_process_order_df["ls"].replace(
        {level_of_services[i]: str(i) for i in range(len(level_of_services))}
    )
    ls_process_order_df["ls_id"] = ls_process_order_df["ls_id"].astype(int)
    ls_process_order_df = (
        ls_process_order_df.groupby(["ls_id", "distance"])
        .agg({"service_quality": list, "ls": "first"})
        .reset_index()
    )
    ls_process_order_df = ls_process_order_df.sort_values(
        ["ls_id", "distance"], ascending=False
    )
    return ls_process_order_df


def __compute_isochrones(G, points, ls_process_order_df, service_quality_col=None):
    H = G.copy()
    if service_quality_col is None:
        points["__service_quality"] = 1
        service_quality_col = "__service_quality"

    for quality, distance, level_of_service in tqdm(
        ls_process_order_df[["service_quality", "distance", "ls"]].itertuples(
            index=False, name=None
        ),
        total=len(ls_process_order_df),
    ):
        if quality is int:
            quality = [quality]
            
        node_ids = list(
            points.loc[points[service_quality_col].isin(quality), "osmid"]
            .dropna()
            .astype(int)
        )
        if len(node_ids) > 0:
            _, _, remaining_dist = graph_processing.__multi_ego_graph(
                H,
                node_ids,
                distance,
                weight="length",
                undirected=True,
            )
            existing_dist = nx.get_node_attributes(
                H, f"remaining_dist_{level_of_service}"
            )
            if len(existing_dist) > 0:
                existing_dist = np.array(
                    list(existing_dist.items())
                )  # [[key, value], ...]
                remaining_dist = np.array(list(remaining_dist.items()))
                # Sort both arrays by key
                existing_dist = existing_dist[np.argsort(existing_dist[:, 0])]
                remaining_dist = remaining_dist[np.argsort(remaining_dist[:, 0])]

                existing_dist = existing_dist[
                    np.isin(existing_dist[:, 0], remaining_dist[:, 0])
                ]

                mask = np.isin(remaining_dist[:, 0], existing_dist[:, 0])
                # Get the indices of the masked rows
                idx = np.where(mask)[0]
                # Compare only the masked rows with existing_dist[:,1]
                to_nan = remaining_dist[idx, 1] <= existing_dist[:, 1]
                # Assign np.nan directly using the original array
                remaining_dist[idx[to_nan], 1] = np.nan

                remaining_dist = remaining_dist[~np.isnan(remaining_dist[:, 1])]
                remaining_dist = dict(remaining_dist.tolist())

            nx.set_node_attributes(
                H, remaining_dist, f"remaining_dist_{level_of_service}"
            )
            nx.set_node_attributes(
                H,
                dict(zip(remaining_dist.keys(), repeat(level_of_service))),
                "level_of_service",
            )

    return H


def __set_edge_level_of_service(
    nodes_gdf, edges_gdf, priority_map, max_priority_map, priority_map_rev
):
    nodes_gdf = nodes_gdf.reset_index().copy()
    if "level_of_service" in edges_gdf.columns:
        edges_gdf["level_of_service"] = (
            edges_gdf["level_of_service"].astype(str).replace(priority_map)
        )
        edges_gdf["level_of_service"] = edges_gdf["level_of_service"].fillna(
            str(max_priority_map + 1)
        )
        edges_gdf["level_of_service"] = edges_gdf["level_of_service"].astype(int)
    else:
        edges_gdf["level_of_service"] = max_priority_map + 1

    nodes_gdf["level_of_service"] = nodes_gdf["level_of_service"].replace(priority_map)
    nodes_gdf["level_of_service"] = nodes_gdf["level_of_service"].fillna(
        str(max_priority_map + 1)
    )
    nodes_gdf["level_of_service"] = nodes_gdf["level_of_service"].astype(int)
    edges_gdf = edges_gdf.reset_index()
    edges_gdf = edges_gdf.merge(
        nodes_gdf[["osmid", "level_of_service"]].rename(
            columns={"osmid": "u", "level_of_service": "level_of_service_u"}
        ),
        on="u",
        how="left",
    )
    edges_gdf = edges_gdf.merge(
        nodes_gdf[["osmid", "level_of_service"]].rename(
            columns={"osmid": "v", "level_of_service": "level_of_service_v"}
        ),
        on="v",
        how="left",
    )
    edges_gdf["level_of_service"] = (
        edges_gdf[["level_of_service_u", "level_of_service_v", "level_of_service"]]
        .min(axis=1)
        .astype(int)
    )

    edges_gdf = edges_gdf.drop(columns=["level_of_service_u", "level_of_service_v"])

    edges_gdf["level_of_service"] = (
        edges_gdf["level_of_service"].astype(str).replace(priority_map_rev)
    )
    edges_gdf = edges_gdf.set_index(["u", "v", "key"])

    nodes_gdf["level_of_service"] = (
        nodes_gdf["level_of_service"].astype(str).replace(priority_map_rev)
    )
    nodes_gdf = nodes_gdf.set_index(["osmid"])
    return edges_gdf, nodes_gdf


def __exact_isochrones(G, ls_process_order_df, min_edge_length):
    level_of_services = list(ls_process_order_df["ls"].drop_duplicates())
    level_of_services.reverse()

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    remaining_dist_cols = [
        c for c in nodes_gdf.columns if c.startswith("remaining_dist_")
    ]
    level_of_services = [
        ls for ls in level_of_services if f"remaining_dist_{ls}" in remaining_dist_cols
    ]
    nodes_gdf[remaining_dist_cols] = nodes_gdf[remaining_dist_cols].fillna(0)

    nodes_gdf = nodes_gdf.reset_index()
    orig_node_ids = list(nodes_gdf["osmid"])
    edges_gdf = edges_gdf.reset_index()

    edges_gdf = edges_gdf.merge(
        nodes_gdf[["osmid"] + remaining_dist_cols],
        left_on="u",
        right_on="osmid",
        how="left",
        suffixes=["_v", "_u"],
    )
    edges_gdf = edges_gdf.drop(columns=["osmid_u"]).rename(columns={"osmid_v": "osmid"})
    edges_gdf = edges_gdf.merge(
        nodes_gdf[["osmid"] + remaining_dist_cols],
        left_on="v",
        right_on="osmid",
        how="left",
        suffixes=["_u", "_v"],
    )
    edges_gdf = edges_gdf.drop(columns=["osmid_v"]).rename(columns={"osmid_u": "osmid"})

    nodes_gdf["level_of_service"] = None
    for i in range(len(level_of_services)):
        ls = level_of_services[len(level_of_services) - 1 - i]
        col = f"remaining_dist_{ls}"
        nodes_gdf.loc[nodes_gdf[col] > min_edge_length, "level_of_service"] = ls

    nodes_gdf = nodes_gdf.drop(columns=remaining_dist_cols)

    max_priority_map = len(level_of_services)
    priority_map_rev = {str(i): val for i, val in enumerate(level_of_services)}
    priority_map = {val: str(i) for i, val in enumerate(level_of_services)}
    priority_map_rev[str(max_priority_map + 1)] = None
    priority_map["nan"] = str(max_priority_map + 1)
    priority_map["None"] = str(max_priority_map + 1)

    edges_gdf[f"last_level_of_service_{level_of_services[-1]}_u"] = None
    edges_gdf[f"last_level_of_service_{level_of_services[-1]}_v"] = None
    for i in range(len(level_of_services)):
        ls = level_of_services[len(level_of_services) - 1 - i]
        remaining_ls = level_of_services[0 : (len(level_of_services) - i - 1)]
        # remaining_ls = [ls for ls in level_of_services[0:(len(level_of_services)-i-1)]
        #                 if ls in remaining_dist_cols]
        col = f"remaining_dist_{ls}"
        edges_gdf.loc[
            (edges_gdf[col + "_u"] + edges_gdf[col + "_v"])
            > (edges_gdf["length"] - min_edge_length),
            "level_of_service",
        ] = ls
        if i < (len(level_of_services) - 1):
            mask_u = (
                edges_gdf[[f"remaining_dist_{j}_u" for j in remaining_ls]].max(axis=1)
                < edges_gdf[col + "_u"]
            )
            mask_v = (
                edges_gdf[[f"remaining_dist_{j}_u" for j in remaining_ls]].max(axis=1)
                < edges_gdf[col + "_v"]
            )
            edges_gdf.loc[
                mask_u & (edges_gdf[col + "_u"] > min_edge_length),
                [f"last_level_of_service_{j}_u" for j in remaining_ls],
            ] = ls
            edges_gdf.loc[
                mask_v & (edges_gdf[col + "_v"] > min_edge_length),
                [f"last_level_of_service_{j}_v" for j in remaining_ls],
            ] = ls

    dist_u = np.zeros(len(edges_gdf))
    dist_v = np.zeros(len(edges_gdf))
    for ls in level_of_services:
        col = f"remaining_dist_{ls}"
        new_dist_u = np.maximum(edges_gdf[col + "_u"].to_numpy(), dist_u)
        new_dist_v = np.maximum(edges_gdf[col + "_v"].to_numpy(), dist_v)

        edges_gdf.loc[
            (edges_gdf[col + "_u"]) > (edges_gdf["length"] - min_edge_length),
            col + "_u",
        ] = 0
        edges_gdf.loc[
            (edges_gdf[col + "_v"]) > (edges_gdf["length"] - min_edge_length),
            col + "_v",
        ] = 0

        edges_gdf.loc[edges_gdf[col + "_u"] < min_edge_length, col + "_u"] = 0
        edges_gdf.loc[edges_gdf[col + "_v"] < min_edge_length, col + "_v"] = 0

        edges_gdf.loc[
            edges_gdf[col + "_u"] > (edges_gdf["length"] - dist_v - min_edge_length),
            col + "_u",
        ] = 0
        edges_gdf.loc[
            edges_gdf[col + "_v"] > (edges_gdf["length"] - dist_u - min_edge_length),
            col + "_v",
        ] = 0

        edges_gdf.loc[
            edges_gdf[col + "_u"] < (dist_u + min_edge_length), col + "_u"
        ] = 0
        edges_gdf.loc[
            edges_gdf[col + "_v"] < (dist_v + min_edge_length), col + "_v"
        ] = 0

        edges_gdf.loc[
            (
                (edges_gdf[col + "_u"] + edges_gdf[col + "_v"])
                > (edges_gdf["length"] - min_edge_length)
            )
            & (edges_gdf[col + "_u"] > 0)
            & (edges_gdf[col + "_v"] > 0),
            [col + "_u", col + "_v"],
        ] = 0

        edges_gdf.loc[edges_gdf[col + "_v"] > 0, col + "_v"] = (
            edges_gdf.loc[edges_gdf[col + "_v"] > 0, "length"]
            - edges_gdf.loc[edges_gdf[col + "_v"] > 0, col + "_v"]
        )

        dist_u = new_dist_u
        dist_v = new_dist_v

    edges_border_gdf = edges_gdf.copy()

    remaining_dist_cols_u_v = [i + "_u" for i in remaining_dist_cols] + [
        i + "_v" for i in remaining_dist_cols
    ]
    edges_gdf = edges_gdf.drop(columns=remaining_dist_cols_u_v)
    edges_gdf = edges_gdf.drop(
        columns=[
            col.replace("remaining_dist_", "last_level_of_service_")
            for col in remaining_dist_cols_u_v
        ]
    )

    values = edges_border_gdf[remaining_dist_cols_u_v].to_numpy()
    mask = values > 0
    row_idx, col_idx = np.where(mask)

    rest_of_cols = [
        col for col in edges_border_gdf.columns if col not in remaining_dist_cols_u_v
    ]
    # Get corresponding row and column names
    rows = edges_border_gdf[rest_of_cols].iloc[row_idx].reset_index(drop=True)
    level_of_services_u_v = [
        c.removeprefix("remaining_dist_") for c in remaining_dist_cols_u_v
    ]
    sources = np.array(level_of_services_u_v)[col_idx]

    projected_dist = values[row_idx, col_idx]

    edges_border_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=edges_gdf.crs)
    edges_border_gdf["level_of_service"] = sources
    edges_border_gdf["level_of_service"] = edges_border_gdf.apply(
        lambda row: row[f'last_level_of_service_{row["level_of_service"]}'], axis=1
    )
    edges_border_gdf = edges_border_gdf.drop(
        columns=[
            col.replace("remaining_dist_", "last_level_of_service_")
            for col in remaining_dist_cols_u_v
        ]
    )
    edges_border_gdf["projected_dist"] = projected_dist

    edges_border_gdf["point"] = edges_border_gdf.interpolate(
        edges_border_gdf["projected_dist"]
    )
    edges_border_gdf["length"] = edges_border_gdf["projected_dist"]

    min_id = nodes_gdf["osmid"].max() + 1
    new_border_node_ids = list(min_id + np.arange(0, len(edges_border_gdf)))
    edges_border_gdf["new_node_id"] = new_border_node_ids

    edges_border_gdf["u"] = edges_border_gdf["u"].astype(int)
    edges_border_gdf["v"] = edges_border_gdf["v"].astype(int)
    edges_border_gdf["key"] = edges_border_gdf["key"].astype(int)
    edges_border_gdf = edges_border_gdf.set_index(["u", "v", "key"])

    nodes_gdf = nodes_gdf.set_index("osmid")
    edges_gdf = edges_gdf.set_index(["u", "v", "key"])

    nodes_gdf, edges_gdf = graph_processing.__split_at_edges(
        nodes_gdf, edges_gdf, edges_border_gdf
    )

    edges_gdf, nodes_gdf = __set_edge_level_of_service(
        nodes_gdf,
        edges_gdf.drop(columns="level_of_service"),
        priority_map,
        max_priority_map,
        priority_map_rev,
    )
    iso_nodes_gdf = nodes_gdf.loc[orig_node_ids]
    edges_gdf, _ = __set_edge_level_of_service(
        iso_nodes_gdf, edges_gdf, priority_map, max_priority_map, priority_map_rev
    )

    H = ox.graph_from_gdfs(nodes_gdf, edges_gdf, graph_attrs=G.graph)
    return H


def graph(
    G,
    points,
    distance_matrix,
    service_quality_col=None,
    level_of_services=None,
    min_edge_length=0,
):
    if service_quality_col is None:
        points['service_quality_col'] = 1 

    ls_process_order_df = __distance_matrix_to_processing_order(
        distance_matrix=distance_matrix, level_of_services=level_of_services
    )
    H = __compute_isochrones(
        G,
        points=points,
        ls_process_order_df=ls_process_order_df,
        service_quality_col=service_quality_col,
    )
    H = __exact_isochrones(
        H, ls_process_order_df=ls_process_order_df, min_edge_length=min_edge_length
    )
    return H
