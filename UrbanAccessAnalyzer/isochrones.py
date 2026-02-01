import geopandas as gpd
import shapely
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
from . import graph_processing
import string
from itertools import repeat
from tqdm import tqdm
import warnings

"""
TODO: There is still a bug in the LoS graph and sometimes the values seem 
to not reach long enaough maybe. 
Or maybe it is right and the distance_matrix is the reason. This happened with the green areas.
TODO: Add source node_id to the accessibility info. 
"""

def __format_quality(v):
    # Missing values
    if pd.isna(v):
        return None

    # Numeric (int, float, numeric string)
    try:
        return f"{round(float(v), 3):.3f}"
    except (TypeError, ValueError):
        # Non-numeric strings like "bad_value"
        return str(v)


def default_distance_matrix(
    poi,
    distance_steps,
    service_quality_column="service_quality"
):
    """
    Create a diagonal-like distance vs service_quality DataFrame with
    accessibility values in the range [0, 1], rounded to 3 decimals.

    Always includes service_quality = 0 with accessibility = 0.
    """

    # Unique service quality levels
    service_quality = poi[service_quality_column].dropna().unique()

    # Always include 0
    service_quality = np.unique(np.append(service_quality, 0.0))

    # Sort best → worst
    service_quality = np.sort(service_quality)[::-1]

    n_sq = len(service_quality)
    n_dist = len(distance_steps)

    max_idx = n_sq + n_dist - 2

    df = pd.DataFrame(index=service_quality, columns=distance_steps, dtype=float)

    for i, sq in enumerate(service_quality):
        for j, dist in enumerate(distance_steps):

            if sq == 0:
                value = 0.0
            else:
                idx = i + j
                value = 1.0 - (idx / max_idx)
                value = max(value, 0.0)

            df.at[sq, dist] = round(value, 3)

    df = df[~np.isclose(df.index, 0)]
    # Unique accessibility values (best → worst)
    accessibility_values = sorted(
        df.stack().unique(), reverse=True
    )

    df[service_quality_column] = df.index
    df = df.reset_index(drop=True)

    return df, accessibility_values


def __distance_matrix_to_processing_order(distance_matrix, accessibility_values=None):
    if isinstance(distance_matrix, list):
        distances = [float(d) for d in distance_matrix]

        # Generate labels A, B, C... for each row
        if accessibility_values is None:
            labels = list(distance_matrix)
        else:
            labels = list(accessibility_values)

        # Build DataFrame
        process_order_df = pd.DataFrame(
            {
                "accessibility_int": range(1, len(distances) + 1),
                "distance": distances,
                "service_quality": 1,
                "accessibility_value": labels,
            }
        )
        process_order_df["distance"] = process_order_df["distance"].astype(float)
        
        process_order_df = process_order_df.sort_values(
            ["accessibility_int", "distance"], ascending=False
        )
        return process_order_df

    if "service_quality" not in distance_matrix:
        raise Exception("Column service_quality should always exist in distance_matrix")

    if accessibility_values is None:
        accessibility_values = distance_matrix.drop(columns=["service_quality"]).to_numpy()
        accessibility_values = np.unique(accessibility_values)[::-1].tolist()

    accessibility_values = [__format_quality(v) for v in accessibility_values]

    distance_matrix["service_quality"] = distance_matrix["service_quality"].map(
        __format_quality
    )
    # Melt the dataframe to long format for easier processing
    melted = distance_matrix.melt(
        id_vars="service_quality", var_name="distance", value_name="accessibility_value"
    )
    melted["distance"] = melted["distance"].astype(float)
    melted["accessibility_value"] = melted["accessibility_value"].map(
        __format_quality
    )


    # For each service_quality and value, find the max distance
    process_order_df = (
        melted.groupby(["service_quality", "accessibility_value"])["distance"].max().reset_index()
    )
    process_order_df["accessibility_int"] = process_order_df["accessibility_value"].replace(
        {accessibility_values[i]: str(i) for i in range(len(accessibility_values))}
    )
    process_order_df["accessibility_int"] = process_order_df["accessibility_int"].astype(int)
    process_order_df = (
        process_order_df.groupby(["accessibility_int", "distance"])
        .agg({"service_quality": list, "accessibility_value": "first"})
        .reset_index()
    )
    process_order_df = process_order_df.sort_values(
        ["accessibility_int", "distance"], ascending=False
    )
    return process_order_df


def __compute_isochrones(G, points, process_order_df, service_quality_col=None, verbose:bool=True):
    H = G.copy()
    if service_quality_col is None:
        points["__service_quality"] = 1
        service_quality_col = "__service_quality"

    points[service_quality_col] = points[service_quality_col].map(
        __format_quality
    )

    for quality, distance, accessibility in tqdm(
        process_order_df[["service_quality", "distance", "accessibility_value"]].itertuples(
            index=False, name=None
        ),
        total=len(process_order_df),
        disable=not verbose,
    ):
        if isinstance(quality, int):
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
                H, f"remaining_dist_{accessibility}"
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
                H, remaining_dist, f"remaining_dist_{accessibility}"
            )
            nx.set_node_attributes(
                H,
                dict(zip(remaining_dist.keys(), repeat(accessibility))),
                "accessibility",
            )

    return H


def __set_edge_accessibility(
    nodes_gdf, edges_gdf, priority_map, max_priority_map, priority_map_rev
):
    """
    Normalize, combine, and restore accessibility values on nodes and edges.
    Ensures no 'nan', 'None' or np.nan leaks into the final output.
    """

    # ---- Ensure maps handle all possible NaN representations ----
    priority_map = priority_map.copy()
    priority_map_rev = priority_map_rev.copy()

    # Inputs may contain these → treat all as missing
    bad_keys = [np.nan, "nan", "None", None]

    for k in bad_keys:
        priority_map[k] = str(max_priority_map + 1)

    # Reverse map: fallback numeric → None
    priority_map_rev[str(max_priority_map + 1)] = None
    for k in ["nan", "None", np.nan, None]:
        priority_map_rev[str(k) if not isinstance(k, float) else "nan"] = None

    # ---- Work on copies ----
    nodes_gdf = nodes_gdf.reset_index().copy()

    # ---- Normalize edge LOS if column exists ----
    if "accessibility" in edges_gdf.columns:
        edges_gdf["accessibility"] = (
            edges_gdf["accessibility"]
            .astype(str)
            .replace(priority_map)
        )

        edges_gdf["accessibility"] = edges_gdf["accessibility"].fillna(
            str(max_priority_map + 1)
        )

        edges_gdf["accessibility"] = edges_gdf["accessibility"].astype(int)

    else:
        edges_gdf["accessibility"] = max_priority_map + 1

    # ---- Normalize node LOS ----
    nodes_gdf["accessibility"] = (
        nodes_gdf["accessibility"]
        .astype(str)
        .replace(priority_map)
    )

    nodes_gdf["accessibility"] = nodes_gdf["accessibility"].fillna(
        str(max_priority_map + 1)
    )

    nodes_gdf["accessibility"] = nodes_gdf["accessibility"].astype(int)

    # ---- Merge node LOS into edges ----
    edges_gdf = edges_gdf.reset_index()

    edges_gdf = edges_gdf.merge(
        nodes_gdf[["osmid", "accessibility"]]
        .rename(columns={"osmid": "u", "accessibility": "accessibility_u"}),
        on="u",
        how="left",
    )

    edges_gdf = edges_gdf.merge(
        nodes_gdf[["osmid", "accessibility"]]
        .rename(columns={"osmid": "v", "accessibility": "accessibility_v"}),
        on="v",
        how="left",
    )

    # ---- Compute edge LOS as minimum ----
    edges_gdf["accessibility"] = (
        edges_gdf[["accessibility_u", "accessibility_v", "accessibility"]]
        .min(axis=1)
        .astype(int)
    )

    edges_gdf = edges_gdf.drop(columns=["accessibility_u", "accessibility_v"])

    # ---- Map integer priorities back to original values ----
    edges_gdf["accessibility"] = (
        edges_gdf["accessibility"]
        .astype(str)
        .replace(priority_map_rev)
    )

    nodes_gdf["accessibility"] = (
        nodes_gdf["accessibility"]
        .astype(str)
        .replace(priority_map_rev)
    )

    # ---- Restore indices ----
    edges_gdf = edges_gdf.set_index(["u", "v", "key"])
    nodes_gdf = nodes_gdf.set_index("osmid")

    return edges_gdf, nodes_gdf


def __exact_isochrones(G, process_order_df, min_edge_length):
    accessibility_values = list(process_order_df["accessibility_value"].drop_duplicates())
    accessibility_values.reverse()

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    remaining_dist_cols = [
        c for c in nodes_gdf.columns
        if (c.startswith("remaining_dist_")
            and (c.removeprefix("remaining_dist_") in accessibility_values))
    ]

    accessibility_values = [
        a for a in accessibility_values if f"remaining_dist_{a}" in remaining_dist_cols
    ]


    if len(accessibility_values) == 0:
        warnings.warn(
            "No nodes have attribute remaining_dist. Probably no isochrones have been generated.",
            UserWarning
        )
        return G
    
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

    nodes_gdf["accessibility"] = None
    for i in range(len(accessibility_values)):
        a = accessibility_values[len(accessibility_values) - 1 - i]
        col = f"remaining_dist_{a}"
        nodes_gdf.loc[nodes_gdf[col] > min_edge_length, "accessibility"] = a

    nodes_gdf = nodes_gdf.drop(columns=remaining_dist_cols)

    max_priority_map = len(accessibility_values)
    priority_map = {val: str(i) for i, val in enumerate(accessibility_values)}
    priority_map_rev = {str(i): val for i, val in enumerate(accessibility_values)}
    priority_map_rev[str(max_priority_map + 1)] = None

    edges_gdf[f"last_accessibility_{accessibility_values[-1]}_u"] = None
    edges_gdf[f"last_accessibility_{accessibility_values[-1]}_v"] = None

    for i in range(len(accessibility_values)):
        a = accessibility_values[len(accessibility_values) - 1 - i]
        remaining_acc_value = accessibility_values[0 : (len(accessibility_values) - i - 1)]
        # remaining_ls = [ls for ls in accessibility_values[0:(len(accessibility_values)-i-1)]
        #                 if ls in remaining_dist_cols]
        col = f"remaining_dist_{a}"
        edges_gdf.loc[
            (edges_gdf[col + "_u"] + edges_gdf[col + "_v"])
            > (edges_gdf["length"] - min_edge_length),
            "accessibility",
        ] = a
        if i < (len(accessibility_values) - 1):
            mask_u = (
                edges_gdf[[f"remaining_dist_{j}_u" for j in remaining_acc_value]].max(axis=1)
                < edges_gdf[col + "_u"]
            )
            mask_v = (
                edges_gdf[[f"remaining_dist_{j}_u" for j in remaining_acc_value]].max(axis=1)
                < edges_gdf[col + "_v"]
            )
            edges_gdf.loc[
                mask_u & (edges_gdf[col + "_u"] > min_edge_length),
                [f"last_accessibility_{j}_u" for j in remaining_acc_value],
            ] = a
            edges_gdf.loc[
                mask_v & (edges_gdf[col + "_v"] > min_edge_length),
                [f"last_accessibility_{j}_v" for j in remaining_acc_value],
            ] = a

    dist_u = np.zeros(len(edges_gdf))
    dist_v = np.zeros(len(edges_gdf))
    for a in accessibility_values:
        col = f"remaining_dist_{a}"
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
            col.replace("remaining_dist_", "last_accessibility_")
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
    accessibility_values_u_v = [
        c.removeprefix("remaining_dist_") for c in remaining_dist_cols_u_v
    ]
    sources = np.array(accessibility_values_u_v)[col_idx]

    projected_dist = values[row_idx, col_idx]

    edges_border_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=edges_gdf.crs)
    if len(edges_border_gdf) > 0:
        edges_border_gdf["accessibility"] = sources
        edges_border_gdf["accessibility"] = edges_border_gdf.apply( 
            lambda row: row[f'last_accessibility_{row["accessibility"]}'], axis=1
        )
        edges_border_gdf = edges_border_gdf.drop(
            columns=[
                col.replace("remaining_dist_", "last_accessibility_")
                for col in remaining_dist_cols_u_v
            ]
        )
        edges_border_gdf["projected_dist"] = projected_dist

        edges_border_gdf["point"] = edges_border_gdf.interpolate(
            edges_border_gdf["projected_dist"]
        )
        edges_border_gdf["length"] = edges_border_gdf["projected_dist"]

        min_id = nodes_gdf["osmid"].max()
        min_id = max(min_id,edges_gdf['u'].max())
        min_id = max(min_id,edges_gdf['v'].max())
        min_id += 1 
        new_border_node_ids = list(min_id + np.arange(0, len(edges_border_gdf)))
        edges_border_gdf["new_node_id"] = new_border_node_ids

        edges_border_gdf["u"] = edges_border_gdf["u"].astype(int)
        edges_border_gdf["v"] = edges_border_gdf["v"].astype(int)
        edges_border_gdf["key"] = edges_border_gdf["key"].astype(int)
        edges_border_gdf = edges_border_gdf.set_index(["u", "v", "key"])

    nodes_gdf = nodes_gdf.set_index("osmid")
    edges_gdf = edges_gdf.set_index(["u", "v", "key"])

    if len(edges_border_gdf) > 0:
        nodes_gdf, edges_gdf = graph_processing.__split_at_edges(
            nodes_gdf, edges_gdf, edges_border_gdf
        )

    edges_gdf, nodes_gdf = __set_edge_accessibility(
        nodes_gdf,
        edges_gdf.drop(columns="accessibility"),
        priority_map,
        max_priority_map,
        priority_map_rev,
    )
    iso_nodes_gdf = nodes_gdf.loc[orig_node_ids]
    edges_gdf, _ = __set_edge_accessibility(
        iso_nodes_gdf, edges_gdf, priority_map, max_priority_map, priority_map_rev
    )

    H = ox.graph_from_gdfs(nodes_gdf, edges_gdf, graph_attrs=G.graph)
    return H


def graph(
    G,
    points,
    distance_matrix,
    service_quality_col=None,
    accessibility_values=None,
    min_edge_length=0,
    max_dist=None,
    verbose:bool=True
):
    points = points.copy()
    if service_quality_col is None:
        points['_service_quality_col'] = 1 
        service_quality_col = '_service_quality_col'
        
    return_points = False

    if "osmid" not in points.columns:
        H, osmids = graph_processing.add_points_to_graph(
            points,
            G,
            max_dist=max_dist, # Maximum distance from point to graph edge to project the point
            min_edge_length=min_edge_length # Minimum edge length after adding the new nodes
        )
        points['osmid'] = osmids # Add the ids of the nodes in the graph to points
        return_points = True
    else:
        H = G.copy()

    if all(points['osmid'].isna()):  # works if points is a pandas DataFrame
        warnings.warn("Points are too far away from edges. No isochrones returned.", UserWarning)
        return G

    if isinstance(distance_matrix,pd.DataFrame) and service_quality_col is not None:
        if service_quality_col in distance_matrix.columns and 'service_quality' not in distance_matrix.columns:
            distance_matrix = distance_matrix.rename(columns={service_quality_col:'service_quality'})

    process_order_df = __distance_matrix_to_processing_order(
        distance_matrix=distance_matrix, accessibility_values=accessibility_values
    )
    H = __compute_isochrones(
        H,
        points=points,
        process_order_df=process_order_df,
        service_quality_col=service_quality_col,
        verbose=verbose
    )

    H = __exact_isochrones(
        H, process_order_df=process_order_df, min_edge_length=min_edge_length
    )
    
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(H)
    try:
        nodes_gdf["accessibility"] = nodes_gdf["accessibility"].astype(float)
    except:
        None

    try:
        edges_gdf["accessibility"] = edges_gdf["accessibility"].astype(float)
    except:
        None 
    
    H = ox.graph_from_gdfs(nodes_gdf, edges_gdf, graph_attrs=H.graph)
    if return_points:
        return H, points
    
    return H



def buffers(
    service_geoms, distance_matrix, accessibility_values, service_quality_col, verbose:bool=True
):
    service_geoms = service_geoms.copy()
    if service_geoms.crs.is_geographic:
        service_geoms = service_geoms.to_crs(service_geoms.estimate_utm_crs())

    if service_quality_col is None:
        service_geoms["__service_quality"] = 1
        service_quality_col = "__service_quality"

    service_geoms[service_quality_col] = service_geoms[service_quality_col].map(
        __format_quality
    )

    if isinstance(distance_matrix,pd.DataFrame) and service_quality_col is not None:
        if service_quality_col in distance_matrix.columns and 'service_quality' not in distance_matrix.columns:
            distance_matrix = distance_matrix.rename(columns={service_quality_col:'service_quality'})

    process_order_df = __distance_matrix_to_processing_order(
        distance_matrix=distance_matrix, accessibility_values=accessibility_values
    )

    accessibility_values_list = list(process_order_df["accessibility_value"].drop_duplicates())
    accessibility_values_list.reverse()
    buffers = {
        a:[] for a in accessibility_values_list
    }
    for quality, distance, accessibility in tqdm(
        process_order_df[["service_quality", "distance", "accessibility_value"]].itertuples(
            index=False, name=None
        ),
        total=len(process_order_df),
        disable=not verbose,
    ):
        if isinstance(quality, int):
            quality = [quality]

        selected_points = service_geoms[service_geoms[service_quality_col].isin(quality)]
        selected_points = selected_points.geometry.union_all().buffer(distance,resolution=4)
        buffers[accessibility].append(selected_points)

    rows = []
    total_geometry = None
    i=0
    for accessibility in accessibility_values_list:
        i+=1
        geom = shapely.unary_union(buffers[accessibility])
        if total_geometry is None:
            total_geometry = geom
            row_geom = geom
        else:
            row_geom = shapely.difference(geom,total_geometry)
            total_geometry = shapely.unary_union([total_geometry,geom])

        rows.append({'accessibility': accessibility, 'accessibility_int': i, 'geometry': row_geom})


    result = gpd.GeoDataFrame(rows, crs=service_geoms.crs)
    result = result.sort_values("accessibility_int")
    
    result['accessibility'] = result["accessibility"].map(
        __format_quality
    )
    return result

