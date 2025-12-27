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

def default_distance_matrix(poi, distance_steps, service_quality_column="service_quality"):
    """
    Create a diagonal-like distance vs service_quality DataFrame and return unique levels.

    Parameters
    ----------
    service_quality : list
        Ordered list of service quality levels (best to worst), e.g., ['I','II','III'].
    distance_steps : list
        Ordered list of distance steps (best to worst), e.g., [250, 500, 1000].

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with service_quality as rows, distance_steps as columns,
        and letters ('A','B','C',...) forming a diagonal pattern.
    accessibility_values : list
        Sorted list of unique letters used in the DataFrame.
    """
    service_quality = np.unique(poi[service_quality_column].dropna())
    # Create letters for values
    letters = list(string.ascii_uppercase)

    # Initialize empty DataFrame
    df = pd.DataFrame(index=service_quality, columns=distance_steps)

    # Fill in diagonal pattern
    for i, sq in enumerate(service_quality):
        for j, dist in enumerate(distance_steps):
            idx = min(i + j, len(letters) - 1)  # prevent out-of-range
            df.at[sq, dist] = letters[idx]

    # Extract unique letters used, sorted in order
    accessibility_values = sorted(df.stack().unique(), key=lambda x: letters.index(x))

    df[service_quality_column] = df.index 
    df = df.reset_index(drop=True)

    return df, accessibility_values


def __distance_matrix_to_processing_order(distance_matrix, accessibility_values):
    if isinstance(distance_matrix, list):
        distances = [float(d) for d in distance_matrix]

        # Generate labels A, B, C... for each row
        if accessibility_values is None:
            labels = list(string.ascii_uppercase[: len(distances)])
        else:
            labels = list(accessibility_values)

        # Build DataFrame
        ls_process_order_df = pd.DataFrame(
            {
                "ls_id": range(1, len(distances) + 1),
                "distance": distances,
                "service_quality": 1,
                "ls": labels,
            }
        )
        ls_process_order_df["distance"] = ls_process_order_df["distance"].astype(float)
        
        ls_process_order_df = ls_process_order_df.sort_values(
            ["ls_id", "distance"], ascending=False
        )
        return ls_process_order_df

    if "service_quality" not in distance_matrix:
        raise Exception("Column service_quality should always exist in distance_matrix")

    if accessibility_values is None:
        accessibility_values = distance_matrix.drop(columns=["service_quality"]).to_numpy()
        accessibility_values = list(np.unique(accessibility_values))

    # Melt the dataframe to long format for easier processing
    melted = distance_matrix.melt(
        id_vars="service_quality", var_name="distance", value_name="ls"
    )
    melted["distance"] = melted["distance"].astype(float)

    # For each service_quality and value, find the max distance
    ls_process_order_df = (
        melted.groupby(["service_quality", "ls"])["distance"].max().reset_index()
    )
    ls_process_order_df["ls_id"] = ls_process_order_df["ls"].replace(
        {accessibility_values[i]: str(i) for i in range(len(accessibility_values))}
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


def __compute_isochrones(G, points, ls_process_order_df, service_quality_col=None, verbose:bool=True):
    H = G.copy()
    if service_quality_col is None:
        points["__service_quality"] = 1
        service_quality_col = "__service_quality"

    for quality, distance, accessibility in tqdm(
        ls_process_order_df[["service_quality", "distance", "ls"]].itertuples(
            index=False, name=None
        ),
        total=len(ls_process_order_df),
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

    # ---- Final cleaning: remove any 'nan'/'None' strings and unify missing values ----

    def _clean(series):
        series = series.replace({"nan": None, "None": None})
        # Convert np.nan/pd.NA to Python None
        series = series.where(series.notna(), None)
        return series.astype(object)

    edges_gdf["accessibility"] = _clean(edges_gdf["accessibility"])
    nodes_gdf["accessibility"] = _clean(nodes_gdf["accessibility"])

    # ---- Restore indices ----
    edges_gdf = edges_gdf.set_index(["u", "v", "key"])
    nodes_gdf = nodes_gdf.set_index("osmid")

    return edges_gdf, nodes_gdf


def __exact_isochrones(G, ls_process_order_df, min_edge_length):
    accessibility_values = list(ls_process_order_df["ls"].drop_duplicates())
    accessibility_values.reverse()

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    remaining_dist_cols = [
        c for c in nodes_gdf.columns
        if (c.startswith("remaining_dist_")
            and (c.removeprefix("remaining_dist_") in accessibility_values))
    ]

    accessibility_values = [
        ls for ls in accessibility_values if f"remaining_dist_{ls}" in remaining_dist_cols
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
        ls = accessibility_values[len(accessibility_values) - 1 - i]
        col = f"remaining_dist_{ls}"
        nodes_gdf.loc[nodes_gdf[col] > min_edge_length, "accessibility"] = ls

    nodes_gdf = nodes_gdf.drop(columns=remaining_dist_cols)

    max_priority_map = len(accessibility_values)
    priority_map = {val: str(i) for i, val in enumerate(accessibility_values)}
    priority_map_rev = {str(i): val for i, val in enumerate(accessibility_values)}
    priority_map_rev[str(max_priority_map + 1)] = None

    edges_gdf[f"last_accessibility_{accessibility_values[-1]}_u"] = None
    edges_gdf[f"last_accessibility_{accessibility_values[-1]}_v"] = None

    for i in range(len(accessibility_values)):
        ls = accessibility_values[len(accessibility_values) - 1 - i]
        remaining_ls = accessibility_values[0 : (len(accessibility_values) - i - 1)]
        # remaining_ls = [ls for ls in accessibility_values[0:(len(accessibility_values)-i-1)]
        #                 if ls in remaining_dist_cols]
        col = f"remaining_dist_{ls}"
        edges_gdf.loc[
            (edges_gdf[col + "_u"] + edges_gdf[col + "_v"])
            > (edges_gdf["length"] - min_edge_length),
            "accessibility",
        ] = ls
        if i < (len(accessibility_values) - 1):
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
                [f"last_accessibility_{j}_u" for j in remaining_ls],
            ] = ls
            edges_gdf.loc[
                mask_v & (edges_gdf[col + "_v"] > min_edge_length),
                [f"last_accessibility_{j}_v" for j in remaining_ls],
            ] = ls

    dist_u = np.zeros(len(edges_gdf))
    dist_v = np.zeros(len(edges_gdf))
    for ls in accessibility_values:
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

    def _clean(series):
        series = series.replace({"nan": None, "None": None})
        # Convert np.nan/pd.NA to Python None
        series = series.where(series.notna(), None)
        return series.astype(object)

    nodes_gdf["accessibility"] = _clean(nodes_gdf["accessibility"])
    edges_gdf["accessibility"] = _clean(edges_gdf["accessibility"])

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
    if service_quality_col is None:
        points['service_quality_col'] = 1 

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

    ls_process_order_df = __distance_matrix_to_processing_order(
        distance_matrix=distance_matrix, accessibility_values=accessibility_values
    )
    H = __compute_isochrones(
        H,
        points=points,
        ls_process_order_df=ls_process_order_df,
        service_quality_col=service_quality_col,
        verbose=verbose
    )

    H = __exact_isochrones(
        H, ls_process_order_df=ls_process_order_df, min_edge_length=min_edge_length
    )
    if return_points:
        return H, points
    
    return H



def buffers(
    service_geoms, distance_matrix, accessibility_values, service_quality_col, verbose:bool=True
):
    if service_geoms.crs.is_geographic:
        service_geoms = service_geoms.to_crs(service_geoms.estimate_utm_crs())

    if service_quality_col is None:
        service_geoms["__service_quality"] = 1
        service_quality_col = "__service_quality"

    ls_process_order_df = __distance_matrix_to_processing_order(
        distance_matrix=distance_matrix, accessibility_values=accessibility_values
    )

    accessibility_values_list = list(ls_process_order_df["ls"].drop_duplicates())
    accessibility_values_list.reverse()
    buffers = {
        ls:[] for ls in accessibility_values_list
    }
    for quality, distance, accessibility in tqdm(
        ls_process_order_df[["service_quality", "distance", "ls"]].itertuples(
            index=False, name=None
        ),
        total=len(ls_process_order_df),
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
    return result.sort_values("accessibility_int")

