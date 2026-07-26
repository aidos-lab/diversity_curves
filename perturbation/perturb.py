import random
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from magnipy.magnitude.distances import get_dist
from scipy.stats import wasserstein_distance

from diversity_curves.process import get_mag_graphs


#### Perturbations ####
def remove_random_non_bridge(G, perturb_degree):
    edges = list(G.edges())
    edges_to_remove = int(perturb_degree * len(edges) / 100)
    edges_to_remove = min(edges_to_remove, len(edges) - 2)

    for _ in range(edges_to_remove):
        edge_to_remove = random.choice(edges)
        G.remove_edge(*edge_to_remove)
        edges = list(G.edges())


def swap_edge(G, perturb_degree):
    edges = list(G.edges())
    edges_to_swap = int(perturb_degree * len(edges) / 100)

    for i in range(edges_to_swap):
        e_1, e_2 = random.sample(edges, 2)

        G.remove_edge(*e_1)
        G.remove_edge(*e_2)

        G.add_edge(e_1[0], e_2[0])
        G.add_edge(e_1[1], e_2[1])

        edges = list(G.edges())


def rewire_edge(G, perturb_degree):
    edges = list(G.edges())
    all_nodes = list(G.nodes())
    edges_to_rewire = int(perturb_degree * len(edges) / 100)

    for u, v in random.sample(edges, k=edges_to_rewire):
        G.remove_edge(u, v)
        available_nodes = set(all_nodes) - {u, v}
        new_v = random.choice(list(available_nodes))
        G.add_edge(u, new_v)


def add_edge(G: nx.Graph, perturb_degree):
    n = G.number_of_nodes()
    max_edges = float(n * (n - 1)) / 2
    edges_to_add = int(perturb_degree / 100.0 * max_edges)

    fully_connected_nodes = set()
    while edges_to_add > 0:
        nodes = list(G.nodes())
        available_nodes = set(nodes) - fully_connected_nodes

        if len(available_nodes) == 0:
            return

        u = random.choice(nodes)
        not_adj_nbhd = list(set(G.nodes()) - set(G.neighbors(u)) - {u})

        if len(not_adj_nbhd) == 0:
            fully_connected_nodes.add(u)
            continue

        v = random.choice(not_adj_nbhd)
        G.add_edge(u, v)
        edges_to_add -= 1


PERTURBATION_FUNCTIONS = {
    "add_edge": add_edge,
    "remove_edge": remove_random_non_bridge,
    "rewire_edge": rewire_edge,
    "swap_edge": swap_edge,
}


#### Embeddings ######

METRIC_NAME = "shortest_path_distance"



def dist_fn(G):
    return get_dist(G=G, X=None, X2=None, metric=METRIC_NAME, mode="structure")


def mag_curve(G_list, n_repeats=3):
    """Diversity curve embedding: magnitude tracked while coarsening each graph."""
    ts = [1]
    max_nodes = max(G.number_of_nodes() for G in G_list)

    # Only a sparse subset of cardinalities is evaluated for speed; this is filled
    # back in below via linear interpolation before graphs are compared.
    compute_at_cardinality = np.linspace(1, 20, num=10, dtype=int)
    curves, _ = get_mag_graphs(
        graphs=G_list,
        dist_fn=dist_fn,
        ts=ts,
        upsample_to=max_nodes,
        n_repeats=n_repeats,
        compute_at_cardinality=compute_at_cardinality,
        mode="structure",
        scores_method="random",
        upsample_method="nodes",
    )

    agg_curves = curves.reset_index()
    agg_curves = agg_curves.groupby(["graph", "index"]).mean()
    agg_curves = agg_curves.pivot_table(index="index", columns="graph", values=curves.columns[0])

    df = agg_curves.copy()
    size_nodes = len(agg_curves.index)
    if size_nodes < int(max_nodes):
        curr_card = list(agg_curves.index)[0]
        curr_card = curr_card if curr_card == 1 else 0
        for idx in list(agg_curves.index):
            if idx - curr_card == 1:
                continue
            else:
                new_rows = pd.DataFrame(
                    np.nan, index=range(curr_card + 1, idx), columns=agg_curves.columns
                )
                df = pd.concat([new_rows, df]).sort_index()
            curr_card = idx

    df.loc[1, :] = 1.0
    df.loc[1:, :] = df.loc[1:, :].interpolate(method="index")

    assert pd.isna(df).sum().sum() == 0, "Diversity curve has unfilled cardinalities"

    graph_ids = [
        group.sort_index().to_numpy().squeeze() for _, group in df.T.groupby(["graph"])
    ]
    return graph_ids


def degree_curve(G):
    return np.array([G.degree(n) for n in G.nodes])


def embed(G_list, embedding_type: str):
    if embedding_type == "mag":
        return mag_curve(G_list)
    else:  # "color": sorted degree sequence
        return np.array([degree_curve(G) for G in G_list])


#### Distances ######

def distance(G, H, dist_type: str):
    """Distance between a baseline and perturbed embedding of the same graph."""
    if dist_type == "integral":
        diff = np.abs(G - H)
        return np.trapz(y=diff, x=list(range(len(diff))))
    elif dist_type == "euclidean":
        return np.linalg.norm(G - H)
    elif dist_type == "wasserstein":
        G_counter = Counter(list(G))
        H_counter = Counter(list(H))
        return wasserstein_distance(
            u_values=np.array(list(G_counter.keys())),
            v_values=np.array(list(H_counter.keys())),
            u_weights=np.array(list(G_counter.values())),
            v_weights=np.array(list(H_counter.values())),
        )
    raise ValueError(f"Unknown distance type {dist_type}")
