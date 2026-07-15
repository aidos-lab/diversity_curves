"""
Graph coarsening via Andreas Loukas's `graph_coarsening` library
(spectral + cut guarantees, JMLR 2019): https://github.com/loukasa/graph-coarsening

No torch dependency -- built on pygsp, numpy, and scipy.

Install (requires internet access; not on PyPI under a simple name, install
from source):
    pip install pygsp numpy scipy matplotlib sortedcontainers networkx
    pip install git+https://github.com/loukasa/graph-coarsening.git

This wraps the library's `coarsen()` function so that:
  - input:  a networkx.Graph
  - output: a networkx.Graph with EXACTLY k nodes

The library's own parameter is a reduction *ratio* r (fraction of nodes to
remove), not a target count k directly. Since integer rounding means a given
r doesn't always land exactly on the requested k, this wrapper computes r
from k and then does a small local search over r to find a value that
produces exactly k nodes (the underlying algorithm is deterministic given a
fixed method and r, so this search is reproducible).
"""

import numpy as np
import scipy as sp
import networkx as nx
import pygsp as gsp

# `graph_coarsening` uses numpy/scipy aliases (np.bool, np.int, np.float,
# sp.newaxis) that were removed in modern numpy/scipy. Restore them so the
# library keeps working.
for _name, _alias in (("bool", np.bool_), ("int", int), ("float", float)):
    if not hasattr(np, _name):
        setattr(np, _name, _alias)
if not hasattr(sp, "newaxis"):
    sp.newaxis = np.newaxis

from graph_coarsening.coarsening_utils import coarsen


def loukas_pool(G: nx.Graph, k: int, method: str = "variation_neighborhood",
                weight: str = "weight", max_search_steps: int = 40):
    """
    Coarsen graph G to exactly k nodes using Loukas's spectral coarsening.

    Parameters
    ----------
    G : networkx.Graph
        Input graph (converted to undirected/simple if needed).
    k : int
        Target number of nodes in the pooled graph.
    method : str
        One of the candidate-set strategies from the library, e.g.
        'variation_neighborhood', 'variation_edges', 'heavy_edge',
        'algebraic_JC', 'affinity_GS', 'kron'.
    weight : str
        Edge attribute to use as weight, if present.
    max_search_steps : int
        Cap on how many candidate ratios r to try while searching for the
        one that yields exactly k nodes.

    Returns
    -------
    G_pooled : networkx.Graph
        Coarsened graph with exactly k nodes (0..k-1), edge attribute
        'weight', and node attribute 'members' (original node labels folded
        into each supernode).
    C : scipy.sparse matrix
        The coarsening/assignment matrix returned by the library (rows =
        supernodes, cols = original nodes).
    """
    if not isinstance(G, nx.Graph):
        raise TypeError(f"G must be a networkx.Graph. Got {type(G)}")
    if G.is_directed() or G.is_multigraph():
        G = nx.Graph(G)

    n = G.number_of_nodes()
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed number of nodes in G ({n})")
    if k == n:
        return G.copy(), None

    nodes = list(G.nodes())
    idx_of = {node: i for i, node in enumerate(nodes)}

    has_weight = any(weight in d for _, _, d in G.edges(data=True))
    W = nx.to_numpy_array(G, nodelist=nodes, weight=weight if has_weight else None)
    G_gsp = gsp.graphs.Graph(W)

    def run_for_r(r):
        C, Gc, Call, Gall = coarsen(G_gsp, K=k, r=r, method=method)
        return C, Gc

    # r is the fraction of nodes to REMOVE; target size ~ n * (1 - r)
    r_target = 1.0 - k / n

    best = None
    last_exc = None
    # Small local search over r to hit the exact target size k, since
    # integer rounding inside the library means r_target doesn't always
    # land exactly on k.
    candidates = [r_target] + [
        r_target + delta
        for step in range(1, max_search_steps // 2 + 1)
        for delta in (step * 0.5 / n, -step * 0.5 / n)
    ]
    for r in candidates:
        r = min(max(r, 0.0), 1.0 - 1.0 / n)
        try:
            C, Gc = run_for_r(r)
        except Exception as exc:
            last_exc = exc
            continue
        if Gc.N == k:
            best = (C, Gc)
            break
        if best is None:
            best = (C, Gc)  # fallback: closest attempt so far

    if best is None:
        raise RuntimeError(
            f"All {len(candidates)} candidate ratios r failed for method="
            f"{method!r} (k={k}, n={n}). Last error: {last_exc!r}"
        ) from last_exc

    C, Gc = best
    if Gc.N != k:
        raise RuntimeError(
            f"Could not find a ratio r landing on exactly k={k} nodes "
            f"after {len(candidates)} tries (closest achieved: {Gc.N}). "
            f"Try a different `method`, or adjust max_search_steps."
        )

    # --- Build the pooled networkx graph from Gc's weighted adjacency ---
    W_pooled = Gc.W.toarray() if hasattr(Gc.W, "toarray") else np.asarray(Gc.W)
    G_pooled = nx.Graph()
    G_pooled.add_nodes_from(range(k))
    for i in range(k):
        for j in range(i + 1, k):
            w = W_pooled[i, j]
            if w > 0:
                G_pooled.add_edge(i, j, weight=float(w))

    # --- Recover cluster membership from C (supernode x original-node) ---
    C_dense = C.toarray() if hasattr(C, "toarray") else np.asarray(C)
    hard_assign = np.argmax(np.abs(C_dense), axis=0)  # for each orig node, its supernode
    members = {c: [] for c in range(k)}
    for i, node in enumerate(nodes):
        members[hard_assign[i]].append(node)
    for c in range(k):
        G_pooled.nodes[c]["members"] = members[c]
        G_pooled.nodes[c]["size"] = len(members[c])

    return G_pooled, C