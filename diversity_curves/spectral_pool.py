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
#import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering


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

    return G_pooled #, C

"""
A matched pair of operations built around the stochastic block model (SBM)
view of a graph:

  sbm_coarsen(G, k)  : fine graph  -> exactly k "block" nodes
  sbm_expand(G, m)   : block graph -> a larger (m-node) graph sampled from
                       the block model implied by G

sbm_coarsen groups nodes into k blocks (via spectral clustering) and stores
enough information on the coarse graph (block sizes as node attr 'size',
and inter/intra-block EDGE COUNTS as edge/self-loop weights) for sbm_expand
to later reconstruct a proper block-probability matrix and sample a new,
larger graph from it -- without needing the original fine graph.

sbm_expand can also be applied to any small graph on its own (not just the
output of sbm_coarsen): if 'size' node attributes or self-loop weights are
missing, sensible defaults are used (equal block sizes, a configurable
intra-block density).

Pure networkx / numpy / scikit-learn -- no torch.
"""

import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering


def _connect_components_by_affinity(G: nx.Graph, affinity: np.ndarray,
                                     index_of: dict):
    """
    Given a graph G (nodes possibly in several components) and an
    affinity matrix indexed via index_of[node], repeatedly add the
    highest-affinity edge that connects two different components until G
    is a single connected component. Added edges are tagged
    edge['synthetic'] = True.
    """
    if G.number_of_nodes() == 0:
        return G
    while not nx.is_connected(G):
        components = list(nx.connected_components(G))
        best = None  # (affinity_value, u, v)
        for ci in range(len(components)):
            for cj in range(ci + 1, len(components)):
                for u in components[ci]:
                    for v in components[cj]:
                        a = affinity[index_of[u], index_of[v]]
                        if best is None or a > best[0]:
                            best = (a, u, v)
        _, u, v = best
        G.add_edge(u, v, weight=max(float(best[0]), 1.0), synthetic=True)
    return G
 
 
def sbm_coarsen(G: nx.Graph, k: int, weight: str = "weight",
                 random_state: int = 42):
    """
    Coarsen G into exactly k block-nodes via spectral clustering. The
    result is guaranteed to be connected.
 
    Returns
    -------
    G_blocks : networkx.Graph
        k nodes (0..k-1), guaranteed connected. Node attrs: 'size' (# of
        original nodes in the block), 'members' (their labels),
        'intra_edges' (# of edges within the block -- no self-loops are
        used to store this). Edge attr 'weight' = # of edges between the
        two blocks (edges added only for connectivity are tagged
        'synthetic': True).
    assignment : dict
        {original_node: block_index}
    """
    if not isinstance(G, nx.Graph):
        raise TypeError(f"G must be a networkx.Graph. Got {type(G)}")
    if G.is_directed() or G.is_multigraph():
        G = nx.Graph(G)
 
    n = G.number_of_nodes()
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed number of nodes in G ({n})")
 
    nodes = list(G.nodes())
 
    if k == n:
        G_blocks = nx.Graph()
        for i, node in enumerate(nodes):
            G_blocks.add_node(i, size=1, members=[node], intra_edges=0)
        for u, v, d in G.edges(data=True):
            G_blocks.add_edge(nodes.index(u), nodes.index(v),
                               weight=d.get(weight, 1))
        return G_blocks, {node: i for i, node in enumerate(nodes)}
 
    has_weight = any(weight in d for _, _, d in G.edges(data=True))
    A = nx.to_numpy_array(G, nodelist=nodes, weight=weight if has_weight else None)
    A = np.maximum(A, 0.0)
 
    sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                             assign_labels="kmeans", random_state=random_state)
    labels = sc.fit_predict(A)
    assignment = {nodes[i]: int(labels[i]) for i in range(n)}
 
    members = {c: [] for c in range(k)}
    for node, c in assignment.items():
        members[c].append(node)
 
    # Raw EDGE COUNTS between/within blocks (off-diagonal = inter-block,
    # diagonal = intra-block, kept separately, no self-loops used).
    block_edge_count = np.zeros((k, k))
    for u, v in G.edges():
        cu, cv = assignment[u], assignment[v]
        block_edge_count[cu, cv] += 1
        if cu != cv:
            block_edge_count[cv, cu] += 1
 
    G_blocks = nx.Graph()
    for c in range(k):
        G_blocks.add_node(c, size=len(members[c]), members=members[c],
                           intra_edges=float(block_edge_count[c, c]))
    for i in range(k):
        for j in range(i + 1, k):
            w = block_edge_count[i, j]
            if w > 0:
                G_blocks.add_edge(i, j, weight=float(w))
 
    # Guarantee connectivity using the strongest available cross-component
    # block affinities (off-diagonal counts, including zeros if that's all
    # that's available).
    off_diag = block_edge_count.copy()
    np.fill_diagonal(off_diag, -1)  # never used for connectivity choice
    index_of = {c: c for c in range(k)}
    G_blocks = _connect_components_by_affinity(G_blocks, off_diag, index_of)
 
    return G_blocks #, assignment
 
 
def sbm_expand(G: nx.Graph, m: int, weight: str = "weight",
               intra_density_default: float = 0.3,
               random_state: int = 42):
    """
    Expand a small "block" graph G into a larger, connected m-node graph
    sampled from the stochastic block model implied by G's structure.
 
    Block sizes come from the node attribute 'size' if present (as
    produced by sbm_coarsen), otherwise blocks are split as evenly as
    possible. Inter-block probabilities come from edge weights
    (interpreted as edge COUNTS: p_ij = w_ij / (size_i * size_j));
    intra-block probability comes from the node attribute 'intra_edges'
    if present, otherwise `intra_density_default` is used.
 
    The output graph is guaranteed to be connected: after SBM sampling,
    any leftover disconnected components are stitched together with
    extra edges (tagged 'synthetic': True), chosen deterministically
    given `random_state`.
 
    Parameters
    ----------
    G : networkx.Graph
        The small "seed"/block graph.
    m : int
        Target number of nodes in the expanded graph (must be >= number
        of nodes in G).
    weight : str
        Edge attribute name to read block edge counts from.
    intra_density_default : float
        Fallback intra-block edge probability for blocks with no
        'intra_edges' attribute.
    random_state : int
        Seed for deterministic sampling.
 
    Returns
    -------
    G_expanded : networkx.Graph
        m nodes, guaranteed connected, each with a 'block' attribute
        indicating which seed node (block) it was sampled from. Edges
        added purely to enforce connectivity are tagged 'synthetic': True.
    """
    if not isinstance(G, nx.Graph):
        raise TypeError(f"G must be a networkx.Graph. Got {type(G)}")
    if G.is_directed() or G.is_multigraph():
        G = nx.Graph(G)
 
    blocks = list(G.nodes())
    n_blocks = len(blocks)
    if m < n_blocks:
        raise ValueError(f"m ({m}) cannot be smaller than the number of "
                          f"blocks in G ({n_blocks})")
 
    rng = np.random.RandomState(random_state)
 
    # --- How many new nodes does each block expand to? ---
    rel_sizes = np.array([G.nodes[b].get("size", 1) for b in blocks], dtype=float)
    rel_sizes = rel_sizes / rel_sizes.sum()
    raw = rel_sizes * m
    sizes = np.floor(raw).astype(int)
    sizes = np.maximum(sizes, 1)
    deficit = m - sizes.sum()
    if deficit > 0:
        frac_order = np.argsort(-(raw - np.floor(raw)))
        for idx in frac_order[:deficit]:
            sizes[idx] += 1
    elif deficit < 0:
        order = np.argsort(-sizes)
        i = 0
        while deficit < 0:
            b = order[i % len(order)]
            if sizes[b] > 1:
                sizes[b] -= 1
                deficit += 1
            i += 1
 
    # --- Block probability matrix ---
    P = np.zeros((n_blocks, n_blocks))
    for i, bi in enumerate(blocks):
        intra = G.nodes[bi].get("intra_edges", None)
        if intra is not None and intra > 0:
            denom = max(sizes[i] * (sizes[i] - 1) / 2.0, 1.0)
            P[i, i] = min(intra / denom, 1.0)
        else:
            P[i, i] = intra_density_default
        for j, bj in enumerate(blocks):
            if j <= i:
                continue
            if G.has_edge(bi, bj):
                w = G[bi][bj].get(weight, 1.0)
                denom = max(sizes[i] * sizes[j], 1.0)
                p = min(w / denom, 1.0)
            else:
                p = 0.0
            P[i, j] = p
            P[j, i] = p
 
    # --- Assign new node indices to blocks ---
    node_block = np.repeat(np.arange(n_blocks), sizes)
    G_expanded = nx.Graph()
    for idx, b in enumerate(node_block):
        G_expanded.add_node(idx, block=blocks[b])
 
    offsets = np.concatenate(([0], np.cumsum(sizes)))
 
    # --- Sample edges according to the block probability matrix ---
    for i in range(n_blocks):
        lo_i, hi_i = offsets[i], offsets[i + 1]
        p_ii = P[i, i]
        for u in range(lo_i, hi_i):
            for v in range(u + 1, hi_i):
                if rng.random_sample() < p_ii:
                    G_expanded.add_edge(u, v)
        for j in range(i + 1, n_blocks):
            lo_j, hi_j = offsets[j], offsets[j + 1]
            p_ij = P[i, j]
            if p_ij <= 0:
                continue
            for u in range(lo_i, hi_i):
                for v in range(lo_j, hi_j):
                    if rng.random_sample() < p_ij:
                        G_expanded.add_edge(u, v)
 
    # --- Guarantee connectivity: stitch together leftover components ---
    if not nx.is_connected(G_expanded):
        block_of = {node: G_expanded.nodes[node]["block"] for node in G_expanded.nodes()}
        block_idx_of = {b: i for i, b in enumerate(blocks)}
        affinity = np.zeros((m, m))
        for u in G_expanded.nodes():
            for v in G_expanded.nodes():
                if u == v:
                    continue
                bi, bj = block_idx_of[block_of[u]], block_idx_of[block_of[v]]
                affinity[u, v] = P[bi, bj]
        affinity += rng.random_sample((m, m)) * 1e-6
        index_of = {node: node for node in G_expanded.nodes()}
        G_expanded = _connect_components_by_affinity(G_expanded, affinity, index_of)
 
    return G_expanded #, None
