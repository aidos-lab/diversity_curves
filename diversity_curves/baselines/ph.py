# from https://github.com/aidos-lab/PH_expressivity

import networkx as nx
import numpy as np
from sklearn.manifold import spectral_embedding
import gudhi as gd

import warnings
import ot
import collections

from gtda.diagrams import PersistenceImage

# filtrations.py


def generate_shortest_path_distance_matrix(G: nx.Graph):
    shortest_paths = nx.all_pairs_shortest_path_length(G)
    nodes = G.nodes
    distance_matrix = np.ones((len(nodes), len(nodes))) * np.inf
    for node, shortest_paths_to_node in shortest_paths:
        for target_node, distance in shortest_paths_to_node.items():
            distance_matrix[node, target_node] = distance
    return distance_matrix


def vietoris_rips(G: nx.Graph) -> gd.RipsComplex:
    distance_matrix = generate_shortest_path_distance_matrix(G)
    cplx = gd.RipsComplex(distance_matrix=distance_matrix)
    return cplx


def alpha_complex(G: nx.Graph, num_components_points: int = 3) -> gd.AlphaComplex:
    # Graphs have no coordinates by default. So first we embed the graph into the euclidean space. We use an spectral
    # embedding technique given by Laplacian Eigenmaps.
    adj = nx.adjacency_matrix(G).todense()
    emb = spectral_embedding(
        adj, n_components=num_components_points, eigen_solver="lobpcg"
    )
    cplx = gd.AlphaComplex(points=emb)
    return cplx


def ollivier_ricci_curvature(G, alpha=0.0, weight=None, prob_fn=None) -> np.ndarray:
    """Calculate Ollivier--Ricci curvature of a graph.

    This function calculates the Ollivier--Ricci curvature of a graph,
    optionally taking (positive) edge weights into account.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    alpha : float
        Provides laziness parameter for default probability measure. The
        measure is not compatible with a user-defined `prob_fn`. If such
        a function is set, `alpha` will be ignored.

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. If None, unweighted curvature is calculated. Notice that
        if `prob_fn` is provided, this parameter will have no effect for
        the calculation of probability measures, but it will be used for
        the calculation of shortest-path distances.

    prob_fn : callable or None
        If set, should refer to a function that calculate a probability
        measure for a given graph and a given node. This callable needs
        to satisfy the following signature:

        ``prob_fn(G, node, node_to_index)``

        Here, `G` refers to the graph, `node` to the node whose measure
        is to be calculated, and `node_to_index` to the lookup map that
        maps a node identifier to a zero-based index.

        If `prob_fn` is set, providing `alpha` will not have an effect.

    Returns
    -------
    np.array
        An array of edge curvature values, following the ordering of
        edges of `G`.
    """
    assert 0 <= alpha <= 1

    # Ensures that we can map a node to its index in the graph,
    # regardless of whether node labels or node names are being
    # used.
    node_to_index = {node: idx for idx, node in enumerate(G.nodes)}

    # This is defined inline anyway, so there is no need to follow the
    # same conventions as for the `prob_fn` parameter.
    def _make_probability_measure(node):
        values = np.zeros(len(G.nodes))
        values[node_to_index[node]] = alpha

        degree = G.degree(node, weight=weight)

        for neighbor in G.neighbors(node):

            if weight is not None:
                w = G[node][neighbor][weight]
            else:
                w = 1.0

            values[node_to_index[neighbor]] = (1 - alpha) * w / degree

        return values

    # We pre-calculate all information about the probability measure,
    # making edge calculations easier later on.
    if prob_fn is None:
        measures = list(map(_make_probability_measure, G.nodes))
    else:
        measures = list(map(lambda x: prob_fn(G, x, node_to_index), G.nodes))

    # This is the cost matrix for calculating the Ollivier--Ricci
    # curvature in practice.
    M = nx.floyd_warshall_numpy(G, weight=weight)

    curvature = []
    # we get a curvature per edge

    for edge in G.edges():
        source, target = edge

        mi = measures[node_to_index[source]]
        mj = measures[node_to_index[target]]

        distance = ot.emd2(mi, mj, M)
        curvature.append(1.0 - distance)

    return np.asarray(curvature)


def _forman_curvature_unweighted(G):
    curvature = []
    for edge in G.edges():
        source, target = edge
        source_degree = G.degree(source)
        target_degree = G.degree(target)

        source_neighbours = set(G.neighbors(source))
        target_neighbours = set(G.neighbors(target))

        n_triangles = len(source_neighbours.intersection(target_neighbours))
        curvature.append(float(4 - source_degree - target_degree + 3 * n_triangles))

    return np.asarray(curvature)


def _forman_curvature_weighted(G, weight):
    has_node_attributes = bool(nx.get_node_attributes(G, weight))

    curvature = []
    for edge in G.edges:
        source, target = edge
        source_weight, target_weight = 1.0, 1.0

        # Makes checking for duplicate edges easier below. We expect the
        # source vertex to be the (lexicographically) smaller one.
        if source > target:
            source, target = target, source

        if has_node_attributes:
            source_weight = G.nodes[source][weight]
            target_weight = G.nodes[target][weight]

        edge_weight = G[source][target][weight]

        e_curvature = source_weight / edge_weight
        e_curvature += target_weight / edge_weight

        parallel_edges = list(G.edges(source, data=weight)) + list(
            G.edges(target, data=weight)
        )

        for u, v, w in parallel_edges:
            if u > v:
                u, v = v, u

            if (u, v) == edge:
                continue
            else:
                e_curvature -= w / np.sqrt(edge_weight * w)

        e_curvature *= edge_weight
        curvature.append(float(e_curvature))

    return np.asarray(curvature)


def forman_curvature(G, weight=None):
    """Calculate Forman--Ricci curvature of a graph.

    This function calculates the Forman--Ricci curvature of a graph,
    optionally taking (positive) node and edge weights into account.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. Will use the same attribute to look up node weights. If
        None, unweighted curvature is calculated.

    Returns
    -------
    np.array
        An array of edge curvature values, following the ordering of
        edges of `G`.
    """
    # This calculation is much more efficient than the weighted one, so
    # we default to it in case there are no weights in the graph.
    if weight is None:
        return _forman_curvature_unweighted(G)
    else:
        return _forman_curvature_weighted(G, weight)


def degrees(G: nx.Graph) -> list:
    """Calculate degrees vector."""
    return [deg for _, deg in nx.degree(G)]


def laplacian_eigenvalues(G: nx.Graph) -> np.ndarray:
    """Calculate Laplacian and return eigenvalues."""
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        return nx.laplacian_spectrum(G)


def get_filtration_function(filtration_type: str) -> callable:
    """
    It returns the filtration function and a boolean to indicate if it is a node-level filtration.
    :param filtration_type:
    :return:
    """
    match filtration_type:
        case "degree":
            return degrees, True
        case "ollivier_ricci_curvature":
            return ollivier_ricci_curvature, False
        case "forman_ricci_curvature":
            return forman_curvature, False
        case "laplacian":
            return laplacian_eigenvalues, True
        case "vietoris_rips":
            return vietoris_rips, False
        case "alpha_complex":
            return alpha_complex, False
        case _:
            raise TypeError(f"Filtration type {filtration_type} not recognized.")


# utils.py
def propagate_node_attribute_to_edges(G, attribute, pooling_fn=max):
    """Propagate a node attribute to edges.

    This function propagates a node attribute, such as the degree,
    to an edge attribute of the same name. This is done by calling
    a pooling function that condenses information of the attribute
    values of nodes incident on an edge.

    Parameters
    ----------
    G : networkx.Graph
        Input graph. Note that this graph will be changed **in place**.

    attribute : str
        Node attribute to use for the propagation procedure.

    pooling_fn : callable
        Function to pool node attribute information. Must be compatible
        with the node attribute type. The pooling function is called to
        summarize all node attributes that belong to an edge, i.e. only
        source and target node attributes.

        The pooling function must return a scalar value when provided
        with the source and target node of an edge. While other types
        of values are supported in principle, they will not result in
        graphs that are amenable to persistent homology calculations.
    """
    edge_attributes = dict()
    node_attributes = nx.get_node_attributes(G, attribute)

    for edge in G.edges(data=False):
        source, target = edge

        edge_attributes[edge] = pooling_fn(
            node_attributes[source], node_attributes[target]
        )

    nx.set_edge_attributes(G, edge_attributes, name=attribute)


def propagate_edge_attribute_to_nodes(G, attribute, pooling_fn=np.sum):
    """Propagate an edge attribute to nodes.

    This function propagates an edge attribute, such as a curvature
    measurement, to a node attribute of the same name. This is done
    by calling a pooling function that condenses information of the
    attribute values of edges incident on a node.

    Parameters
    ----------
    G : networkx.Graph
        Input graph. Note that this graph will be changed **in place**.

    attribute : str
        Edge attribute to use for the propagation procedure.

    pooling_fn : callable
        Function to pool edge attribute information. Must be compatible
        with the edge attribute type. The pooling function is called to
        summarize all edge attributes that belong to a node, i.e. *all*
        attributes belonging to incident edges.
    """
    node_attributes = collections.defaultdict(list)

    for edge in G.edges(data=True):
        source, target, data = edge

        node_attributes[source].append(data[attribute])
        node_attributes[target].append(data[attribute])

    node_attributes = {
        node: pooling_fn(values) for node, values in node_attributes.items()
    }

    nx.set_node_attributes(G, node_attributes, name=attribute)


def pad_diagrams(diagrams):
    """Pad persistence diagrams to be of the same length."""
    D = np.vstack(diagrams)

    min_d = int(np.min(D[:, 2]))
    max_d = int(np.max(D[:, 2]))

    max_features_per_dim = {}

    for diagram in diagrams:
        for dim in range(min_d, max_d + 1):
            n_features = np.sum(diagram[:, 2] == dim)

            max_features_per_dim[dim] = max(
                n_features, max_features_per_dim.get(dim, 2)
            )

    for dim, max_features in max_features_per_dim.items():
        for index, diagram in enumerate(diagrams):
            n_features = np.sum(diagram[:, 2] == dim)

            offset = max_features - n_features
            padding = np.zeros((offset, 3))

            # Ensure that values are being padded with the right
            # dimension.
            padding[:, 2] = dim

            # Somewhat inelegant: we just replace the stored diagram
            # with a padded one.
            diagrams[index] = np.vstack((diagram, padding))

    return diagrams


# diagrams.py
def obtain_persistence_diagram(st: gd.SimplexTree, k: int):
    persistence_pairs = st.persistence(persistence_dim_max=True)

    diagrams = []

    for dimension in range(k + 1):
        diagram = [
            (c, d, dimension) for dim, (c, d) in persistence_pairs if dim == dimension
        ]

        if len(diagram) == 0:
            diagram = [(0, 0, dimension)]

        diagrams.append(diagram)

    return diagrams


def calculate_persistent_homology_from_rips_complex(cmplx: gd.RipsComplex, k: int = 3):
    st = cmplx.create_simplex_tree(max_dimension=k)
    return obtain_persistence_diagram(st, k)


def calculate_persistent_homology_from_alpha_complex(
    cmplx: gd.AlphaComplex, k: int = 3
):
    st = cmplx.create_simplex_tree()
    return obtain_persistence_diagram(st, k)


def calculate_persistent_homology(G: nx.Graph, k: int = 3):
    """Calculate persistent homology of graph clique complex."""
    st = gd.SimplexTree()

    for v, w in G.nodes(data=True):
        weight = w["filtration"]
        st.insert([v], filtration=weight)

    for u, v, w in G.edges(data=True):
        weight = w["filtration"]
        st.insert([u, v], filtration=weight)

    st.make_filtration_non_decreasing()
    st.expansion(k)
    return obtain_persistence_diagram(st, k)


def get_filtration_annotated_graph_non_vr(
    graph: nx.Graph, filtration_fn: callable, node_level: bool = False
) -> nx.Graph:
    graph = graph.copy()  # We don't want to change the original graph
    filtration_values = filtration_fn(graph)
    if node_level:
        filtration_values = {v: c for v, c in zip(graph.nodes(), filtration_values)}
        nx.set_node_attributes(graph, filtration_values, "filtration")
    else:
        # Assign edge-based attribute. This is the assignment procedure
        # for when we are dealing with curvature filtrations.
        filtration_values = {e: c for e, c in zip(graph.edges(), filtration_values)}
        nx.set_edge_attributes(graph, filtration_values, "filtration")
    if node_level:
        propagate_node_attribute_to_edges(graph, "filtration")
    else:
        propagate_edge_attribute_to_nodes(graph, "filtration", pooling_fn=lambda x: -1)
    # FIX: propagate_edge_attribute_to_nodes does not propagate to nodes that are not connected to any other node.
    # As the only edge-filtrations are the Ollivier--Ricci and the Forman--Ricci curvatures, and the filtration values
    # for nodes are set to -1 by default, we can use this value to fill the missing values.
    if filtration_fn == ollivier_ricci_curvature or filtration_fn == forman_curvature:
        nx.set_node_attributes(graph, -1, "filtration")
    return graph


def get_filtration_annotated_graph(
    graph: nx.Graph, filtration_fn: callable, node_level: bool = False
) -> nx.Graph:
    if filtration_fn in [vietoris_rips, alpha_complex]:
        return filtration_fn(graph)
    else:
        return get_filtration_annotated_graph_non_vr(graph, filtration_fn, node_level)


def calculate_persistence_diagram(
    graph: nx.Graph, filtration_fn: callable, k: int, node_level: bool = False
) -> np.ndarray:
    if filtration_fn == vietoris_rips:
        cmplx = filtration_fn(graph)
        diagrams = calculate_persistent_homology_from_rips_complex(cmplx, k=k)
    elif filtration_fn == alpha_complex:
        cmplx = filtration_fn(graph)
        diagrams = calculate_persistent_homology_from_alpha_complex(cmplx, k=k)
    else:
        graph = get_filtration_annotated_graph(graph, filtration_fn, node_level)
        # Now we compute persistence diagrams
        diagrams = calculate_persistent_homology(graph, k=k)
    diagrams = np.vstack(diagrams)
    return diagrams


def calculate_persistence_diagrams(
    graphs: list[nx.Graph], filtration_fn: callable, k: int, node_level: bool = False
) -> list[np.ndarray]:
    persistence_diagrams = [
        calculate_persistence_diagram(graph, filtration_fn, k, node_level)
        for graph in graphs
    ]
    return persistence_diagrams


# classification_tasks.py
def obtain_input_from_graphs(graphs: list[nx.Graph], filtration_fn_name, k):
    filtration_fn, node_level = get_filtration_function(filtration_fn_name)
    persistence_diagrams = calculate_persistence_diagrams(
        graphs, filtration_fn, k, node_level=node_level
    )
    all_diagrams = pad_diagrams(persistence_diagrams)
    all_diagrams = np.stack(all_diagrams)
    # Remove infinities by associating the maximum value of creation and destruction.
    # This is a not a canonical way to handle infinities, and must be taken into account
    # for other applications.
    max_creation = np.max(np.ma.masked_invalid(all_diagrams)[:, :, 0])
    max_destruction = np.max(np.ma.masked_invalid(all_diagrams)[:, :, 1])
    all_diagrams[np.isinf(all_diagrams)] = max(max_creation, max_destruction)
    # Get topological vectorizations of persistence diagrams.
    featuriser = PersistenceImage(n_bins=10)
    X = featuriser.fit_transform(all_diagrams)
    X = X.reshape(X.shape[0], -1)
    return X
