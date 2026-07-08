from magnipy.magnitude.compute import compute_magnitude_from_distances
import copy
import numpy as np
import networkx as nx
from magnipy.magnitude.convergence import mag_convergence
from magnipy.magnitude.scales import get_scales, median_heuristic


def compute_magnitude_graph(
    G,
    dist_fn,
    ts=None,
    target_value=None,
    n_ts=10,
    log_scale=False,
    scale_finding="convergence",
    method="cholesky",
    get_weights=False,
):
    """
    Compute the magnitude of a graph using a specified distance function.
    The magnitude is computed either across a fixed choice of scales
    or until the magnitude function has reached a certain target value.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    dist_fn : function
        A function that takes a graph as input and returns a distance matrix.
    ts : None or array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
        Alternativally, if ts is None, the evaluation scales will be choosen automatically.
    scale_finding : str
        The method used to find a convergence scale. Must be one of 'convergence' or 'median_heuristic'.
        If 'convergence' is chosen, the scale will be found by evaluating magnitude at increasing
        scales until the magnitude function reaches the target value.
        If 'median_heuristic' is chosen, the scale will be found using the median heuristic.
        Only used if ts is None.
    target_value : float
        The value of margnitude that should be reached. Only used if ts is None and scale_finding is 'convergence'.
    n_ts : int
        The number of evaluation scales that should be sampled. Only used if ts is None.
    log_scale : bool
        If True sample evaluation scales on a logarithmic scale instead of evenly. Only used if ts is None.
    method : str
        The method used to compute magnitude. If 'cholesky' is chosen, the Cholesky decomposition
        will be used to compute magnitude. If 'spread' is chosen, the spread of a metric space will be computed.
    get_weights : bool
        If True output the magnitude weights. If False output the magnitude function.

    Returns
    -------
    magnitude : array_like, shape (`n_ts`, ) or shape (`n_obs`, `n_ts`)
        Either a vector with the values of the magnitude function
        or a matrix with the magnitude weights (whose ij-th entry is the magnitude weight
        of the ith observation evaluated at the jth scaling parameter).
    ts : array_like, shape (`n_ts`, )
        The scales at which magnitude has been evaluated.
    """

    subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    if ts is None:
        if G.number_of_nodes() == 1:
            raise Exception(
                "We cannot find the convergence scale for a one point space!"
            )

        if scale_finding == "convergence":

            def comp_mag(G, ts):
                return compute_magnitude_subgraphs(
                    G=G,
                    ts=ts,
                    dist_fn=dist_fn,
                    subgraphs=subgraphs,
                    method=method,
                    one_point_property=True,
                    perturb_singularities=True,
                    positive_magnitude=False,
                )[0]

            if target_value is None:
                target_value = 0.95 * G.number_of_nodes()
            else:
                if target_value >= G.number_of_nodes():
                    raise Exception(
                        "The target value needs to be smaller than the cardinality!"
                    )
                if 0 >= target_value:
                    raise Exception("The target value needs to be larger than 0!")

            t_conv = guess_convergence_scale(
                G, comp_mag=comp_mag, target_value=target_value, guess=10
            )
        else:
            t_conv = median_heuristic(dist_fn, G=None, subgraphs=subgraphs)

        if n_ts == 1:
            ts = [t_conv]
        else:
            ts = get_scales(
                t_conv,
                n_ts,
                log_scale=log_scale,
                one_point_property=True,
            )

    magnitude, ts = compute_magnitude_subgraphs(
        G,
        ts,
        dist_fn,
        subgraphs=subgraphs,
        method=method,
        get_weights=get_weights,
        one_point_property=True,
        perturb_singularities=True,
        positive_magnitude=False,
    )

    return magnitude, ts


def compute_magnitude_subgraphs(
    G,
    ts,
    dist_fn,
    mode="structure",
    subgraphs=None,
    method="cholesky",
    get_weights=False,
    one_point_property=True,
    perturb_singularities=True,
    positive_magnitude=False,
    input_distances=True,
):
    """
    Compute the magnitude of a graph using a specified distance function.
    The magnitude is computed across a fixed choice of scales.
    This function computes the magnitude of each connected component
    of the graph separately and sums them up to obtain the total magnitude.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    dist_fn : function
        A function that takes a graph as input and returns a distance matrix.
    subgraphs : list of networkx.Graph, optional
        A list of subgraphs. If provided, the magnitude will be computed
        on each subgraph and summed up. If None, the connected components
        of the graph will be used as subgraphs.
    method : str
        The method used to compute magnitude. If 'cholesky' is chosen, the Cholesky decomposition
        will be used to compute magnitude. If 'spread' is chosen, the spread of a metric space will be computed.
    get_weights : bool
        If True output the magnitude weights. If False output the magnitude function.

    Returns
    -------
    magnitude : array_like, shape (`n_ts`, ) or shape (`n_obs`, `n_ts`)
        Either a vector with the values of the magnitude function
        or a matrix with the magnitude weights (whose ij-th entry is the magnitude weight
        of the ith observation evaluated at the jth scaling parameter).
    ts : array_like, shape (`n_ts`, )
        The scales at which magnitude has been evaluated.
    """
    # original_G = copy.deepcopy(G)
    if subgraphs is None:
        subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    mags = []

    for s in subgraphs:
        if mode == "structure":
            D = dist_fn(G=s)
            # D = dist_fn(G=s)
        elif mode == "attributes":
            features = np.array([s.nodes[node]["feature"] for node in s.nodes()])
            D = dist_fn(X=features)
        elif mode == "full":
            features = np.array([s.nodes[node]["feature"] for node in s.nodes()])
            D = dist_fn(X=features, G=s)
            # D_struct = dist_fn(G=s)
            # D = D_attr + D_struct

        mag = compute_magnitude_from_distances(
            D,
            ts=ts,
            method=method,
            get_weights=get_weights,
            one_point_property=one_point_property,
            perturb_singularities=perturb_singularities,
            positive_magnitude=positive_magnitude,
            input_distances=input_distances,
        )
        mags.append(mag)

    if get_weights:
        # weights = np.zeros((original_G.number_of_nodes(), len(ts)))
        # node_idx = {node: idx for idx, node in enumerate(original_G.nodes)}

        # for subgraph, (mag) in zip(subgraphs, mags):
        #    for nn, node in enumerate(subgraph.nodes):
        #        weights[node_idx[node], :] = mag[nn]
        mag_weights = np.vstack(mags)
        mag_weights = np.sort(mag_weights, axis=0)
        return mag_weights, ts
    else:
        total_magnitude = np.sum([mag for mag in mags], axis=0)
        return total_magnitude, ts


def guess_convergence_scale(D, comp_mag, target_value, guess=10):
    """
    Compute the scale at which the magnitude function has reached a certain target value
    using numeric root-finding.
    The target value is typically set to a high proportion of the cardinality.
    This pocedure assumes the magnitude function is typically non-decreasing.

    Parameters
    ----------
    D : array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    target_value : float
        The value of margnitude that should be reached.
        This value needs to be larger than 1 and smaller than the cardinality of the space.
    comp_mag : function
        A function that computes the magnitude given a distance matrix and a vector of scales.
    guess :
        An initial guess for the scaling parameter.

    Returns
    -------
    t_conv : float
        The scaling parameter at which the magnitude function reaches the target value.
    """

    def f(x, W=D):
        mag = comp_mag(W, ts=[x])
        return mag[0] - target_value

    ### n/t =< Mag(t) =< t^n |A|
    ### 1 =< Mag(t) * t/n
    ### n/Mag(t) =< t #Meckes for Euclidean space
    lower_guess = 0
    f_guess = f(guess)
    while f_guess < 0:
        lower_guess = guess
        guess = guess * 10
        f_guess = f(guess)

    t_conv = mag_convergence(lower_guess, guess, f, max_iterations=100)
    return t_conv
