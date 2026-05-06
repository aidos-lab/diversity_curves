"Try custom graph distances not yet implemented in magnipy."

from magnipy.magnitude.distances import _compute_spectrum
import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances
import warnings
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from hodgelaplacians import HodgeLaplacians
from magnipy.magnitude.compute import compute_magnitude_from_distances
from toponetx import SimplicialComplex

def compute_heat_kernel_distances(L2, t=1.0):
    """
    Computes the heat kernel distance matrix from a Hodge Laplacian L2.
    
    Parameters:
    L2 : ndarray, shape (F, F)
        The 2nd Hodge Laplacian (d2.T @ d2)
    t  : float
        The time scale (diffusion time). 
        Small t = local structure; Large t = global topology.
    """
    # 1. Compute eigenvalues and eigenvectors
    # eigh is used because L2 is symmetric
    evals, evecs = eigh(L2)
    
    # 2. Apply the heat kernel scaling (e^-t*lambda)
    # We ignore very small eigenvalues to avoid numerical noise
    heat_coords = evecs * np.exp(-t * evals)
    
    # 3. Compute pairwise Euclidean distances in the heat kernel space
    # pdist computes compressed distance, squareform expands it to a matrix
    dist_matrix = squareform(pdist(heat_coords))
    
    return dist_matrix

def compute_diffusion_distances(L2, t=1.0):
    """
    Computes the diffusion distance matrix from a Hodge Laplacian L2.
    
    Parameters:
    L2 : ndarray, shape (F, F)
        The 2nd Hodge Laplacian (d2.T @ d2)
    t  : float
        The time scale (diffusion time). 
        Small t = local structure; Large t = global topology.
    """
    # 1. Compute eigenvalues and eigenvectors
    # eigh is used because L2 is symmetric
    evals, evecs = eigh(L2)
    
    # 2. Apply the diffusion scaling (e^-t*lambda)
    # We ignore very small eigenvalues to avoid numerical noise
    diffusion_coords = evecs * np.exp(-t * evals)
    
    # 3. Compute pairwise Euclidean distances in the diffusion space
    # pdist computes compressed distance, squareform expands it to a matrix
    dist_matrix = squareform(pdist(diffusion_coords))
    
    return dist_matrix

def nx_to_simplicial_complex(G: nx.Graph) -> SimplicialComplex:
    """Convert a NetworkX graph to a TopoNetX SimplicialComplex."""
    simplices = []
    # Add nodes as 0-simplices
    for node in G.nodes():
        simplices.append([node])
    # Add edges as 1-simplices
    for u, v in G.edges():
        simplices.append([u, v])
    # Create the SimplicialComplex
    K = SimplicialComplex()
    K.add_simplices_from(simplices)
    return K

def compute_mag_L2(mani, t=1):
    simplices = mani.triangulation
    try:
        hl = HodgeLaplacians(simplices)
        L2 = hl.getHodgeLaplacian(2)
        L2 = L2.toarray()
        D2 = compute_diffusion_distances(L2, t=t)
        magK = compute_magnitude_from_distances(D2, ts=[1], method="spread")
    except:
        magK = [np.nan]
    return magK[0]

def hodge_diffusion_distance(simplices,h=2, t=1):
    hl = HodgeLaplacians(simplices)
    L2 = hl.getHodgeLaplacian(h)
    L2 = L2.toarray()
    D2 = compute_diffusion_distances(L2, t=t)
    return D2

def hodge_heat_kernel_distance(simplices,h=2, t=1):
    hl = HodgeLaplacians(simplices)
    L2 = hl.getHodgeLaplacian(h)
    L2 = L2.toarray()
    D2 = compute_heat_kernel_distances(L2, t=t)
    return D2

def hodge_diffusion_distance_graph(graph,h=2, t=1):
    simplicial_complex = nx_to_simplicial_complex(graph)
    simplices = simplicial_complex.simplices
    hl = HodgeLaplacians(simplices)
    L2 = hl.getHodgeLaplacian(h)
    L2 = L2.toarray()
    D2 = compute_diffusion_distances(L2, t=t)
    return D2

def hodge_diffusion_distance_faces(manifold,h=2, t=1):
    #simplicial_complex = nx_to_simplicial_complex(graph)
    #print(manifold)
    #print(len(manifold))
    simplices = manifold[h]#simplicial_complex.simplices
    hl = HodgeLaplacians(simplices)
    L2 = hl.getHodgeLaplacian(h)
    L2 = L2.toarray()
    D2 = compute_diffusion_distances(L2, t=t)
    return D2

def hodge_heat_kernel_distance_faces(manifold,h=2, t=1):
    #simplicial_complex = nx_to_simplicial_complex(graph)
    #print(manifold)
    #print(len(manifold))
    simplices = manifold[h]#simplicial_complex.simplices
    hl = HodgeLaplacians(simplices)
    L2 = hl.getHodgeLaplacian(h)
    L2 = L2.toarray()
    D2 = compute_heat_kernel_distances(L2, t=t)
    return D2

def attributed_diffusion_distance(G=None, A=None, X=None, **kwargs):
    """Calculate diffusion distance between vertices of a graph.
    https://www.seas.upenn.edu/~aribeiro/preprints/c_2015_huang_etal.pdf

    Parameters
    ----------
    G : nx.Graph
        Input graph. All attributes of the graph will be ignored in
        the subsequent calculations.

    num_steps : int
        Number of steps for the diffusion operator that is used for
        the distance calculations.

    Returns
    -------
    np.array
        Matrix of distance values
    """
    t = kwargs.get("num_steps", 1)
    n_jobs = kwargs.get("n_jobs", None)
    symmetric = kwargs.get("symmetric", True)
    norm = kwargs.get("norm", True)
    alpha = kwargs.get("alpha", 1)

    if not norm and not symmetric:
        warnings.warn(
            "Assuming default Laplacian, which is symmetric.",
            UserWarning,
        )
        symmetric = True

    if G is None and A is None:
        raise Exception("Either a graph G or adjacencies A must be provided.")

    if G is not None and A is not None:
        raise Exception("Provide either a graph G or adjacencies A, not both.")

    if A is None:
        A = nx.adjacency_matrix(G).todense()
    D = np.diag(A.sum(axis=1))

    # Bail out early on if there are isolated nodes.
    if (np.diagonal(D) == 0).any():
        return np.nan

    L = D - A

    # NEW VERSION
    #L = _compute_laplacian(A, D, norm=norm, symmetric=symmetric)
    #psi = _compute_psi(L, t, symmetric=symmetric)

    if X is None:
        if G.nodes[next(iter(G.nodes))].get("feature") is not None:
            X = np.array([G.nodes[i]["feature"] for i in G.nodes])
        else:
            raise Exception("Feature matrix X must be provided.")
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = A.shape[0]
    if X.shape[0] != n:
        raise Exception("Number of rows in X must match number of nodes in A.")
    L = np.asarray(L)
    M = np.eye(n) + alpha * L
    try:
        psi = np.linalg.solve(M, X)
    except np.linalg.LinAlgError:
        # fallback to pseudo-inverse if M is singular
        psi = np.linalg.pinv(M) @ X

    if np.iscomplexobj(psi):
        warnings.warn(
            "Input data contains complex numbers. The imaginary part will be discarded.",
            UserWarning,
        )
        psi = np.real(psi)
    return pairwise_distances(psi, metric="euclidean", n_jobs=n_jobs)