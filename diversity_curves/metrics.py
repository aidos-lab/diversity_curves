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