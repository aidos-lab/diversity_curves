import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap
import random
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from sklearn.neighbors import KNeighborsClassifier
import pickle
from tqdm import tqdm
import pandas as pd
import glob
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from anndata import AnnData
from scipy.spatial import distance
from numpy import linalg as LA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import sklearn
import pandas as pd
from sklearn.manifold import TSNE
import umap
import glob, os
from sklearn.neighbors import kneighbors_graph
import time
import sys
from scipy import sparse
from scipy.stats import rankdata
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.neighbors import KNeighborsClassifier
import random
from pathlib import Path
from geosketch import gs
import networkx as nx
from diversity_curves.process import (
    get_mag_graphs,
    get_weight_diffs,
    get_funs_diff,
    get_mag_areas,
    get_mag_funs_graphs,
    get_funs_df,
)
from magnipy.magnitude.distances import get_dist
import pandas as pd
from diversity_curves.coarsen import repeat_edge_dropping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from diversity_curves.diversity import compute_magnitude_graph
import signal
import argparse
import time
import os
import pickle as pkl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", type=str, nargs="?", default="spread", help="Which method to use"
    )
    args = parser.parse_args()

    method = args.method

    in_path = "../"
    path = in_path + "data/sc/real_data_dict.pkl"

    with open(path, "rb") as f:
        x = pickle.load(f)

    types = pd.read_csv(in_path + "/data/sc/data_type.csv")

    features = "scaled"

    results_path = "./results/"

    N = 200
    n_neighbors = 10

    if os.path.exists(results_path) == False:
        os.makedirs(results_path)

    if os.path.exists(f"{results_path}G_nx.pkl"):
        G_nx = pkl.load(open(f"{results_path}G_nx.pkl", "rb"))
    else:
        n_obs = []
        n_vars = []
        names = []
        types_list = []
        G_nx = []
        keys = x.keys()
        for k in keys:
            name = k
            print(name)
            typet = x[k]["data_type"]
            print(typet)
            data = x[k]["pca_data"]
            n_obs.append(data.shape[0])
            n_vars.append(data.shape[1])
            names.append(name)
            types_list.append(typet)
            try:
                sketch_index = gs(data, N, replace=False)
                data = data[sketch_index]
            except:
                print("Error in geosketch for dataset ", name)
            data_sketch = AnnData(data)
            sc.pp.neighbors(data_sketch, n_neighbors=n_neighbors)
            sc.tl.umap(data_sketch)
            sc.tl.leiden(data_sketch, resolution=0.5)
            # sc.pl.umap(data_sketch, color="leiden", title=name)
            graph = data_sketch.obsp["connectivities"]
            nx_graph = nx.from_numpy_array(graph.toarray())
            data_norm = StandardScaler().fit_transform(data_sketch.X)
            for j in range(data_sketch.shape[0]):
                nx_graph.nodes[j]["feature"] = data_norm[j]
            G_nx.append(nx_graph)

    pkl.dump(G_nx, open(f"{results_path}G_nx.pkl", "wb"))

    if method == "spread":

        for metric in ["shortest_path_distance", "diffusion_distance"]:

            def dist_fn(G):
                return get_dist(G=G, X=None, X2=None, metric=metric, mode="structure")

            compute_at_cardinality = np.linspace(10, N, 10, dtype=int)

            all_spreads2, weights_lists2 = get_mag_graphs(
                G_nx,
                dist_fn,
                ts=[1],
                upsample_to=N,
                n_repeats=3,
                compute_at_cardinality=compute_at_cardinality,
                mode="structure",
            )
            fun_df2 = get_funs_df(all_spreads2)

            fun_df2.to_csv(f"{results_path}emb_spread_{metric}.csv")

        def dist_fn(G):
            return get_dist(
                G=G,
                X=None,
                X2=None,
                metric="euclidean",
                mode="attributes",
                check_for_duplicates=False,
            )

        compute_at_cardinality = np.linspace(10, N, 10, dtype=int)

        all_spreads, weights_lists = get_mag_graphs(
            G_nx,
            dist_fn,
            ts=[1],
            upsample_to=N,
            n_repeats=3,
            compute_at_cardinality=compute_at_cardinality,
        )

        fun_df = get_funs_df(all_spreads)

        fun_df.to_csv(f"{results_path}emb_spread_features.csv")

    elif method in ["netlsd", "graph2", "feather"]:

        G_nx_fixed = [
            nx.convert_node_labels_to_integers(g, ordering="sorted") for g in G_nx
        ]

        from karateclub import NetLSD, Graph2Vec, FeatherGraph

        if method == "netlsd":
            embedding = NetLSD()
        elif method == "graph2":
            embedding = Graph2Vec()
        elif method == "feather":
            embedding = FeatherGraph()

        embedding.fit([g.copy() for g in G_nx_fixed])
        array = embedding.get_embedding()

        pd.DataFrame(array).to_csv(f"{results_path}{method}.csv", index=False)

    elif method == "kernel":
        from grakel import GraphKernel
        from grakel import Graph, graph_from_networkx

        Gs = list()
        for graph in G_nx:
            nx.set_node_attributes(graph, 1, name="feature")  # dummy label for nx
            node_labels = {u: 1 for u in graph.nodes()}  # dummy label
            grakel_gen = graph_from_networkx(
                [graph], as_Graph=True, node_labels_tag="feature"
            )
            Gs.append(next(grakel_gen))
        kernel_names = [
            "shortest_path",
            "pyramid_match",
            "neighborhood_hash",
            "weisfeiler_lehman",
            "core_framework",
            "weisfeiler_lehman_optimal_assignment",
        ]

        for k_ in kernel_names:
            # General kernels
            if k_ in ["weisfeiler_lehman", "core_framework"]:
                gk = GraphKernel(
                    kernel=[{"name": k_}, {"name": "shortest_path"}], normalize=True
                )

            # Base kernels
            else:
                gk = GraphKernel(kernel=[{"name": k_}], normalize=True)

            def handler(signum, frame):
                raise TimeoutError

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(300)

            # Calculate the kernel matrix
            kernel_success = False
            try:
                K = gk.fit_transform(Gs)
                kernel_success = True

                pd.DataFrame(K).to_csv(f"{results_path}kernel_{k_}.csv", index=False)

            except Exception as e:
                print(f"Kernel {k_} failed with error: {e}")
            finally:
                signal.alarm(0)

    elif method == "toper":
        import toper

        m_list = []
        time_list = []
        filts = ["degree", "popularity", "fricci", "oricci", "closeness"]

        for filt in filts:

            def handler(signum, frame):
                raise TimeoutError

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(800)
            try:
                F = toper.main(G_nx, [filt], num_segm=501)
                if filt == "fricci":
                    filt = "forricci"
                elif filt == "oricci":
                    filt = "olricci"

                array = np.array(F[f"{filt}_super"])

                pd.DataFrame(array).to_csv(
                    f"{results_path}toper_{filt}_super.csv", index=False
                )

                print(f"{filt}_super")
                array = np.array(F[f"{filt}_sub"])

                pd.DataFrame(array).to_csv(
                    f"{results_path}toper_{filt}_sub.csv", index=False
                )

                print(f"{filt}_sub")
                m_list.append(filt)

            except TimeoutError:
                print("Timed out!")
            finally:
                signal.alarm(0)

    elif method == "spectral_zoo":
        from diversity_curves.baselines.spectral_zoo import est_moment

        moments = []
        for graph in G_nx:
            moments.append(est_moment(graph, 10000, 4))

        array = np.array(moments)
        array = array[:, ~np.all(array == 0, axis=0)]

        pd.DataFrame(array).to_csv(
            f"{results_path}spectral_zoo_moments.csv", index=False
        )
