import networkx as nx
import random
import numpy as np
import pandas as pd
import time
from grakel import datasets, GraphKernel, graph_from_networkx
from karateclub import NetLSD, Graph2Vec, FeatherGraph
import toper
import argparse
import signal

import mag_graph
from magnipy.magnitude.distances import get_dist
from mag_graph.process import get_mag_graphs, get_weight_diffs, get_funs_diff, get_mag_areas, get_mag_funs_graphs, get_funs_df
from mag_graph.baselines.spectral_zoo import est_moment
from mag_graph.baselines.ph import obtain_input_from_graphs

class OutOfTimeException(Exception):
    def __init__(self, message='Exceeded time limit'):
        super(OutOfTimeException, self).__init__(message)

def handler(signum, frame):
    signame = signal.Signals(signum).name
    print(f'Signal handler called with signal {signame} ({signum})')
    raise OutOfTimeException()

def gen_graphs_complexity(n_nodes,  n_graphs, nprop=2):
    np.random.seed(43)
    random.seed(314)
    graphs = {n: [] for n in n_nodes}
    for n in n_nodes:
        for i in range(n_graphs):
            graph = nx.erdos_renyi_graph(n=n, p=nprop/n)
            nx.set_node_attributes(graph, 1, name='feature') #dummy label for nx
            graphs[n].append(graph)
    return graphs

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--method', type=str, help='Whether to do regression on the parameters')

    args = argparser.parse_args()
    method = args.method

    # n_nodes = [2**i for i in range(4, 6)]
    n_nodes = [2**i for i in range(7, 14)]
    n_graphs = 30
    n_reruns = 10
    max_n_mag = 100

    graphs = gen_graphs_complexity(n_nodes=n_nodes, n_graphs=n_graphs, nprop=1)

    metric = 'shortest_path_distance'
    times = {}

    # magnitude
    if method == 'magnitude':
        times[f'spread_{metric}'] = {}
        for n in n_nodes:
            samples = []
            for i in range(n_reruns):
                
                compute_at_cardinality = [i for i in range(10,max_n_mag, 20)]
                start_time = time.perf_counter()
                def dist_fn(G):
                    return get_dist(G=G, X=None, X2=None, metric=metric, mode="structure")#"euclidean", mode="attributes") # structure # full

                all_spreads, weights_lists = get_mag_graphs(graphs[n], dist_fn, ts=[1], n_repeats=1, scores_method='random', compute_at_cardinality=compute_at_cardinality, mode="structure")

                end_time = time.perf_counter()

                samples.append(end_time - start_time) 
            times[f'spread_{metric}'][(n, 'mean')] = np.mean(samples)
            times[f'spread_{metric}'][(n, 'std')] = np.std(samples)

    # baselines
    # kernels
    if method == 'kernel':
        kernel_names = [
                    "shortest_path", 
                    #"graphlet_sampling", 
                    "pyramid_match", 
                    #"svm_theta",
                    "neighborhood_hash",
                    "subtree_wl",
                    "odd_sth",
                    "propagation",
                    "vertex_histogram",
                    "weisfeiler_lehman",
                    "core_framework",
                    "weisfeiler_lehman_optimal_assignment"#,
                    #"edge_histogram",
                    #"multiscale_laplacian"
                    ]
        for k_ in kernel_names:
            times[f'kernel_{k_}'] = {}
            for n in n_nodes:
                samples = []
                n_success = 0
                for i in range(n_reruns):

                    grakel_gen = graph_from_networkx(graphs[n], as_Graph=True, node_labels_tag='feature')
                    Gs = list(grakel_gen)
                    start_time = time.perf_counter()
                    # General kernels
                    if k_ in ["weisfeiler_lehman" , "core_framework"]:
                        gk = GraphKernel(kernel=[{"name": k_}, {"name": "shortest_path"}], normalize=True)
                    # Base kernels
                    else:
                        gk = GraphKernel(kernel=[{"name": k_}], normalize=True)
                    # Calculate the kernel matrix
                    try:
                        K = gk.fit_transform(Gs)
                        end_time = time.perf_counter()
                        samples.append(end_time - start_time)
                        success += 1
                    except Exception as e:
                        print(f"Kernel {k_} failed with error: {e}")
                if n_success == n_reruns:
                    times[f'kernel_{k_}'][(n, 'mean')] = np.mean(samples)
                    times[f'kernel_{k_}'][(n, 'std')] = np.std(samples)

    # graph embeddings
    if method == 'embedding':
        embeddings = {
            'netLSD': NetLSD(),
            'graph2vec': Graph2Vec(),
            'feathergraph': FeatherGraph()
        }

        for embedding_name in embeddings:
            times[f'embedding_{embedding_name}'] = {}
            for n in n_nodes:
                samples = []
                for i in range(n_reruns):
                    start_time = time.perf_counter()
                    embedding = embeddings[embedding_name]
                    embedding.fit([g.copy() for g in graphs[n]])
                    array = embedding.get_embedding()
                    end_time = time.perf_counter()
                    samples.append(end_time - start_time)
                times[f'embedding_{embedding_name}'][(n, 'mean')] = np.mean(samples)
                times[f'embedding_{embedding_name}'][(n, 'std')] = np.std(samples)


    # graph embeddings
    if method == 'ph':
        ph_filtrations = ['degree', 'ollivier_ricci_curvature', 'forman_ricci_curvature', 'laplacian', 'vietoris_rips', 'alpha_complex']

        for filtration in ph_filtrations:
            times[f'ph_{filtration}'] = {}
            for n in range(n_nodes):
                samples = []
                n_success = 0
                for i in range(n_reruns):
                    try:
                        start_time = time.perf_counter()
                        array_ph = obtain_input_from_graphs(graphs[n], filtration_fn_name=filtration, k=2)
                        end_time = time.perf_counter()
                        samples.appemd(end_time - start_time)
                        success += 1
                    except MemoryError as e:
                        print(f'Memory Error {e} for filtration {filtration}')
                if n_success == n_reruns:
                    times[f'embedding_{embedding_name}'][(n, 'mean')] = np.mean(samples)
                    times[f'embedding_{embedding_name}'][(n, 'std')] = np.std(samples)

    #toper
    if method == 'toper':
        for filt in ["oricci"]: #, "closeness" "degree", "popularity", "fricci", 
            times[f'toper_{filt}'] = {}
            for n in n_nodes:
                samples = []
                success = 0
                for i in range(n_reruns):
                    try:
                        # signal.signal(signal.SIGALRM, handler)
                        # signal.alarm(5)
                        start_time = time.perf_counter()
                        F = toper.main(graphs[n], [filt])#, num_segm=501) "closeness",
                        end_time = time.perf_counter()
                        # signal.alarm(0)
                        # if filt == "fricci":
                        #     filt = "forricci"
                        # elif filt == "oricci":
                        #     filt = "olricci"
                        # array_super = np.array(F[f"{filt}_super"])
                        # array_sub = np.array(F[f"{filt}_sub"])
                        # array_sub_super = np.array(np.concatenate([F[f"{filt}_sub"], F[f"{filt}_super"]], axis=1))
                        # print(f'shape of the toper arrays: {array_sub.shape}, {array_sub_super.shape}, {array_super.shape}')
                        samples.append(end_time - start_time)
                        success += 1
                    except OutOfTimeException as e:
                        print(f"Toper failed with error: {e}")
                        samples.append(5)
                        success = True
                    except Exception as e:
                        print(f"Toper failed with error: {e}")
                    # finally:
                    #     signal.alarm(0)
                if success:
                    # if filt == "forricci":
                    #     filt = "fricci"
                    # elif filt == "olricci":
                    #     filt = "oricci"
                    times[f'toper_{filt}'][(n, 'mean')] = np.mean(samples)
                    times[f'toper_{filt}'][(n, 'std')] = np.std(samples)
            
        
        
    # Spectral Zoo
    if method == 'spectralzoo':
        times[f'spectral_zoo'] = {}
        for n in n_nodes:
            samples = []
            for i in range(n_reruns):
                start_time = time.perf_counter()
                moments = []
                for graph in graphs[n]:
                    moments.append(est_moment(graph, 10000, 4))
                #np.array(moments)
                end_time = time.perf_counter()
                array=np.array(moments)#[:, 1:]
                array = array[:, ~np.all(array == 0, axis=0)]
                samples.append(end_time - start_time)
            times[f'spectral_zoo'][(n, 'mean')] = np.mean(samples)
            times[f'spectral_zoo'][(n, 'std')] = np.std(samples)

    times_df = pd.DataFrame.from_dict(times, orient="index")
    times_df.to_csv(f"./results/time_complexities_{method}_ngraphs_{n_graphs}_maxnodes_{max(n_nodes)}.csv")