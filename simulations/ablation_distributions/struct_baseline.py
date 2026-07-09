import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import argparse
import os
import random
import networkx as nx

from grakel import GraphKernel
from karateclub import NetLSD, Graph2Vec, FeatherGraph
import toper

from sklearn.manifold import MDS
from sklearn.decomposition import PCA

from magnipy.magnitude.distances import get_dist
from diversity_curves.process import get_mag_graphs, get_funs_df, get_mag_funs_graphs

from diversity_curves.simulate import simulate_graphs_kwargs
from diversity_curves.classifiers import knn_classifier
from diversity_curves.clustering_performance import compute_clustering_scores_resample, compute_silhouette_score_resamples, compute_rand_agglo, compute_rand_kmeans, all_clustering_results_euclidean
from diversity_curves.baselines.spectral_zoo import est_moment
from diversity_curves.baselines.ph import obtain_input_from_graphs

if __name__ == '__main__':
    np.random.seed(43)
    random.seed(314)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output_dir", type=str, default="./simulations/results/sim_distributions/", help="Output directory for results")
    argparser.add_argument('--ngraphs', type=int, default=3, help='Number of graphs that are generated per cardinality')
    argparser.add_argument('--nrange', type=int, nargs=2, default=[10,30], help='specify the range of cardinalities of the generated graphs')
    argparser.add_argument('--nsteps', type=int, default=1, help='Specify the stepsize for the cardinalities of the generated graphs')
    

    args = argparser.parse_args()
    results_dir = args.output_dir
    n_low = args.nrange[0]
    n_high = args.nrange[1]
    n_steps = args.nsteps
    n_graphs = args.ngraphs

    # Setting the parameters as reported in the paper
    model_params = {
        'ER': {'ER_probs': [0.75], 
                'ER_nps': None,
        },
        'RPG':{
            'RPG_probs': [[0.9, 0.1]], 
            'RPG_props': [1/3, 1/3], 
        },
        'SBM': {
            'SBM_probs': [0.8, 0.05], 
            'SBM_blocks': [[1/4, 1/4, 1/4]]
            },
        'RG': {
            'RG_probs': [0.25]
        },
    }

    # choice of graph metric we use for the spread computation
    graph_metrics = ['shortest_path_distance']

    # get configuration of all models
    models = list(model_params.keys())
    all_configs = {}
    for model in models:
        model_config = tuple(model_params[model][param] for param in model_params[model].keys())
        all_configs[model] = model_config
    configuration = tuple(all_configs[model] for model in models)

    noderange = range(n_low, n_high, n_steps)
    downsample_to = None

    scores = ['acc_mean', 'silh_mean', 'ari_agglo_mean', 'db_mean',	'ch_mean', 'ari_kmeans_mean']
    scores_asc = {'acc_mean': False, 'silh_mean': False, 'ari_agglo_mean': False, 'db_mean': True,'ch_mean': False, 'ari_kmeans_mean': False}

    res_dict = {}

    if not os.path.exists(results_dir): 
        os.makedirs(results_dir, exist_ok=True) 

    # generate the graphs of all models and create label per model
    G_nx = []
    Gs = []
    n_nodes = []
    y_models = []
    class_labels = {}
    for i_m, model in enumerate(model_params):
        model_kwargs = model_params[model]
        print(f'model {model} kwargs {model_kwargs}')
        G_nx_model, Gs_model, y_model, y_probs_model, n_nodes_model = simulate_graphs_kwargs(
            model = model,
            n_graphs=n_graphs,
            nodes_range=noderange,
            **model_kwargs,
            )
        G_nx.extend(G_nx_model)
        Gs.extend(Gs_model)
        n_nodes.extend(n_nodes_model)
        y_models.extend([i_m]*len(G_nx_model))
        class_labels[i_m] = model

    upsample_to = max(n_nodes) 
    y=y_models
    hue_names = [class_labels[i] for i in y]

    # setting parameters for plots
    sns.set_style('white')
    matplotlib.rcParams.update({'font.size': 20})
    
    ##################################
    ### computing diversity curves ###
    ##################################

    for metric in graph_metrics:

        res_dict[f'divcurves_{metric}'] = {}
        res_dict[f'divcurves_pca_{metric}'] = {}
        res_dict[f'div_function_{metric}'] = {}

        print(f"Computing MAG with metric: {metric}")

        def dist_fn(G):
            return get_dist(G=G, X=None, X2=None, metric=metric, mode="structure")
        
        all_spreads, weights_lists = get_mag_graphs(G_nx, dist_fn, ts=[1], upsample_to=upsample_to, downsample_to=downsample_to, n_repeats=3)
        
        mag_funs = get_funs_df(all_spreads)
        mag_funs_interpolated = mag_funs.interpolate(method='linear', axis='columns')

        # compute clustering metrics based on euclidean distance of diversity curve vectors 
        ari_kmeans_mean, ari_kmeans_std = compute_rand_kmeans(np.array(mag_funs_interpolated), y)
        ari_agglo_mean, ari_agglo_std = compute_rand_agglo(np.array(mag_funs_interpolated), y, metric='euclidean')

        means, stds = compute_clustering_scores_resample(np.array(mag_funs_interpolated), y)

        res_dict[f'divcurves_{metric}']['silh_mean'], res_dict[f'divcurves_{metric}']['db_mean'], res_dict[f'divcurves_{metric}']['ch_mean'] = means
        res_dict[f'divcurves_{metric}']['silh_std'], res_dict[f'divcurves_{metric}']['db_std'], res_dict[f'divcurves_{metric}']['ch_std'] = stds
        res_dict[f'divcurves_{metric}']['ari_agglo_mean'] = ari_agglo_mean
        res_dict[f'divcurves_{metric}']['ari_agglo_std'] = ari_agglo_std
        res_dict[f'divcurves_{metric}']['ari_kmeans_mean'] = ari_kmeans_mean
        res_dict[f'divcurves_{metric}']['ari_kmeans_std'] = ari_kmeans_std

        # computing accuracies
        if n_graphs > 1:
            test_accs = knn_classifier(np.array(mag_funs_interpolated), np.array(y), precomputed=False, groups=np.array(n_nodes))
        else:
            test_accs = knn_classifier(np.array(mag_funs_interpolated), np.array(y), precomputed=False)
        
        res_dict[f'divcurves_{metric}']['acc_mean'] = np.mean(test_accs)
        res_dict[f'divcurves_{metric}']['acc_std'] = np.std(test_accs)


        # doing PCA with Euclidean distance between the diversity curves
        pca_embedding = PCA(n_components=2)
        X_pca = pca_embedding.fit_transform(np.array(mag_funs_interpolated))


        # computing clustering scores and accuracies on transformed data from PCA
        if n_graphs > 1:
            test_accs = knn_classifier(X_pca, np.array(y), precomputed=False, groups=np.array(n_nodes))
        else:
            test_accs = knn_classifier(X_pca, np.array(y), precomputed=False)
        
        res_dict[f'divcurves_pca_{metric}']['acc_mean'] = np.mean(test_accs)
        res_dict[f'divcurves_pca_{metric}']['acc_std'] = np.std(test_accs)
        all_clustering_results_euclidean(f'divcurves_pca_{metric}', res_dict, X_pca, y)

        # creating plots
        fig = sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=hue_names, palette='tab10')
        fig.legend(
            frameon=False, 
            loc=(1.04, 0),
            bbox_to_anchor=(1, 0.5)
            )
        sns.despine()
        plt.savefig(results_dir + f"divcurves_pca_{metric}.pdf", dpi=300, format='pdf', bbox_inches="tight")
        plt.savefig(results_dir + f"divcurves_pca_{metric}.png", dpi=300, format='png', bbox_inches="tight")
        plt.close()

        # ####################################################
        # ## Computing diversity baseline, spread functinon ##
        # ####################################################

        # mags, _ = get_mag_funs_graphs(G_nx, dist_fn, ts=np.linspace(1,5,20), method="spread", get_weights=False)
        # mag_funs = np.vstack(mags)

        # if n_graphs > 1:
        #     test_accs = knn_classifier(mag_funs, np.array(y), precomputed=False, groups=np.array(n_nodes))
        # else:
        #     test_accs = knn_classifier(mag_funs, np.array(y), precomputed=False)

        # res_dict[f'div_function_{metric}']['acc_mean'] = np.mean(test_accs)
        # res_dict[f'div_function_{metric}']['acc_std'] = np.std(test_accs)

        # all_clustering_results_euclidean(f'div_function_{metric}', res_dict, mag_funs, y)


    ###########################
    ### computing baselines ###
    ###########################


    struct_baselines = {
        'nnodes': lambda G: G.number_of_nodes(),
        'nedges': lambda G: G.number_of_edges(),
        'density': lambda G: nx.density(G),
        'nconn_components': lambda G: nx.number_connected_components(G),
    }

    # for collecting a summary statistic vector
    sum_stat = []

    # one at a time
    for n_baseline in struct_baselines:
        array = np.reshape(np.array([struct_baselines[n_baseline](G_nx[i]) for i in range(len(G_nx))]), (-1,1))

        res_dict[f'structure_baseline_{n_baseline}'] = {}
        res_dict[f'structure_baseline_{n_baseline}'] = {}
        res_dict[f'structure_baseline_{n_baseline}'] = {}

        all_clustering_results_euclidean(f'structure_baseline_{n_baseline}', res_dict, array, y) 
        
        if n_graphs > 1:
            test_accs = knn_classifier(array, np.array(y), precomputed=False, groups=np.array(n_nodes))
        else:
            test_accs = knn_classifier(array, np.array(y), precomputed=False)
        res_dict[f'structure_baseline_{n_baseline}']['acc_mean'] = np.mean(test_accs)
        res_dict[f'structure_baseline_{n_baseline}']['acc_std'] = np.std(test_accs)

        sum_stat.append(array)
    
    # summary statistic
    array_sum_stat = np.concatenate([a for a in sum_stat], axis=1)
    
    res_dict[f'structure_baseline_summary'] = {}
    res_dict[f'structure_baseline_summary'] = {}
    res_dict[f'structure_baseline_summary'] = {}

    all_clustering_results_euclidean(f'structure_baseline_summary', res_dict, array_sum_stat, y) 
    
    if n_graphs > 1:
        test_accs = knn_classifier(array_sum_stat, np.array(y), precomputed=False, groups=np.array(n_nodes))
    else:
        test_accs = knn_classifier(array_sum_stat, np.array(y), precomputed=False)
    res_dict[f'structure_baseline_summary']['acc_mean'] = np.mean(test_accs)
    res_dict[f'structure_baseline_summary']['acc_std'] = np.std(test_accs)

    
    #creating dataframes and saving results
    res_df = pd.DataFrame.from_dict(res_dict, orient='index')

    df_n_nodes = pd.DataFrame({"n_nodes": n_nodes})
    res_df.to_csv(f"{results_dir}/res_df.csv")
    df_n_nodes.to_csv(f"{results_dir}/n_nodes.csv")