import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import networkx as nx
import argparse
import itertools
from time import time
import os
import argparse
from karateclub import NetLSD, Graph2Vec, FeatherGraph
from grakel import GraphKernel
import random

import toper

from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from magnipy.magnitude.distances import get_dist
from diversity_curves.process import get_mag_graphs, get_weight_diffs, get_funs_diff, get_funs_df

from diversity_curves.simulate import simulate_graphs_kwargs
from diversity_curves.classifiers import knn_classifier
from diversity_curves.clustering_performance import compute_clustering_scores, compute_clustering_scores_resample, compute_silhouette_score_resamples, compute_rand_agglo, compute_rand_kmeans
from diversity_curves.baselines.spectral_zoo import est_moment

from karateclub import NetLSD, Graph2Vec, FeatherGraph

def all_clustering_results_euclidean(method, df, X, y):
    ari_kmeans_mean, ari_kmeans_std = compute_rand_kmeans(X, y)
    ari_agglo_mean, ari_agglo_std = compute_rand_agglo(X, y, metric='euclidean')

    means, stds = compute_clustering_scores_resample(X, y)

    df[method]['silh_mean'], df[method]['db_mean'], df[method]['ch_mean'] = means
    df[method]['silh_std'], df[method]['db_std'], df[method]['ch_std'] = stds
    df[method]['ari_agglo_mean'] = ari_agglo_mean
    df[method]['ari_agglo_std'] = ari_agglo_std
    df[method]['ari_kmeans_mean'] = ari_kmeans_mean
    df[method]['ari_kmeans_std'] = ari_kmeans_std

    return

if __name__ == '__main__':
    np.random.seed(43)
    random.seed(314)
    
### now only one config per model allowed!
    models_configs = {
    'ER': {'probs': [0.1, 0.5, 0.9], #[0.5, 0.75, 0.9] [0.25, 0.5, 0.75]
        'nps': None}, # , [0.5, 1, 2]
    # 'BA': {
    #     'ms': [[3,4,5], [2,5,8]], #[3,4,5]
    #     'initial': [None, '2star', '2star_edge'] #None,  '2star', '2star_edge'
    #     },
    
    'SBM': {
        'probs': [0.6, 0.01], #[0.6, 0.01], [0.9, 0.01], [0.75, 0.01], [0.5, 0.01]
        'blocks': [[1/2], [1/3, 1/3], [1/4, 1/4, 1/4]] #[[1/2], [1/4], [1/16]], [[1/2], [1/3, 1/3], [1/4, 1/4, 1/4]]
        },
    # 'NWS': {
    #     'probs': [[0.01, 0.25, 0.9], [0.1, 0.5, 0.9], [0.01, 0.25, 0.75]], # [0.1, 0.5, 0.9], [0.01, 0.25, 0.75]
    #     'k': [2, 4, 8] #, 8 , 6
    #     },
    'RG': {
        'probs': [0.25, 0.5, 0.75]
    },
    'RPG':{
        'probs': [[0.9, 0.01], [0.9, 0.1], [0.9, 0.5]], #[[0.75, 0.1], [0.5, 0.1], [0.25, 0.1]], [[0.75, 0.05], [0.5, 0.05], [0.25, 0.05]], [[0.9, 0.1], [0.5, 0.1], [0.1, 0.1]],  [[0.9, 0.05], [0.5, 0.05], [0.25, 0.05]]
        'props': [1/3, 1/3] #, [1/2]],
    },

    # 'RL': {
    #     'probs': [[[0.5, 0.5], [0.5, 0], [0.5, 0.9]]],
    #     'node_var': [2, 5],
    # },
    # 'PLC': {
    #     'probs': [[0.25, 0.5, 0.75]],
    #     'm': [2]
    # },
    # 'W': {
    #     'alphas':
    #     'beta':
    # },
    # 'WS': {
    #     'probs': [[0.01, 0.25, 0.9]], # [0.01, 0.1, 0.25, 0.5, 0.75] [0.01, 0.25, 0.5]
    #     'k': [2, 4] #, 8 , 6
    # },
    # 'CWS': {
    #     'probs': [[0.01, 0.25, 0.9]], # [0.01, 0.1, 0.25, 0.5, 0.75] [0.01, 0.25, 0.5]
    #     'k': [2, 4] #, 8 , 6
    # },
    }

    models = list(models_configs.keys())

    scores = ['acc_mean', 'silh_mean', 'ari_agglo_mean', 'db_mean',	'ch_mean', 'ari_kmeans_mean']
    scores_asc = {'acc_mean': False, 'silh_mean': False, 'ari_agglo_mean': False, 'db_mean': True,'ch_mean': False, 'ari_kmeans_mean': False}

    graph_metrics = ['shortest_path_distance', "diffusion_distance", "resistance_distance"]
    score_methods = ['full', 'random', 'approx_all']


    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output_dir", type=str, default="./results/metric_score/params", help="Output directory for results")
    argparser.add_argument('--ngraphs', type=int, default=1, help='Number of graphs that are generated per cardinality')
    argparser.add_argument('--nrange', type=int, nargs=2, default=[10,30], help='specify the range of number of nodes of the graphs')
    argparser.add_argument('--nsteps', type=int, default=1, help='stepsize for node range generation')
    argparser.add_argument('--with_regression', type=bool, nargs=1, default=False, help='Whether to do regression on the parameters')

    args = argparser.parse_args()
    output_dir = args.output_dir
    n_low = args.nrange[0]
    n_high = args.nrange[1]
    n_steps = args.nsteps
    n_graphs = args.ngraphs
    with_regression = args.with_regression

    results_dir = f'{output_dir}/combined/{models}/nodes_{n_low}_to_{n_high}_stepsize_{n_steps}/'
    if n_graphs > 1:
        results_dir += f'n_graphs_{n_graphs}/'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    configs = {}
    res_dfs = {}

    for model in models:
        model_results = {}
        # scores_dfs = []

        model_params = models_configs[model].keys()
        
        model_kwargs = {f'{model}_{para}': models_configs[model][para] for para in model_params}
        config = [models_configs[model][para] for para in model_params]
        print(model_kwargs)

        G_nx, Gs, y, y_probs, n_nodes = simulate_graphs_kwargs(
                model = model, n_graphs=n_graphs,
                nodes_range=range(n_low,n_high, n_steps),
                **model_kwargs,
                )

        # TODO
        # compute_at_cardinality = list(range(n_low, (n_low+n_high)//2))
        compute_at_cardinality = None
        upsample_to = max(n_nodes) #n_high
        # upsample_to = (n_low+n_high)//2
        downsample_to = None

        ##################################
        ### computing diversity curves ###
        ##################################
    
        for metric in graph_metrics:
            for score_method in score_methods:

                model_results[f'divcurves_funs_diffs_{metric}_scores_{score_method}'] = {}
                model_results[f'divcurves_{metric}_scores_{score_method}'] = {}
                model_results[f'div_one_scale_{metric}_scores_{score_method}'] = {}
                model_results[f'divcurves_mds_funs_diffs_{metric}_scores_{score_method}'] = {}
                model_results[f'divcurves_mds_{metric}_scores_{score_method}'] = {}
                model_results[f'divcurves_pca_funs_diffs_{metric}_scores_{score_method}'] = {}
                model_results[f'divcurves_pca_{metric}_scores_{score_method}'] = {}

                print(f"Computing MAG with metric: {metric}, score method: {score_method}")

                def dist_fn(G):
                    return get_dist(G=G, X=None, X2=None, metric=metric, mode="structure")#"euclidean", mode="attributes") # structure # full
                
                all_spreads, weights_lists = get_mag_graphs(
                    [g.copy() for g in G_nx], 
                    dist_fn,
                    ts=[1],
                    upsample_to=upsample_to,
                    downsample_to=downsample_to,
                    n_repeats=3,
                    scores_method=score_method,
                    mode='structure',
                    set_seed=False
                )

                mag_funs = get_funs_df(all_spreads)
                mag_funs_interpolated = mag_funs.interpolate(method='linear', axis='columns')
                funs_diffs = get_funs_diff(all_spreads)

                # compute clustering metrics with precomputed difference based on area between curves
                ari_agglo_mean, ari_agglo_std = compute_rand_agglo(funs_diffs, y, metric='precomputed')
                silh_mean, silh_std = compute_silhouette_score_resamples(funs_diffs, y)

                model_results[f'divcurves_funs_diffs_{metric}_scores_{score_method}']['silh_mean'] = silh_mean
                model_results[f'divcurves_funs_diffs_{metric}_scores_{score_method}']['silh_std'] = silh_std
                model_results[f'divcurves_funs_diffs_{metric}_scores_{score_method}']['ari_agglo_mean'] = ari_agglo_mean
                model_results[f'divcurves_funs_diffs_{metric}_scores_{score_method}']['ari_agglo_std'] = ari_agglo_std

                # compute clustering metrics based on euclidean distance of diversity curve vectors 
                ari_kmeans_mean, ari_kmeans_std = compute_rand_kmeans(np.array(mag_funs_interpolated), y)
                ari_agglo_mean, ari_agglo_std = compute_rand_agglo(np.array(mag_funs_interpolated), y, metric='euclidean')

                means, stds = compute_clustering_scores_resample(np.array(mag_funs_interpolated), y)

                model_results[f'divcurves_{metric}_scores_{score_method}']['silh_mean'], model_results[f'divcurves_{metric}_scores_{score_method}']['db_mean'], model_results[f'divcurves_{metric}_scores_{score_method}']['ch_mean'] = means
                model_results[f'divcurves_{metric}_scores_{score_method}']['silh_std'], model_results[f'divcurves_{metric}_scores_{score_method}']['db_std'], model_results[f'divcurves_{metric}_scores_{score_method}']['ch_std'] = stds
                model_results[f'divcurves_{metric}_scores_{score_method}']['ari_agglo_mean'] = ari_agglo_mean
                model_results[f'divcurves_{metric}_scores_{score_method}']['ari_agglo_std'] = ari_agglo_std
                model_results[f'divcurves_{metric}_scores_{score_method}']['ari_kmeans_mean'] = ari_kmeans_mean
                model_results[f'divcurves_{metric}_scores_{score_method}']['ari_kmeans_std'] = ari_kmeans_std

                # computing accuracies
                if n_graphs > 1:
                    test_accs = knn_classifier(funs_diffs, np.array(y), precomputed=True, groups=np.array(n_nodes))
                else:
                    test_accs = knn_classifier(funs_diffs, np.array(y), precomputed=True)
                
                model_results[f'divcurves_funs_diffs_{metric}_scores_{score_method}']['acc_mean'] = np.mean(test_accs)
                model_results[f'divcurves_funs_diffs_{metric}_scores_{score_method}']['acc_std'] = np.std(test_accs)

                if n_graphs > 1:
                    test_accs = knn_classifier(np.array(mag_funs_interpolated), np.array(y), precomputed=False, groups=np.array(n_nodes))
                else:
                    test_accs = knn_classifier(np.array(mag_funs_interpolated), np.array(y), precomputed=False)
                
                model_results[f'divcurves_{metric}_scores_{score_method}']['acc_mean'] = np.mean(test_accs)
                model_results[f'divcurves_{metric}_scores_{score_method}']['acc_std'] = np.std(test_accs)


                # MDS
                # funs_diff
                embedding = MDS(n_components=2, dissimilarity="precomputed")
                X_transformed = embedding.fit_transform(funs_diffs)

                all_clustering_results_euclidean(f'divcurves_mds_funs_diffs_{metric}_scores_{score_method}', model_results, X_transformed, y)
                
                # sns.set_style('white')
                # if not os.path.exists(results_dir + f'{model}/{config}/'):
                #     os.makedirs(results_dir + f'{model}/{config}/')

                # sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=y, palette='tab10')
                # sns.despine()
                # plt.savefig(results_dir + f"{model}/{config}/divcurves_mds_funs_diffs_{metric}_scores_{score_method}.pdf", dpi=300, format='pdf')
                # plt.savefig(results_dir + f"{model}/{config}/divcurves_mds_funs_diffs_{metric}_scores_{score_method}.png", dpi=300, format='png')
                # plt.close()

                # sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=n_nodes)
                # sns.despine()
                # plt.savefig(results_dir + f"{model}/{config}/divcurves_mds_funs_diffs_n_nodes_{metric}_scores_{score_method}.pdf", dpi=300, format='pdf')
                # plt.savefig(results_dir + f"{model}/{config}/divcurves_mds_funs_diffs_n_nodes_{metric}_scores_{score_method}.png", dpi=300, format='png')
                # plt.close()

                # MDS
                # euclidean distance
                embedding = MDS(n_components=2, dissimilarity="euclidean")
                X_transformed = embedding.fit_transform(np.array(mag_funs_interpolated))

                all_clustering_results_euclidean(f'divcurves_mds_{metric}_scores_{score_method}', model_results, X_transformed, y)
                # sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=y, palette='tab10')
                # sns.despine()
                # plt.savefig(results_dir + f"{model}/{config}/divcurves_mds_{metric}.pdf", dpi=300, format='pdf')
                # plt.savefig(results_dir + f"{model}/{config}/divcurves_mds_{metric}.png", dpi=300, format='png')
                # plt.close()

                # sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=n_nodes)
                # sns.despine()
                # plt.savefig(results_dir + f"{model}/{config}/divcurves_mds_n_nodes_{metric}.pdf", dpi=300, format='pdf')
                # plt.savefig(results_dir + f"{model}/{config}/divcurves_mds_n_nodes_{metric}.png", dpi=300, format='png')
                # plt.close()

                # PCA
                # euclidean
                pca_embedding = PCA(n_components=2)
                X_pca = pca_embedding.fit_transform(np.array(mag_funs_interpolated))

                all_clustering_results_euclidean(f'divcurves_pca_funs_diffs_{metric}_scores_{score_method}', model_results, X_pca, y)
                # sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='tab10')
                # sns.despine()
                # plt.savefig(results_dir + f"{model}/{config}/divcurves_pca_funs_diffs_{metric}.pdf", dpi=300, format='pdf')
                # plt.savefig(results_dir + f"{model}/{config}/divcurves_pca_funs_diffs_{metric}.png", dpi=300, format='png')
                # plt.close()

                # sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=n_nodes)
                # sns.despine()
                # plt.savefig(results_dir + f"{model}/{config}/divcurves_pca_funs_diffs_n_nodes{metric}.pdf", dpi=300, format='pdf')
                # plt.savefig(results_dir + f"{model}/{config}/divcurves_pca_funs_diffs_n_nodes{metric}.png", dpi=300, format='png')
                # plt.close()



        # save in big dictionary
        configs[model] = str(config)
        model_results_df = pd.DataFrame.from_dict(model_results, orient='index')
        res_dfs[model]= model_results_df

    res_combined = pd.concat([res_dfs[key] for key in res_dfs.keys()], axis=1, keys=res_dfs.keys())
 
    # save results
    df_n_nodes = pd.DataFrame({"n_nodes": n_nodes})
    df_n_nodes.to_csv(f"{results_dir}/n_nodes.csv")

    res_combined.to_csv(f"{results_dir}/results_combined.csv")

    df_configs = pd.DataFrame.from_dict(configs, orient='index', columns=['configurations'])
    df_configs.to_csv(f'{results_dir}/configs.csv')



