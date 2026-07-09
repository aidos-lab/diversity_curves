import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import random
import argparse
from karateclub import NetLSD, Graph2Vec, FeatherGraph
from grakel import GraphKernel
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
    argparser.add_argument("--output_dir", type=str, default="./simulations/results/sim_params/", help="Output directory for results")
    argparser.add_argument('--models', type=str, nargs='*', default=None, help='select the models to be computed, has to be a subset of the models specifies in models_configs')
    argparser.add_argument('--ngraphs', type=int, default=3, help='Number of graphs that are generated per cardinality')
    argparser.add_argument('--nrange', type=int, nargs=2, default=[10,30], help='specify the range of cardinalities of the generated graphs')
    argparser.add_argument('--nsteps', type=int, default=1, help='Specify the stepsize for the cardinalities of the generated graphs')

    args = argparser.parse_args()
    results_dir = args.output_dir
    n_low = args.nrange[0]
    n_high = args.nrange[1]
    n_steps = args.nsteps
    n_graphs = args.ngraphs

    # setting the parameters as reported in the paper
    models_configs = {
    'ER': {'ER_probs': [0.1, 0.5, 0.9], 
        'ER_nps': None}, 
    'SBM': {
        'SBM_probs': [0.6, 0.01], 
        'SBM_blocks': [[1/2], [1/3, 1/3], [1/4, 1/4, 1/4]] 
        },
    'RG': {
        'RG_probs': [0.25, 0.5, 0.75]
    },
    'RPG':{
        'RPG_probs': [[0.9, 0.01], [0.9, 0.1], [0.9, 0.5]],
        'RPG_props': [1/3, 1/3] 
    },
    }


    if args.models is None:
        models = models = list(models_configs.keys())
    else:
        models = args.models

    # choice of graph metric we use for the spread computation
    graph_metrics = ['shortest_path_distance']

    scores = ['acc_mean', 'silh_mean', 'ari_agglo_mean', 'db_mean',	'ch_mean', 'ari_kmeans_mean']
    scores_asc = {'acc_mean': False, 'silh_mean': False, 'ari_agglo_mean': False, 'db_mean': True,'ch_mean': False, 'ari_kmeans_mean': False}

    # results_dir = f'{output_dir}/nodes_{n_low}_to_{n_high}_stepsize_{n_steps}/n_graphs_{n_graphs}/'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    
    all_configs = {}
    all_results = {}

    for model in models:
        model_results = {}
        scores_dfs = []
        model_params = models_configs[model].keys()
        config = [models_configs[model][para] for para in model_params]
        model_kwargs = models_configs[model]

        # setting parameters and folder for plots
        sns.set_style('white')
        matplotlib.rcParams.update({'font.size': 18})
        if not os.path.exists(results_dir + f'{model}/'):
            os.makedirs(results_dir + f'{model}/')

        # generate graphs
        G_nx, Gs, y, y_probs, n_nodes = simulate_graphs_kwargs(
                model = model, n_graphs=n_graphs,
                nodes_range=range(n_low,n_high, n_steps),
                **model_kwargs,
                )
        
        # labels for PCA plots
        match model:
            case 'ER':
                hue_names = {str(i): f'$p={p}$' for i, p in enumerate(model_kwargs['ER_probs'])}
            case 'RG':
                hue_names = {str(i): f'$r={r}$' for i, r in enumerate(model_kwargs['RG_probs'])}
            case 'SBM':
                hue_names = {
                    str(0): '$2$ communities',
                    str(1): '$3$ communities',
                    str(2): '$4$ communities'}
            case 'RPG':
                hue_names = {str(i): f'$p_{{in}}={p[0]}, p_{{out}}={p[1]}$' for i, p in enumerate(model_kwargs['RPG_probs'])}
        
        compute_at_cardinality = None
        upsample_to = max(n_nodes) 
        downsample_to = None
        
        ##################################
        ### computing diversity curves ###
        ##################################
    
        for metric in graph_metrics:

            model_results[f'divcurves_{metric}'] = {}
            model_results[f'divcurves_pca_{metric}'] = {}
            model_results[f'div_function_{metric}'] = {}

            print(f"Computing MAG with metric: {metric}")

            def dist_fn(G):
                return get_dist(G=G, X=None, X2=None, metric=metric, mode="structure")
            
            all_spreads, weights_lists = get_mag_graphs([g.copy() for g in G_nx], dist_fn, ts=[1], upsample_to=upsample_to, downsample_to=downsample_to, n_repeats=3, scores_method='random', compute_at_cardinality=compute_at_cardinality)

            mag_funs = get_funs_df(all_spreads)
            mag_funs_interpolated = mag_funs.interpolate(method='linear', axis='columns')

            # compute clustering metrics based on euclidean distance of diversity curve vectors 
            ari_kmeans_mean, ari_kmeans_std = compute_rand_kmeans(np.array(mag_funs_interpolated), y)
            ari_agglo_mean, ari_agglo_std = compute_rand_agglo(np.array(mag_funs_interpolated), y, metric='euclidean')

            means, stds = compute_clustering_scores_resample(np.array(mag_funs_interpolated), y)

            model_results[f'divcurves_{metric}']['silh_mean'], model_results[f'divcurves_{metric}']['db_mean'], model_results[f'divcurves_{metric}']['ch_mean'] = means
            model_results[f'divcurves_{metric}']['silh_std'], model_results[f'divcurves_{metric}']['db_std'], model_results[f'divcurves_{metric}']['ch_std'] = stds
            model_results[f'divcurves_{metric}']['ari_agglo_mean'] = ari_agglo_mean
            model_results[f'divcurves_{metric}']['ari_agglo_std'] = ari_agglo_std
            model_results[f'divcurves_{metric}']['ari_kmeans_mean'] = ari_kmeans_mean
            model_results[f'divcurves_{metric}']['ari_kmeans_std'] = ari_kmeans_std

            # computing accuracies
            if n_graphs > 1:
                test_accs = knn_classifier(np.array(mag_funs_interpolated), np.array(y), precomputed=False, groups=np.array(n_nodes))
            else:
                test_accs = knn_classifier(np.array(mag_funs_interpolated), np.array(y), precomputed=False)
            
            model_results["divcurves_"+metric]['acc_mean'] = np.mean(test_accs)
            model_results["divcurves_"+metric]['acc_std'] = np.std(test_accs)

            # doing PCA with Euclidean distance between the diversity curves
            pca_embedding = PCA(n_components=2)
            X_pca = pca_embedding.fit_transform(np.array(mag_funs_interpolated))
            
            # computing clustering scores and accuracies on transformed data from PCA
            all_clustering_results_euclidean(f'divcurves_pca_{metric}', model_results, X_pca, y)
            if n_graphs > 1:
                test_accs = knn_classifier(X_pca, np.array(y), precomputed=False, groups=np.array(n_nodes))
            else:
                test_accs = knn_classifier(X_pca, np.array(y), precomputed=False)
            
            model_results[f'divcurves_pca_{metric}']['acc_mean'] = np.mean(test_accs)
            model_results[f'divcurves_pca_{metric}']['acc_std'] = np.std(test_accs)

            #creating PCA plots
            fig = sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='tab10')
            handles, labels = fig.get_legend_handles_labels()
            fig.legend(
                handles=handles,
                labels= [hue_names[old_label] for old_label in labels],
                frameon=False,
                loc = (1.04, 0),
                bbox_to_anchor=(1, 0.5),
            )
            sns.despine()
            plt.savefig(results_dir + f"{model}/divcurves_pca_{metric}.pdf", dpi=300, format='pdf', bbox_inches="tight")
            plt.savefig(results_dir + f"{model}/divcurves_pca_{metric}.png", dpi=300, format='png', bbox_inches="tight")
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

            # model_results[f'div_function_{metric}']['acc_mean'] = np.mean(test_accs)
            # model_results[f'div_function_{metric}']['acc_std'] = np.std(test_accs)

            # all_clustering_results_euclidean(f'div_function_{metric}', model_results, mag_funs, y)

        
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

            model_results[f'structure_baseline_{n_baseline}'] = {}
            model_results[f'structure_baseline_{n_baseline}'] = {}
            model_results[f'structure_baseline_{n_baseline}'] = {}

            all_clustering_results_euclidean(f'structure_baseline_{n_baseline}', model_results, array, y) 
            
            if n_graphs > 1:
                test_accs = knn_classifier(array, np.array(y), precomputed=False, groups=np.array(n_nodes))
            else:
                test_accs = knn_classifier(array, np.array(y), precomputed=False)
            model_results[f'structure_baseline_{n_baseline}']['acc_mean'] = np.mean(test_accs)
            model_results[f'structure_baseline_{n_baseline}']['acc_std'] = np.std(test_accs)

            sum_stat.append(array)
        
        # summary statistic
        array_sum_stat = np.concatenate([a for a in sum_stat], axis=1)
        
        model_results[f'structure_baseline_summary'] = {}
        model_results[f'structure_baseline_summary'] = {}
        model_results[f'structure_baseline_summary'] = {}

        all_clustering_results_euclidean(f'structure_baseline_summary', model_results, array_sum_stat, y) 
        
        if n_graphs > 1:
            test_accs = knn_classifier(array_sum_stat, np.array(y), precomputed=False, groups=np.array(n_nodes))
        else:
            test_accs = knn_classifier(array_sum_stat, np.array(y), precomputed=False)
        model_results[f'structure_baseline_summary']['acc_mean'] = np.mean(test_accs)
        model_results[f'structure_baseline_summary']['acc_std'] = np.std(test_accs)


        # storing results of model
        res_df = pd.DataFrame.from_dict(model_results, orient='index')
        all_results[model] = res_df
        all_configs[model] = str(config)

    # combine results of all models
    res_combined = pd.concat([all_results[key] for key in all_results.keys()], axis=1, keys=all_results.keys())
 
    # save results
    df_n_nodes = pd.DataFrame({"n_nodes": n_nodes})
    df_n_nodes.to_csv(f"{results_dir}/n_nodes.csv")

    res_combined.to_csv(f"{results_dir}/results_combined.csv")

    df_best_configs = pd.DataFrame.from_dict(all_configs, orient='index', columns=['configurations'])
    df_best_configs.to_csv(f'{results_dir}/configs.csv')



