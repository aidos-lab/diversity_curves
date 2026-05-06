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
import random
import argparse
from karateclub import NetLSD, Graph2Vec, FeatherGraph
from grakel import GraphKernel

import toper

from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from magnipy.magnitude.distances import get_dist
from diversity_curves.process import get_mag_graphs, get_weight_diffs, get_funs_diff, get_funs_df, get_mag_funs_graphs

from diversity_curves.simulate import simulate_graphs_kwargs
from diversity_curves.classifiers import knn_classifier
from diversity_curves.clustering_performance import compute_clustering_scores, compute_clustering_scores_resample, compute_silhouette_score_resamples, compute_rand_agglo, compute_rand_kmeans, all_clustering_results_euclidean
from diversity_curves.baselines.spectral_zoo import est_moment
# from diversity_curves.baselines.ph import obtain_input_from_graphs

from karateclub import NetLSD, Graph2Vec, FeatherGraph



if __name__ == '__main__':
    np.random.seed(43)
    random.seed(314)
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output_dir", type=str, default="./simulations/results/", help="Output directory for results")
    argparser.add_argument('--models', type=str, nargs='*', default=None, help='select the models to be computed')
    argparser.add_argument('--ngraphs', type=int, default=1, help='Number of graphs that are generated per cardinality')
    argparser.add_argument('--nrange', type=int, nargs=2, default=[10,30], help='specify the range of number of nodes of the graphs')
    argparser.add_argument('--nsteps', type=int, default=1, help='stepsize for node range generation')

    args = argparser.parse_args()
    output_dir = args.output_dir
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

    scores = ['acc_mean', 'silh_mean', 'ari_agglo_mean', 'db_mean',	'ch_mean', 'ari_kmeans_mean']
    scores_asc = {'acc_mean': False, 'silh_mean': False, 'ari_agglo_mean': False, 'db_mean': True,'ch_mean': False, 'ari_kmeans_mean': False}

    graph_metrics = ['shortest_path_distance']

    if args.models is None:
        models = models = list(models_configs.keys())
    else:
        models = args.models

    results_dir = f'{output_dir}/sim_params/nodes_{n_low}_to_{n_high}_stepsize_{n_steps}/n_graphs_{n_graphs}/'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # best_configs = {}
    # best_res_dfs = {}
    # legend_title = {
    #     'ER': None,
    #     'RG': None,
    #     'SBM': 'Number of communities',
    #     'RPG': None,
    #     'DBA': None,
    # }
    all_configs = {}
    all_results = {}

    for model in models:
        model_results = {}
        scores_dfs = []
        # res_dfs = []
        # best_median_rank = []
        # best_min_rank = []

        model_params = models_configs[model].keys()
        # model_configs = list(itertools.product(*(models_configs[model][para] for para in model_params)))

        # model_kwargs = {f'{model}_{para}': config[i] for i, para in enumerate(model_params)}

        # for config in model_configs:
        # model_results = {}
        config = [models_configs[model][para] for para in model_params]
        
        model_kwargs = models_configs[model]

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
        

        # compute_at_cardinality = list(range(n_low, (n_low+n_high)//2))
        compute_at_cardinality = None
        upsample_to = max(n_nodes) #n_high
        # upsample_to = (n_low+n_high)//2
        downsample_to = None

        ##################################
        ### computing diversity curves ###
        ##################################
    
        for metric in graph_metrics:

            # model_results[f'divcurves_funs_diffs_{metric}'] = {}
            model_results[f'divcurves_{metric}'] = {}
            # model_results[f'divcurves_mds_funs_diffs_{metric}'] = {}
            # model_results[f'divcurves_mds_{metric}'] = {}
            # model_results[f'divcurves_pca_funs_diffs_{metric}'] = {}
            model_results[f'divcurves_pca_{metric}'] = {}
            # model_results[f'div_one_scale_{metric}'] = {}
            model_results[f'div_function_{metric}'] = {}

            print(f"Computing MAG with metric: {metric}")

            def dist_fn(G):
                return get_dist(G=G, X=None, X2=None, metric=metric, mode="structure")#"euclidean", mode="attributes") # structure # full
            
            all_spreads, weights_lists = get_mag_graphs([g.copy() for g in G_nx], dist_fn, ts=[1], upsample_to=upsample_to, downsample_to=downsample_to, n_repeats=3, scores_method='random', compute_at_cardinality=compute_at_cardinality)

            mag_funs = get_funs_df(all_spreads)
            mag_funs_interpolated = mag_funs.interpolate(method='linear', axis='columns')
            # funs_diffs = get_funs_diff(all_spreads)

            # # compute clustering metrics with precomputed difference based on area between curves
            # ari_agglo_mean, ari_agglo_std = compute_rand_agglo(funs_diffs, y, metric='precomputed')
            # silh_mean, silh_std = compute_silhouette_score_resamples(funs_diffs, y)

            # model_results[f'divcurves_funs_diffs_{metric}']['silh_mean'] = silh_mean
            # model_results[f'divcurves_funs_diffs_{metric}']['silh_std'] = silh_std
            # model_results[f'divcurves_funs_diffs_{metric}']['ari_agglo_mean'] = ari_agglo_mean
            # model_results[f'divcurves_funs_diffs_{metric}']['ari_agglo_std'] = ari_agglo_std

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

            # # computing accuracies
            # if n_graphs > 1:
            #     test_accs = knn_classifier(funs_diffs, np.array(y), precomputed=True, groups=np.array(n_nodes))
            # else:
            #     test_accs = knn_classifier(funs_diffs, np.array(y), precomputed=True)
            
            # model_results["divcurves_funs_diffs_"+metric]['acc_mean'] = np.mean(test_accs)
            # model_results["divcurves_funs_diffs_"+metric]['acc_std'] = np.std(test_accs)

            if n_graphs > 1:
                test_accs = knn_classifier(np.array(mag_funs_interpolated), np.array(y), precomputed=False, groups=np.array(n_nodes))
            else:
                test_accs = knn_classifier(np.array(mag_funs_interpolated), np.array(y), precomputed=False)
            
            model_results["divcurves_"+metric]['acc_mean'] = np.mean(test_accs)
            model_results["divcurves_"+metric]['acc_std'] = np.std(test_accs)


            # # MDS
            # # funs_diff
            # embedding = MDS(n_components=2, dissimilarity="precomputed")
            # X_transformed = embedding.fit_transform(funs_diffs)

            # all_clustering_results_euclidean(f'divcurves_mds_funs_diffs_{metric}', model_results, X_transformed, y)

            # if n_graphs > 1:
            #     test_accs = knn_classifier(X_transformed, np.array(y), precomputed=False, groups=np.array(n_nodes))
            # else:
            #     test_accs = knn_classifier(X_transformed, np.array(y), precomputed=False)
            
            # model_results["divcurves_mds_funs_diffs_"+metric]['acc_mean'] = np.mean(test_accs)
            # model_results["divcurves_mds_funs_diffs_"+metric]['acc_std'] = np.std(test_accs)
            
            sns.set_style('white')
            if not os.path.exists(results_dir + f'{model}/{str(config)}/'):
                os.makedirs(results_dir + f'{model}/{str(config)}/')

            # sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=y, palette='tab10')
            # sns.despine()
            # plt.savefig(results_dir + f"{model}/{config}/divcurves_mds_funs_diffs_{metric}.pdf", dpi=300, format='pdf')
            # plt.savefig(results_dir + f"{model}/{config}/divcurves_mds_funs_diffs_{metric}.png", dpi=300, format='png')
            # plt.close()

            # sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=n_nodes)
            # sns.despine()
            # plt.savefig(results_dir + f"{model}/{config}/divcurves_mds_funs_diffs_n_nodes_{metric}.pdf", dpi=300, format='pdf')
            # plt.savefig(results_dir + f"{model}/{config}/divcurves_mds_funs_diffs_n_nodes_{metric}.png", dpi=300, format='png')
            # plt.close()

            # # MDS
            # # euclidean distance
            # embedding = MDS(n_components=2, dissimilarity="euclidean")
            # X_transformed = embedding.fit_transform(np.array(mag_funs_interpolated))

            # all_clustering_results_euclidean(f'divcurves_mds_{metric}', model_results, X_transformed, y)

            # if n_graphs > 1:
            #     test_accs = knn_classifier(X_transformed, np.array(y), precomputed=False, groups=np.array(n_nodes))
            # else:
            #     test_accs = knn_classifier(X_transformed, np.array(y), precomputed=False)
            
            # model_results[f'divcurves_mds_{metric}']['acc_mean'] = np.mean(test_accs)
            # model_results[f'divcurves_mds_{metric}']['acc_std'] = np.std(test_accs)

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

            all_clustering_results_euclidean(f'divcurves_pca_{metric}', model_results, X_pca, y)

            matplotlib.rcParams.update({'font.size': 18})
            sns.set_style('white')
            fig = sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='tab10')
            handles, labels = fig.get_legend_handles_labels()
            fig.legend(
                handles=handles,
                labels= [hue_names[old_label] for old_label in labels],
                frameon=False,
                # title = legend_title[model],
                loc = (1.04, 0),
                bbox_to_anchor=(1, 0.5),
            )
            sns.despine()
            plt.savefig(results_dir + f"{model}/{str(config)}/divcurves_pca_{metric}.pdf", dpi=300, format='pdf', bbox_inches="tight")
            plt.savefig(results_dir + f"{model}/{str(config)}/divcurves_pca_{metric}.png", dpi=300, format='png', bbox_inches="tight")
            plt.close()


            sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=n_nodes)
            sns.despine()
            plt.savefig(results_dir + f"{model}/{config}/divcurves_pca_n_nodes{metric}.pdf", dpi=300, format='pdf')
            plt.savefig(results_dir + f"{model}/{config}/divcurves_pca_n_nodes{metric}.png", dpi=300, format='png')
            plt.close()

            if n_graphs > 1:
                test_accs = knn_classifier(X_pca, np.array(y), precomputed=False, groups=np.array(n_nodes))
            else:
                test_accs = knn_classifier(X_pca, np.array(y), precomputed=False)
            
            model_results[f'divcurves_pca_{metric}']['acc_mean'] = np.mean(test_accs)
            model_results[f'divcurves_pca_{metric}']['acc_std'] = np.std(test_accs)


            # mag baselines
            # # magnitude/spread one scale
            # mags_original = []
            # for gr in all_spreads.graph.unique():
            #     spread_df_sub = all_spreads.loc[all_spreads.graph == gr,:]
            #     n_nodes_value = n_nodes[gr]
            #     this_mag = spread_df_sub.loc[n_nodes_value,:].iloc[0,:]
            #     mags_original.append(this_mag)
            # mags_original = pd.concat(mags_original, axis=1).T
            # mags = mags_original.iloc[:,0].values.reshape(-1, 1)

            # #clustering
            # all_clustering_results_euclidean(f'div_one_scale_{metric}', model_results, mags, y)

            # #accuracies
            # if n_graphs > 1:
            #     test_accs = knn_classifier(mags, np.array(y), precomputed=False, groups=np.array(n_nodes))
            # else:
            #     test_accs = knn_classifier(mags, np.array(y), precomputed=False)
            
            # model_results[f'div_one_scale_{metric}']['acc_mean'] = np.mean(test_accs)
            # model_results[f'div_one_scale_{metric}']['acc_std'] = np.std(test_accs)

            # usual magnitude function

            mags, _ = get_mag_funs_graphs(G_nx, dist_fn, ts=np.linspace(1,5,20), method="spread", get_weights=False)
            mag_funs = np.vstack(mags)

            if n_graphs > 1:
                test_accs = knn_classifier(mag_funs, np.array(y), precomputed=False, groups=np.array(n_nodes))
            else:
                test_accs = knn_classifier(mag_funs, np.array(y), precomputed=False)

            model_results[f'div_function_{metric}']['acc_mean'] = np.mean(test_accs)
            model_results[f'div_function_{metric}']['acc_std'] = np.std(test_accs)

            all_clustering_results_euclidean(f'div_function_{metric}', model_results, mag_funs, y)

            # mag_funs = pd.DataFrame(mag_funs)
            # n_graphs = len(G_nx)
            # diff_matrix_mag = np.zeros((n_graphs, n_graphs))
            # for i in range(n_graphs):
            #     for j in range(i+1, n_graphs):
            #         diff = np.abs(mag_funs.loc[i,:] - mag_funs.loc[j,:])
            #         #diff.dropna(inplace=True)
            #         diff_matrix_mag[i, j] = np.trapz(y=diff.values, x=diff.index)

            # diff_matrix_mag = diff_matrix_mag + diff_matrix_mag.T
        

        ###########################
        ### computing baselines ###
        ###########################

        # graph embeddings
        embeddings = {
            'netLSD': NetLSD(),
            'graph2vec': Graph2Vec(),
            'feathergraph': FeatherGraph()
        }

        for embedding_name in embeddings:
            embedding = embeddings[embedding_name]
            embedding.fit([g.copy() for g in G_nx])
            array = embedding.get_embedding()

            model_results[f'graph_embedding_{embedding_name}'] = {}
            model_results[f'graph_embedding_mds_{embedding_name}'] = {}
            model_results[f'graph_embedding_pca_{embedding_name}'] = {}

            all_clustering_results_euclidean(f'graph_embedding_{embedding_name}', model_results, array, y) 
            
            # sns.scatterplot(x=array[:,0], y=array[:,1], hue=y, palette='tab10') #, palette=cust_cmap
            # sns.despine()
            # #plt.show()
            # plt.savefig(results_dir + f"spectral_zoo.pdf", dpi=300)
            # plt.close()
            if n_graphs > 1:
                test_accs = knn_classifier(array, np.array(y), precomputed=False, groups=np.array(n_nodes))
            else:
                test_accs = knn_classifier(array, np.array(y), precomputed=False)
            model_results[f'graph_embedding_{embedding_name}']['acc_mean'] = np.mean(test_accs)
            model_results[f'graph_embedding_{embedding_name}']['acc_std'] = np.std(test_accs)

            # MDS
            # embedding = MDS(n_components=2, metric="euclidean")
            # X_transformed = embedding.fit_transform(array)

            # all_clustering_results_euclidean(f'graph_embedding_mds_{embedding_name}', model_results, X_transformed, y)
            # sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=y, palette='tab10')
            # sns.despine()
            # plt.savefig(results_dir + f"graph_embedding_mds_{embedding_name}.png", dpi=300)
            # plt.close()

            # sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=n_nodes)
            # sns.despine()
            # plt.savefig(results_dir + f"graph_embedding_mds_{embedding_name}.png", dpi=300)
            # plt.close()
            # PCA
            # pca_embedding = PCA(n_components=2)
            # X_pca = pca_embedding.fit_transform(array)

            # all_clustering_results_euclidean(f'graph_embedding_pca_{embedding_name}', model_results, X_pca, y)
            # sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='tab10')
            # sns.despine()
            # plt.savefig(results_dir + f"graph_embedding_pca_{embedding_name}.png", dpi=300)
            # plt.close()

            # sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=n_nodes)
            # sns.despine()
            # plt.savefig(results_dir + f"graph_embedding_pca_{embedding_name}.png", dpi=300)
            # plt.close()

        # kernels
        kernel_names = [
                    "shortest_path", 
                    #"graphlet_sampling", 
                    "pyramid_match", 
                    #"svm_theta",
                    "neighborhood_hash",
                    # "subtree_wl",
                    # "odd_sth",
                    # "propagation",
                    # "vertex_histogram",
                    "weisfeiler_lehman",
                    "core_framework",
                    "weisfeiler_lehman_optimal_assignment"#,
                    #"edge_histogram",
                    #"multiscale_laplacian"
                    ]

        for k_ in kernel_names:
            # General kernels
            if k_ in ["weisfeiler_lehman" , "core_framework"]:
                gk = GraphKernel(kernel=[{"name": k_}, {"name": "shortest_path"}], normalize=True)
            
            # Base kernels
            else:
                gk = GraphKernel(kernel=[{"name": k_}], normalize=True)

            # Calculate the kernel matrix
            kernel_success = False
            try:
                K = gk.fit_transform(Gs)
                kernel_success = True
            except Exception as e:
                print(f"Kernel {k_} failed with error: {e}")
            if kernel_success:
                model_results[f'kernel_{k_}'] = {}
                model_results[f'kernel_mds_{k_}'] = {}
                # precomputed clustering metrics
                ari_agglo_mean, ari_agglo_std = compute_rand_agglo(-K+K.max(), y, metric='precomputed')
                silh_mean, silh_std = compute_silhouette_score_resamples(-K+K.max(), y)

                model_results[f'kernel_{k_}']['silh_mean'] = silh_mean
                model_results[f'kernel_{k_}']['silh_std'] = silh_std
                model_results[f'kernel_{k_}']['ari_agglo_mean'] = ari_agglo_mean
                model_results[f'kernel_{k_}']['ari_agglo_std'] = ari_agglo_std

                # accuracies
                if n_graphs > 1:
                    test_accs = knn_classifier(-K+K.max(), np.array(y), precomputed=True, groups=np.array(n_nodes))
                else:
                    test_accs = knn_classifier(-K+K.max(), np.array(y), precomputed=True)
                model_results[f'kernel_{k_}']['acc_mean'] = np.mean(test_accs)
                model_results[f'kernel_{k_}']['acc_std'] = np.std(test_accs)

                # MDS
                if (np.var(-K+K.max(), axis=0) == 0).sum() < (-K+K.max()).shape[1]:
                    embedding = MDS(n_components=2, dissimilarity="precomputed")
                    X_transformed = embedding.fit_transform(-K+K.max())
                    all_clustering_results_euclidean(f'kernel_mds_{k_}', model_results, X_transformed, y)

                    # accuracies
                    if n_graphs > 1:
                        test_accs = knn_classifier(X_transformed, np.array(y), precomputed=False, groups=np.array(n_nodes))
                    else:
                        test_accs = knn_classifier(X_transformed, np.array(y), precomputed=False)
                    model_results[f'kernel_mds_{k_}']['acc_mean'] = np.mean(test_accs)
                    model_results[f'kernel_mds_{k_}']['acc_std'] = np.std(test_accs)
                    # sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=y, palette='tab10')
                    # sns.despine()
                    # plt.savefig(results_dir + f"kernel_mds_{k_}.png", dpi=300)
                    # plt.close()

                    # sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=n_nodes)
                    # sns.despine()
                    # plt.savefig(results_dir + f"kernel_mds_nnodes_{k_}.png", dpi=300)
                    # plt.close()
                else:
                    print(f'There are {(np.var(-K+K.max(), axis=0) == 0).sum()} out of {(-K+K.max()).shape[1]} zero variance cols: {-K+K.max()}')


        # topER
        all_sub = []
        for filt in ["degree", "popularity", "fricci", "oricci", "closeness"]:
            F = toper.main(G_nx, [filt])#, num_segm=501) "closeness",
            if filt == "fricci":
                filt = "forricci"
            elif filt == "oricci":
                filt = "olricci"
            array_super = np.array(F[f"{filt}_super"])
            array_sub = np.array(F[f"{filt}_sub"])
            array_sub_super = np.array(np.concatenate([F[f"{filt}_sub"], F[f"{filt}_super"]], axis=1))
            print(f'shape of the toper arrays: {array_sub.shape}, {array_sub_super.shape}, {array_super.shape}')

            if not np.isnan(array_super).any():
                model_results[f'toper_{filt}_super'] = {}

                all_clustering_results_euclidean(f'toper_{filt}_super', model_results, array_super, y)
                
                # sns.scatterplot(x=array_super[:,0], y=array_super[:,1], hue=y, palette='tab10') #, palette=cust_cmap
                # sns.despine()
                # #plt.show()
                # plt.savefig(results_dir + f"toper_{filt}_super.png", dpi=300)
                # plt.close()

                if n_graphs > 1:
                    test_accs = knn_classifier(array_super, np.array(y), precomputed=False, groups=np.array(n_nodes))
                else:
                    test_accs = knn_classifier(array_super, np.array(y), precomputed=False)
                model_results[f'toper_{filt}_super']['acc_mean'] = np.mean(test_accs)
                model_results[f'toper_{filt}_super']['acc_std'] = np.std(test_accs)
            else:
                print(f'array super contains nan, did not compute clustering metrics and accuracies')

            if not np.isnan(array_sub).any():
                model_results[f'toper_{filt}_sub'] = {}

                all_clustering_results_euclidean(f'toper_{filt}_sub', model_results, array_sub, y)
                
                # sns.scatterplot(x=array_sub[:,0], y=array_sub[:,1], hue=y, palette='tab10') #, palette=cust_cmap
                # sns.despine()
                # #plt.show()
                # plt.savefig(results_dir + f"toper_{filt}_sub.png", dpi=300)
                # plt.close()

                if n_graphs > 1:
                    test_accs = knn_classifier(array_sub, np.array(y), precomputed=False, groups=np.array(n_nodes))
                else:
                    test_accs = knn_classifier(array_sub, np.array(y), precomputed=False)
                model_results[f'toper_{filt}_sub']['acc_mean'] = np.mean(test_accs)
                model_results[f'toper_{filt}_sub']['acc_std'] = np.std(test_accs)
                
                all_sub.append(array_sub)
            else:
                print(f'array sub contains nan, did not compute clustering metrics and accuracies')

            if not np.isnan(array_sub_super).any():
                model_results[f'toper_{filt}_sub_super'] = {}

                all_clustering_results_euclidean(f'toper_{filt}_sub_super', model_results, array_sub_super, y)

                if n_graphs > 1:
                    test_accs = knn_classifier(array_sub_super, np.array(y), precomputed=False, groups=np.array(n_nodes))
                else:
                    test_accs = knn_classifier(array_sub_super, np.array(y), precomputed=False)
                model_results[f'toper_{filt}_sub_super']['acc_mean'] = np.mean(test_accs)
                model_results[f'toper_{filt}_sub_super']['acc_std'] = np.std(test_accs)
            else:
                print(f'array sub_super contains nan, did not compute clustering metrics and accuracies')

        if len(all_sub) > 0:    
            all_sub = np.concatenate(all_sub, axis=1)
        if not np.isnan(all_sub).any():
            model_results[f'toper_all_sub'] = {}
            all_clustering_results_euclidean(f'toper_all_sub', model_results, all_sub, y)
            if n_graphs > 1:
                test_accs = knn_classifier(all_sub, np.array(y), precomputed=False, groups=np.array(n_nodes))
            else:
                test_accs = knn_classifier(all_sub, np.array(y), precomputed=False)
            model_results[f'toper_all_sub']['acc_mean'] = np.mean(test_accs)
            model_results[f'toper_all_sub']['acc_std'] = np.std(test_accs)
        else:
            print('array_sub_super contains nan, cannot compute clustering metrics')
    
        # spectralzoo
        moments = []
        for graph in G_nx:
            moments.append(est_moment(graph, 10000, 4))
        #np.array(moments)
        array=np.array(moments)#[:, 1:]
        array = array[:, ~np.all(array == 0, axis=0)]

        model_results['spectral_zoo'] = {}

        all_clustering_results_euclidean('spectral_zoo', model_results, array, y)

        # sns.scatterplot(x=array[:,0], y=array[:,1], hue=y, palette='tab10') #, palette=cust_cmap
        # sns.despine()
        # #plt.show()
        # plt.savefig(results_dir + f"spectral_zoo.png", dpi=300)
        # plt.close()

        if n_graphs > 1:
            test_accs = knn_classifier(array, np.array(y), precomputed=False, groups=np.array(n_nodes))
        else:
            test_accs = knn_classifier(array, np.array(y), precomputed=False)
        model_results['spectral_zoo']['acc_mean'] = np.mean(test_accs)
        model_results['spectral_zoo']['acc_std'] = np.std(test_accs)

        # # persistent homology
        # ph_filtrations = ['degree', 'ollivier_ricci_curvature', 'forman_ricci_curvature', 'laplacian', 'vietoris_rips', 'alpha_complex']

        # for filtration in ph_filtrations:
        #     model_results[f'persistent_homology_{filtration}'] = {}
        #     model_results[f'persistent_homology_mds_{filtration}'] = {}
        #     model_results[f'persistent_homology_pca_{filtration}'] = {}

        #     print(f'computing persistent homology with {filtration} filtration.')
        #     success = False
        #     try:
        #         array_ph = obtain_input_from_graphs(G_nx, filtration_fn_name=filtration, k=2)
        #         success = True
        #     except MemoryError as e:
        #         print(f'Memory Error {e} for filtration {filtration}')

        #     if success:
        #         # compute clusterin metrics based on euclidean distance of persistent homology
        #         all_clustering_results_euclidean(f'persistent_homology_{filtration}', model_results, array_ph, y)

        #         # computing accuracies
        #         if n_graphs > 1:
        #             test_accs = knn_classifier(array_ph, np.array(y), precomputed=False, groups=np.array(n_nodes))
        #         else:
        #             test_accs = knn_classifier(array_ph, np.array(y), precomputed=False)
                
        #         model_results[f'persistent_homology_{filtration}']['acc_mean'] = np.mean(test_accs)
        #         model_results[f'persistent_homology_{filtration}']['acc_std'] = np.std(test_accs)

        #         # MDS
        #         # euclidean distance
        #         embedding = MDS(n_components=2, dissimilarity="euclidean")
        #         X_transformed = embedding.fit_transform(array_ph)

        #         all_clustering_results_euclidean(f'persistent_homology_mds_{filtration}', model_results, X_transformed, y)
                
        #         # computing accuracies
        #         if n_graphs > 1:
        #             test_accs = knn_classifier(X_transformed, np.array(y), precomputed=False, groups=np.array(n_nodes))
        #         else:
        #             test_accs = knn_classifier(X_transformed, np.array(y), precomputed=False)
                
        #         model_results[f'persistent_homology_mds_{filtration}']['acc_mean'] = np.mean(test_accs)
        #         model_results[f'persistent_homology_mds_{filtration}']['acc_std'] = np.std(test_accs)

        #         # PCA
        #         # euclidean
        #         pca_embedding = PCA(n_components=2)
        #         X_pca = pca_embedding.fit_transform(array_ph)

        #         all_clustering_results_euclidean(f'persistent_homology_pca_{filtration}', model_results, X_pca, y)
                
        #         # computing accuracies
        #         if n_graphs > 1:
        #             test_accs = knn_classifier(X_pca, np.array(y), precomputed=False, groups=np.array(n_nodes))
        #         else:
        #             test_accs = knn_classifier(X_pca, np.array(y), precomputed=False)
                
        #         model_results[f'persistent_homology_pca_{filtration}']['acc_mean'] = np.mean(test_accs)
        #         model_results[f'persistent_homology_pca_{filtration}']['acc_std'] = np.std(test_accs)

        # get rank for performance of model config
        # ranks = {}

        res_df = pd.DataFrame.from_dict(model_results, orient='index')
        all_results[model] = res_df
        all_configs[model] = str(config)

        # for score in scores:
        #     ranks[score] = res_df[score].rank(axis=0, method='min', ascending=scores_asc[score])
        
        # scores_df = pd.DataFrame.from_dict(ranks, orient='columns')
        # scores_df.loc[:,'median'] = scores_df.median(axis=1, skipna=True)
        # scores_df.loc[:,'min'] = scores_df.min(axis=1)
        # scores_dfs.append(scores_df)

        # best_median_rank.append(min(scores_df.loc[(['divcurves_funs_diffs_shortest_path_distance', 'divcurves_shortest_path_distance', 'divcurves_mds_funs_diffs_shortest_path_distance','divcurves_mds_shortest_path_distance','divcurves_pca_shortest_path_distance'], 'median')].values)) #'divcurves_pca_funs_diffs_shortest_path_distance', 
        # best_min_rank.append(min(scores_df.loc[(['divcurves_funs_diffs_shortest_path_distance', 'divcurves_shortest_path_distance', 'divcurves_mds_funs_diffs_shortest_path_distance','divcurves_mds_shortest_path_distance','divcurves_pca_shortest_path_distance'], 'min')].values))

        # # find best model config per model  and save
        # best_configs[model] = str(model_configs[np.argmin(best_median_rank)])
        # best_res_dfs[model] = model_results[np.argmin(best_median_rank)]

        # best_configs[model] = str(model_configs[np.argmin(best_min_rank)])
        # best_res_dfs[model] = model_results[np.argmin(best_min_rank)]

    res_combined = pd.concat([all_results[key] for key in all_results.keys()], axis=1, keys=all_results.keys())
 
    # save results
    df_n_nodes = pd.DataFrame({"n_nodes": n_nodes})
    df_n_nodes.to_csv(f"{results_dir}/n_nodes.csv")

    res_combined.to_csv(f"{results_dir}/results_combined.csv")

    df_best_configs = pd.DataFrame.from_dict(all_configs, orient='index', columns=['configurations'])
    df_best_configs.to_csv(f'{results_dir}/configs.csv')



