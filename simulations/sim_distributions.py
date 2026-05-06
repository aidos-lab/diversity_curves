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

from grakel import GraphKernel
from karateclub import NetLSD, Graph2Vec, FeatherGraph

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


# #testing custom colormap https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10
# def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
#     if nc > plt.get_cmap(cmap).N:
#         raise ValueError("Too many categories for colormap.")
#     if continuous:
#         ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
#     else:
#         ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
#     cols = np.zeros((nc*nsc, 3))
#     for i, c in enumerate(ccolors):
#         chsv = matplotlib.colors.rgb_to_hsv(c[:3])
#         arhsv = np.tile(chsv,nsc).reshape(nsc,3)
#         arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
#         arhsv[:,2] = np.linspace(chsv[2],1,nsc)
#         rgb = matplotlib.colors.hsv_to_rgb(arhsv)
#         cols[i*nsc:(i+1)*nsc,:] = rgb       
#     cmap = matplotlib.colors.ListedColormap(cols)
#     return cmap

if __name__ == '__main__':
    np.random.seed(43)
    random.seed(314)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output_dir", type=str, default="./simulations/results/sim_distributions", help="Output directory for results")
    argparser.add_argument('--ngraphs', type=int, default=1, help='Number of graphs that are generated per cardinality')
    argparser.add_argument('--nrange', type=int, nargs=2, default=[10,30], help='specify the range of number of nodes of the graphs')
    argparser.add_argument('--nsteps', type=int, default=1, help='stepsize for node range generation')
    

    args = argparser.parse_args()
    output_dir = args.output_dir
    n_low = args.nrange[0]
    n_high = args.nrange[1]
    n_steps = args.nsteps
    n_graphs = args.ngraphs

    # ---- PARAMETERS TO CHOSE ----
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
    
    # noderanges = [range(10, 20)] #[5, 50], [5, 35],[5, 15][20, 55], [30,65], [40, 75], [20, 40], [20, 55], [10, 30], [10, 50] range(35, 50), range(20, 50, 2), range(5, 50, 3), range(10,30)
    noderange = range(n_low, n_high, n_steps)
    # n_graphs = 3

    downsample_to = None

    graph_metrics = ['shortest_path_distance']

    # class_per_config = False


    # --------------------------
    
    # cust_cmap = categorical_cmap(nc=4, nsc=1)
    cust_cmap = 'tab10'

    models = list(model_params.keys())
    all_configs = {}
    for model in models:
        #print(f"Model: {model}")
        # model_variants = model_params[model].keys()
        # model_combinations = list(itertools.product(*(model_params[model][variant] for variant in model_variants)))
        model_config = tuple(model_params[model][param] for param in model_params[model].keys())
        # print(f'model:{model} combinations:{model_combinations}')
        all_configs[model] = model_config
    configuration = tuple(all_configs[model] for model in models)
    print(f'total combs: {configuration}')
    # clustering results
    results = {}
    results_resample_mean = {}
    results_resample_std = {}

    scores = ['acc_mean', 'silh_mean', 'ari_agglo_mean', 'db_mean',	'ch_mean', 'ari_kmeans_mean']
    scores_asc = {'acc_mean': False, 'silh_mean': False, 'ari_agglo_mean': False, 'db_mean': True,'ch_mean': False, 'ari_kmeans_mean': False}


    # accuracies, classification different models
    accuracies = {}



    # for combination in total_combs:
        # print(combination)

    # for noderange in noderanges:
    res_dict = {}

    results_dir = f'{output_dir}/{configuration}/nodes_{min(noderange)}_to_{max(noderange)}/n_graphs_{n_graphs}/'
    # if n_graphs > 1:
    #     results_dir += f'n_graphs_{n_graphs}/'
    # if class_per_config:
    #     results_dir += 'class_per_config/'
    if not os.path.exists(results_dir): 
        os.makedirs(results_dir, exist_ok=True) 

    G_nx = []
    Gs = []
    y_configs = []
    n_nodes = []
    y_models = []
    class_labels = {}
    for i_m, model in enumerate(model_params):
        # model_kwargs = {}
        # # print(f'i_m:{i_m}')
        # for i_para, para in enumerate(model_params[model].keys()):
        #     # print(f'i_para: {i_para}')
        #     model_kwargs[model+'_'+para] = configuration[i_m][i_para]
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
        y_configs.extend([y_i+max(y_configs, default=0)+1 for y_i in y_model])
        print(y_configs)
        n_nodes.extend(n_nodes_model)
        y_models.extend([i_m]*len(G_nx_model))
        class_labels[i_m] = model

    upsample_to = max(n_nodes) 

    # if class_per_config:
    #     y=y_configs
    # else:
    y=y_models
    hue_names = [class_labels[i] for i in y]

    ##################################
    ### computing diversity curves ###
    ##################################

    for metric in graph_metrics:

        # res_dict[f'divcurves_funs_diffs_{metric}'] = {}
        res_dict[f'divcurves_{metric}'] = {}
        # res_dict[f'divcurves_mds_funs_diffs_{metric}'] = {}
        # res_dict[f'divcurves_mds_{metric}'] = {}
        # res_dict[f'divcurves_pca_funs_diffs_{metric}'] = {}
        res_dict[f'divcurves_pca_{metric}'] = {}
        # res_dict[f'div_one_scale_{metric}'] = {}
        res_dict[f'div_function_{metric}'] = {}

        def dist_fn(G):
            return get_dist(G=G, X=None, X2=None, metric=metric, mode="structure")
        
        # want to try what happens with ts = [1]
        all_spreads, weights_lists = get_mag_graphs(G_nx, dist_fn, ts=[1], upsample_to=upsample_to, downsample_to=downsample_to, n_repeats=3)
        
        mag_funs = get_funs_df(all_spreads)
        mag_funs_interpolated = mag_funs.interpolate(method='linear', axis='columns')
        funs_diffs = get_funs_diff(all_spreads)

        
        #plot_all_spread(all_spreads)

        # # compute clustering metrics with precomputed difference based on area between curves
        # ari_agglo_mean, ari_agglo_std = compute_rand_agglo(funs_diffs, y, metric='precomputed')
        # silh_mean, silh_std = compute_silhouette_score_resamples(funs_diffs, y)

        # res_dict[f'divcurves_funs_diffs_{metric}']['silh_mean'] = silh_mean
        # res_dict[f'divcurves_funs_diffs_{metric}']['silh_std'] = silh_std
        # res_dict[f'divcurves_funs_diffs_{metric}']['ari_agglo_mean'] = ari_agglo_mean
        # res_dict[f'divcurves_funs_diffs_{metric}']['ari_agglo_std'] = ari_agglo_std

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
        # if n_graphs > 1:
        #     test_accs = knn_classifier(funs_diffs, np.array(y), precomputed=True, groups=np.array(n_nodes))
        # else:
        #     test_accs = knn_classifier(funs_diffs, np.array(y), precomputed=True)
        
        # res_dict[f'divcurves_funs_diffs_{metric}']['acc_mean'] = np.mean(test_accs)
        # res_dict[f'divcurves_funs_diffs_{metric}']['acc_std'] = np.std(test_accs)

        if n_graphs > 1:
            test_accs = knn_classifier(np.array(mag_funs_interpolated), np.array(y), precomputed=False, groups=np.array(n_nodes))
        else:
            test_accs = knn_classifier(np.array(mag_funs_interpolated), np.array(y), precomputed=False)
        
        res_dict[f'divcurves_{metric}']['acc_mean'] = np.mean(test_accs)
        res_dict[f'divcurves_{metric}']['acc_std'] = np.std(test_accs)


        # MDS
        # funs_diff
        # embedding = MDS(n_components=2, dissimilarity="precomputed")
        # X_transformed = embedding.fit_transform(funs_diffs)

        # all_clustering_results_euclidean(f'divcurves_mds_funs_diffs_{metric}', res_dict, X_transformed, y)

        # if n_graphs > 1:
        #     test_accs = knn_classifier(X_transformed, np.array(y), precomputed=False, groups=np.array(n_nodes))
        # else:
        #     test_accs = knn_classifier(X_transformed, np.array(y), precomputed=False)
        
        # res_dict["divcurves_mds_funs_diffs_"+metric]['acc_mean'] = np.mean(test_accs)
        # res_dict["divcurves_mds_funs_diffs_"+metric]['acc_std'] = np.std(test_accs)
        
        sns.set_style('white')
        matplotlib.rcParams.update({'font.size': 20})

        # fig = sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=hue_names, palette='tab10')
        # fig.legend(frameon = False)
        # # sns.despine()
        # plt.savefig(results_dir + f"divcurves_mds_funs_diffs_{metric}.pdf", dpi=300, format='pdf')
        # plt.savefig(results_dir + f"divcurves_mds_funs_diffs_{metric}.png", dpi=300, format='png')
        # plt.close()

        # fig = sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=n_nodes)
        # fig.legend(frameon = False)
        # # sns.despine()
        # plt.savefig(results_dir + f"divcurves_mds_funs_diffs_n_nodes_{metric}.pdf", dpi=300, format='pdf')
        # plt.savefig(results_dir + f"divcurves_mds_funs_diffs_n_nodes_{metric}.png", dpi=300, format='png')
        # plt.close()

        # # MDS
        # # euclidean distance
        # embedding = MDS(n_components=2, dissimilarity="euclidean")
        # X_transformed = embedding.fit_transform(np.array(mag_funs_interpolated))

        # all_clustering_results_euclidean(f'divcurves_mds_{metric}', res_dict, X_transformed, y)

        # if n_graphs > 1:
        #     test_accs = knn_classifier(X_transformed, np.array(y), precomputed=False, groups=np.array(n_nodes))
        # else:
        #     test_accs = knn_classifier(X_transformed, np.array(y), precomputed=False)
        
        # res_dict[f'divcurves_mds_{metric}']['acc_mean'] = np.mean(test_accs)
        # res_dict[f'divcurves_mds_{metric}']['acc_std'] = np.std(test_accs)

        # fig = sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=hue_names, palette='tab10')
        # fig.legend(frameon = False)
        # # sns.despine()
        # plt.savefig(results_dir + f"divcurves_mds_{metric}.pdf", dpi=300, format='pdf')
        # plt.savefig(results_dir + f"divcurves_mds_{metric}.png", dpi=300, format='png')
        # plt.close()

        # fig = sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=n_nodes)
        # fig.legend(frameon = False)
        # # sns.despine()
        # plt.savefig(results_dir + f"divcurves_mds_n_nodes_{metric}.pdf", dpi=300, format='pdf')
        # plt.savefig(results_dir + f"divcurves_mds_n_nodes_{metric}.png", dpi=300, format='png')
        # plt.close()

        
        # PCA
        # euclidean
        pca_embedding = PCA(n_components=2)
        X_pca = pca_embedding.fit_transform(np.array(mag_funs_interpolated))

        all_clustering_results_euclidean(f'divcurves_pca_{metric}', res_dict, X_pca, y)
        fig = sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=hue_names, palette='tab10')
        fig.legend(
            frameon=False, 
            loc=(1.04, 0),
            bbox_to_anchor=(1, 0.5)
            )
        # sns.despine()
        plt.savefig(results_dir + f"divcurves_pca_{metric}.pdf", dpi=300, format='pdf', bbox_inches="tight")
        plt.savefig(results_dir + f"divcurves_pca_{metric}.png", dpi=300, format='png', bbox_inches="tight")
        plt.close()

        fig = sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=n_nodes)
        fig.legend(frameon = False)
        # sns.despine()
        plt.savefig(results_dir + f"divcurves_pca_n_nodes{metric}.pdf", dpi=300, format='pdf')
        plt.savefig(results_dir + f"divcurves_pca_n_nodes{metric}.png", dpi=300, format='png')
        plt.close()

        if n_graphs > 1:
            test_accs = knn_classifier(X_pca, np.array(y), precomputed=False, groups=np.array(n_nodes))
        else:
            test_accs = knn_classifier(X_pca, np.array(y), precomputed=False)
        
        res_dict[f'divcurves_pca_{metric}']['acc_mean'] = np.mean(test_accs)
        res_dict[f'divcurves_pca_{metric}']['acc_std'] = np.std(test_accs)

        # mag baselines
        # # magnitude/spread one curve
        # mags_original = []
        # for gr in all_spreads.graph.unique():
        #     spread_df_sub = all_spreads.loc[all_spreads.graph == gr,:]
        #     n_nodes_value = n_nodes[gr]
        #     this_mag = spread_df_sub.loc[n_nodes_value,:].iloc[0,:]
        #     mags_original.append(this_mag)
        # mags_original = pd.concat(mags_original, axis=1).T
        # mags = mags_original.iloc[:,0].values.reshape(-1, 1)

        #clustering
        # all_clustering_results_euclidean(f'div_one_scale_{metric}', res_dict, mags, y)

        # #accuracies
        # if n_graphs > 1:
        #     test_accs = knn_classifier(mags, np.array(y), precomputed=False, groups=np.array(n_nodes))
        # else:
        #     test_accs = knn_classifier(mags, np.array(y), precomputed=False)
        
        # res_dict[f'div_one_scale_{metric}']['acc_mean'] = np.mean(test_accs)
        # res_dict[f'div_one_scale_{metric}']['acc_std'] = np.std(test_accs)

        # usual magnitude function


                # ### try magnitude functions across multiple scales ###
        mags, _ = get_mag_funs_graphs(G_nx, dist_fn, ts=np.linspace(1,5,20), method="spread", get_weights=False)
        mag_funs = np.vstack(mags)

        if n_graphs > 1:
            test_accs = knn_classifier(mag_funs, np.array(y), precomputed=False, groups=np.array(n_nodes))
        else:
            test_accs = knn_classifier(mag_funs, np.array(y), precomputed=False)

        res_dict[f'div_function_{metric}']['acc_mean'] = np.mean(test_accs)
        res_dict[f'div_function_{metric}']['acc_std'] = np.std(test_accs)

        all_clustering_results_euclidean(f'div_function_{metric}', res_dict, mag_funs, y)


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

        res_dict[f'graph_embedding_{embedding_name}'] = {}
        res_dict[f'graph_embedding_mds_{embedding_name}'] = {}
        res_dict[f'graph_embedding_pca_{embedding_name}'] = {}

        all_clustering_results_euclidean(f'graph_embedding_{embedding_name}', res_dict, array, y) 
        
        # sns.scatterplot(x=array[:,0], y=array[:,1], hue=y, palette='tab10') #, palette=cust_cmap
        # sns.despine()
        # #plt.show()
        # plt.savefig(results_dir + f"spectral_zoo.pdf", dpi=300)
        # plt.close()
        if n_graphs > 1:
            test_accs = knn_classifier(array, np.array(y), precomputed=False, groups=np.array(n_nodes))
        else:
            test_accs = knn_classifier(array, np.array(y), precomputed=False)
        res_dict[f'graph_embedding_{embedding_name}']['acc_mean'] = np.mean(test_accs)
        res_dict[f'graph_embedding_{embedding_name}']['acc_std'] = np.std(test_accs)

        # MDS
        # embedding = MDS(n_components=2, metric="euclidean")
        # X_transformed = embedding.fit_transform(array)

        # all_clustering_results_euclidean(f'graph_embedding_mds_{embedding_name}', res_dict, X_transformed, y)
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

        # all_clustering_results_euclidean(f'graph_embedding_pca_{embedding_name}', res_dict, X_pca, y)
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
                "weisfeiler_lehman_optimal_assignment",
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
            res_dict[f'kernel_{k_}'] = {}
            res_dict[f'kernel_mds_{k_}'] = {}
            # precomputed clustering metrics
            ari_agglo_mean, ari_agglo_std = compute_rand_agglo(-K+K.max(), y, metric='precomputed')
            silh_mean, silh_std = compute_silhouette_score_resamples(-K+K.max(), y)

            res_dict[f'kernel_{k_}']['silh_mean'] = silh_mean
            res_dict[f'kernel_{k_}']['silh_std'] = silh_std
            res_dict[f'kernel_{k_}']['ari_agglo_mean'] = ari_agglo_mean
            res_dict[f'kernel_{k_}']['ari_agglo_std'] = ari_agglo_std

            # accuracies
            if n_graphs > 1:
                test_accs = knn_classifier(-K+K.max(), np.array(y), precomputed=True, groups=np.array(n_nodes))
            else:
                test_accs = knn_classifier(-K+K.max(), np.array(y), precomputed=True)
            res_dict[f'kernel_{k_}']['acc_mean'] = np.mean(test_accs)
            res_dict[f'kernel_{k_}']['acc_std'] = np.std(test_accs)

            # MDS
            if (np.var(-K+K.max(), axis=0) == 0).sum() < (-K+K.max()).shape[1]:
                embedding = MDS(n_components=2, dissimilarity="precomputed")
                X_transformed = embedding.fit_transform(-K+K.max())
                all_clustering_results_euclidean(f'kernel_mds_{k_}', res_dict, X_transformed, y)
                # sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=y, palette='tab10')
                # sns.despine()
                # plt.savefig(results_dir + f"kernel_mds_{k_}.png", dpi=300)
                # plt.close()

                # sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=n_nodes)
                # sns.despine()
                # plt.savefig(results_dir + f"kernel_mds_nnodes_{k_}.png", dpi=300)
                # plt.close()
                # accuracies
                if n_graphs > 1:
                    test_accs = knn_classifier(X_transformed, np.array(y), precomputed=False, groups=np.array(n_nodes))
                else:
                    test_accs = knn_classifier(X_transformed, np.array(y), precomputed=False)
                res_dict[f'kernel_mds_{k_}']['acc_mean'] = np.mean(test_accs)
                res_dict[f'kernel_mds_{k_}']['acc_std'] = np.std(test_accs)
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
            res_dict[f'toper_{filt}_super'] = {}

            all_clustering_results_euclidean(f'toper_{filt}_super', res_dict, array_super, y)
            
            # sns.scatterplot(x=array_super[:,0], y=array_super[:,1], hue=y, palette='tab10') #, palette=cust_cmap
            # sns.despine()
            # #plt.show()
            # plt.savefig(results_dir + f"toper_{filt}_super.png", dpi=300)
            # plt.close()

            if n_graphs > 1:
                test_accs = knn_classifier(array_super, np.array(y), precomputed=False, groups=np.array(n_nodes))
            else:
                test_accs = knn_classifier(array_super, np.array(y), precomputed=False)
            res_dict[f'toper_{filt}_super']['acc_mean'] = np.mean(test_accs)
            res_dict[f'toper_{filt}_super']['acc_std'] = np.std(test_accs)
        else:
            print(f'array super contains nan, did not compute clustering metrics and accuracies')

        if not np.isnan(array_sub).any():
            res_dict[f'toper_{filt}_sub'] = {}

            all_clustering_results_euclidean(f'toper_{filt}_sub', res_dict, array_sub, y)
            
            # sns.scatterplot(x=array_sub[:,0], y=array_sub[:,1], hue=y, palette='tab10') #, palette=cust_cmap
            # sns.despine()
            # #plt.show()
            # plt.savefig(results_dir + f"toper_{filt}_sub.png", dpi=300)
            # plt.close()

            if n_graphs > 1:
                test_accs = knn_classifier(array_sub, np.array(y), precomputed=False, groups=np.array(n_nodes))
            else:
                test_accs = knn_classifier(array_sub, np.array(y), precomputed=False)
            res_dict[f'toper_{filt}_sub']['acc_mean'] = np.mean(test_accs)
            res_dict[f'toper_{filt}_sub']['acc_std'] = np.std(test_accs)
            
            all_sub.append(array_sub)
        else:
            print(f'array sub contains nan, did not compute clustering metrics and accuracies')

        if not np.isnan(array_sub_super).any():
            res_dict[f'toper_{filt}_sub_super'] = {}

            all_clustering_results_euclidean(f'toper_{filt}_sub_super', res_dict, array_sub_super, y)

            if n_graphs > 1:
                test_accs = knn_classifier(array_sub_super, np.array(y), precomputed=False, groups=np.array(n_nodes))
            else:
                test_accs = knn_classifier(array_sub_super, np.array(y), precomputed=False)
            res_dict[f'toper_{filt}_sub_super']['acc_mean'] = np.mean(test_accs)
            res_dict[f'toper_{filt}_sub_super']['acc_std'] = np.std(test_accs)
        else:
            print(f'array sub_super contains nan, did not compute clustering metrics and accuracies')

    if len(all_sub) > 0:    
        all_sub = np.concatenate(all_sub, axis=1)
    if not np.isnan(all_sub).any():
        res_dict[f'toper_all_sub'] = {}
        all_clustering_results_euclidean(f'toper_all_sub', res_dict, all_sub, y)
        if n_graphs > 1:
            test_accs = knn_classifier(all_sub, np.array(y), precomputed=False, groups=np.array(n_nodes))
        else:
            test_accs = knn_classifier(all_sub, np.array(y), precomputed=False)
        res_dict[f'toper_all_sub']['acc_mean'] = np.mean(test_accs)
        res_dict[f'toper_all_sub']['acc_std'] = np.std(test_accs)
    else:
        print('array_sub_super contains nan, cannot compute clustering metrics')

    # spectralzoo
    moments = []
    for graph in G_nx:
        moments.append(est_moment(graph, 10000, 4))
    #np.array(moments)
    array=np.array(moments)#[:, 1:]
    array = array[:, ~np.all(array == 0, axis=0)]

    res_dict['spectral_zoo'] = {}

    all_clustering_results_euclidean('spectral_zoo', res_dict, array, y)

    # sns.scatterplot(x=array[:,0], y=array[:,1], hue=y, palette='tab10') #, palette=cust_cmap
    # sns.despine()
    # #plt.show()
    # plt.savefig(results_dir + f"spectral_zoo.png", dpi=300)
    # plt.close()

    if n_graphs > 1:
        test_accs = knn_classifier(array, np.array(y), precomputed=False, groups=np.array(n_nodes))
    else:
        test_accs = knn_classifier(array, np.array(y), precomputed=False)
    res_dict['spectral_zoo']['acc_mean'] = np.mean(test_accs)
    res_dict['spectral_zoo']['acc_std'] = np.std(test_accs)

    # # persistent homology
    # ph_filtrations = ['degree', 'ollivier_ricci_curvature', 'forman_ricci_curvature', 'laplacian', 'vietoris_rips', 'alpha_complex']

    # for filtration in ph_filtrations:
    #     res_dict[f'persistent_homology_{filtration}'] = {}
    #     res_dict[f'persistent_homology_mds_{filtration}'] = {}
    #     res_dict[f'persistent_homology_pca_{filtration}'] = {}

    #     print(f'computing persistent homology with {filtration} filtration.')
    #     success = False
    #     try:
    #         array_ph = obtain_input_from_graphs(G_nx, filtration_fn_name=filtration, k=2)
    #         success = True
    #     except MemoryError as e:
    #         print(f'Memory Error {e} for filtration {filtration}')

    #     if success:
    #         # compute clusterin metrics based on euclidean distance of persistent homology
    #         all_clustering_results_euclidean(f'persistent_homology_{filtration}', res_dict, array_ph, y)

    #         # computing accuracies
    #         if n_graphs > 1:
    #             test_accs = knn_classifier(array_ph, np.array(y), precomputed=False, groups=np.array(n_nodes))
    #         else:
    #             test_accs = knn_classifier(array_ph, np.array(y), precomputed=False)
            
    #         res_dict[f'persistent_homology_{filtration}']['acc_mean'] = np.mean(test_accs)
    #         res_dict[f'persistent_homology_{filtration}']['acc_std'] = np.std(test_accs)

    #         # MDS
    #         # euclidean distance
    #         embedding = MDS(n_components=2, dissimilarity="euclidean")
    #         X_transformed = embedding.fit_transform(array_ph)

    #         all_clustering_results_euclidean(f'persistent_homology_mds_{filtration}', res_dict, X_transformed, y)
            
    #         # computing accuracies
    #         if n_graphs > 1:
    #             test_accs = knn_classifier(X_transformed, np.array(y), precomputed=False, groups=np.array(n_nodes))
    #         else:
    #             test_accs = knn_classifier(X_transformed, np.array(y), precomputed=False)
            
    #         res_dict[f'persistent_homology_mds_{filtration}']['acc_mean'] = np.mean(test_accs)
    #         res_dict[f'persistent_homology_mds_{filtration}']['acc_std'] = np.std(test_accs)

    #         # PCA
    #         # euclidean
    #         pca_embedding = PCA(n_components=2)
    #         X_pca = pca_embedding.fit_transform(array_ph)

    #         all_clustering_results_euclidean(f'persistent_homology_pca_{filtration}', res_dict, X_pca, y)
            
    #         # computing accuracies
    #         if n_graphs > 1:
    #             test_accs = knn_classifier(X_pca, np.array(y), precomputed=False, groups=np.array(n_nodes))
    #         else:
    #             test_accs = knn_classifier(X_pca, np.array(y), precomputed=False)
            
    #         res_dict[f'persistent_homology_pca_{filtration}']['acc_mean'] = np.mean(test_accs)
    #         res_dict[f'persistent_homology_pca_{filtration}']['acc_std'] = np.std(test_accs)

    
    #creating dataframes and saving results
    res_df = pd.DataFrame.from_dict(res_dict, orient='index')

    df_n_nodes = pd.DataFrame({"n_nodes": n_nodes})
    res_df.to_csv(f"{results_dir}/res_df.csv")
    df_n_nodes.to_csv(f"{results_dir}/n_nodes.csv")

    # results_df = pd.DataFrame.from_dict(results, orient="index", columns=["silhouette", "davies_bouldin", "calinski_harabasz"])
    # results_df.to_csv(results_dir + "clustering_results.csv")

    # resample_mean_df = pd.DataFrame.from_dict(results_resample_mean, orient="index", columns=["silhouette_mean", "davies_bouldin_mean", "calinski_harabasz_mean"])
    # resample_mean_df.to_csv(results_dir + "clustering_results_resample_mean.csv")

    # resample_std_df = pd.DataFrame.from_dict(results_resample_std, orient="index", columns=["silhouette_std", "davies_bouldin_std", "calinski_harabasz_std"])
    # resample_std_df.to_csv(results_dir + "clustering_results_resample_std.csv")