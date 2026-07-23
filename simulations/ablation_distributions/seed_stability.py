import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import argparse
import os
import random

from grakel import GraphKernel
from karateclub import NetLSD, Graph2Vec, FeatherGraph
import toper

from sklearn.manifold import MDS
from sklearn.decomposition import PCA

from magnipy.magnitude.distances import get_dist
from diversity_curves.process import get_mag_graphs, get_funs_df, get_mag_funs_graphs

from diversity_curves.simulate import simulate_graphs_kwargs
from diversity_curves.classifiers import knn_classifier
from diversity_curves.clustering_performance import (
    compute_clustering_scores_resample,
    compute_silhouette_score_resamples,
    compute_rand_agglo,
    compute_rand_kmeans,
    all_clustering_results_euclidean,
    all_clustering_results_euclidean_append,
)
from diversity_curves.baselines.spectral_zoo import est_moment
from diversity_curves.baselines.ph import obtain_input_from_graphs

if __name__ == "__main__":


    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--output_dir",
        type=str,
        default="./simulations/results/ablations/",
        help="Output directory for results",
    )
    argparser.add_argument(
        "--ngraphs",
        type=int,
        default=3,
        help="Number of graphs that are generated per cardinality",
    )
    argparser.add_argument(
        "--nrange",
        type=int,
        nargs=2,
        default=[10, 30],
        help="specify the range of cardinalities of the generated graphs",
    )
    argparser.add_argument(
        "--nsteps",
        type=int,
        default=1,
        help="Specify the stepsize for the cardinalities of the generated graphs",
    )
    argparser.add_argument(
        '--nreps',
        type=int,
        default=1,
        help='Specify the number of repetitions in the diversity curves computation'
    )
    argparser.add_argument(
        '--nseed',
        type=int,
        default=10,
        help='specify the number of different seeds for the ablation study',
    )
    argparser.add_argument(
        '--comparison_method',
        type = str,
        default='all_but_self',
        help='Choose to the diversity curves of which graphs we can compare when computing the control. Either `all_but_self`: chooses random graph that is differnt from itself; `different_group`: choose a random graph from different distribution.'
    )
    args = argparser.parse_args()
    results_dir = args.output_dir
    n_low = args.nrange[0]
    n_high = args.nrange[1]
    n_steps = args.nsteps
    n_graphs = args.ngraphs
    n_rep = args.nreps
    n_seeds = args.nseed
    comp_method = args.comparison_method

    # Setting the parameters as reported in the paper
    model_params = {
        "ER": {
            "ER_probs": [0.75],
            "ER_nps": None,
        },
        "RPG": {
            "RPG_probs": [[0.9, 0.1]],
            "RPG_props": [1 / 3, 1 / 3],
        },
        "SBM": {"SBM_probs": [0.8, 0.05], "SBM_blocks": [[1 / 4, 1 / 4, 1 / 4]]},
        "RG": {"RG_probs": [0.25]},
    }

    # choice of graph metric we use for the spread computation
    graph_metrics = ["shortest_path_distance"]

    # get configuration of all models
    models = list(model_params.keys())
    all_configs = {}
    for model in models:
        model_config = tuple(
            model_params[model][param] for param in model_params[model].keys()
        )
        all_configs[model] = model_config
    configuration = tuple(all_configs[model] for model in models)

    noderange = range(n_low, n_high, n_steps)
    downsample_to = None

    scores = [
        "acc_mean",
        "silh_mean",
        "ari_agglo_mean",
        "db_mean",
        "ch_mean",
        "ari_kmeans_mean",
    ]
    scores_asc = {
        "acc_mean": False,
        "silh_mean": False,
        "ari_agglo_mean": False,
        "db_mean": True,
        "ch_mean": False,
        "ari_kmeans_mean": False,
    }

    res_dict = {}

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    avg_diffs = []
    std_diffs = []
    all_diffs = []
    all_diffs_meancurve = []
    avg_rand_diffs = []
    std_rand_diffs = []
    all_rand_diffs = []
    all_rand_diffs_meancurve = []
    accs_seeds = {}

    # generate the graphs of all models and create label per model
    G_nx = []
    Gs = []
    n_nodes = []
    y_models = []
    class_labels = {}
    for i_m, model in enumerate(model_params):
        model_kwargs = model_params[model]
        print(f"model {model} kwargs {model_kwargs}")
        G_nx_model, Gs_model, y_model, y_probs_model, n_nodes_model = (
            simulate_graphs_kwargs(
                model=model,
                n_graphs=n_graphs,
                nodes_range=noderange,
                **model_kwargs,
            )
        )
        G_nx.extend(G_nx_model)
        Gs.extend(Gs_model)
        n_nodes.extend(n_nodes_model)
        y_models.extend([i_m] * len(G_nx_model))
        class_labels[i_m] = model

    upsample_to = max(n_nodes)
    y = y_models
    hue_names = [class_labels[i] for i in y]

    # setting parameters for plots
    sns.set_style("white")
    matplotlib.rcParams.update({"font.size": 20})

    ##################################
    ### computing diversity curves ###
    ##################################

    for metric in graph_metrics:

        mag_funs_per_seed = []
    
        res_dict[f"divcurves_{metric}"] = {}
        res_dict[f"divcurves_pca_{metric}"] = {}
        accs_seeds[f"divcurves_{metric}"] = []
        accs_seeds[f"divcurves_pca_{metric}"] = []

        for score in scores:
            res_dict[f"divcurves_{metric}"][score] = []
            res_dict[f"divcurves_pca_{metric}"][score] = []
            score_std = score[:-4]+'std'
            res_dict[f"divcurves_{metric}"][score_std] = []
            res_dict[f"divcurves_pca_{metric}"][score_std] = []

        for seed in range(n_seeds):
            np.random.seed(seed)
            random.seed(seed)
            
            print(f"Computing MAG with metric: {metric} for seed: {seed}")

            def dist_fn(G):
                return get_dist(G=G, X=None, X2=None, metric=metric, mode="structure")

            all_spreads, weights_lists = get_mag_graphs(
                G_nx,
                dist_fn,
                ts=[1],
                upsample_to=upsample_to,
                downsample_to=downsample_to,
                n_repeats=n_rep,
                scores_method='random',
                mode='structure',
                set_seed=False
            )

            mag_funs = get_funs_df(all_spreads)
            mag_funs_interpolated = mag_funs.interpolate(method="linear", axis="columns")

            mag_funs_per_seed.append(mag_funs_interpolated)

            # # compute clustering metrics based on euclidean distance of diversity curve vectors
            # ari_kmeans_mean, ari_kmeans_std = compute_rand_kmeans(
            #     np.array(mag_funs_interpolated), y
            # )
            # ari_agglo_mean, ari_agglo_std = compute_rand_agglo(
            #     np.array(mag_funs_interpolated), y, metric="euclidean"
            # )

            # means, stds = compute_clustering_scores_resample(
            #     np.array(mag_funs_interpolated), y
            # )


            # res_dict[f"divcurves_{metric}"]["silh_mean"].append(means[0]),
            # res_dict[f"divcurves_{metric}"]["db_mean"].append(means[1]),
            # res_dict[f"divcurves_{metric}"]["ch_mean"].append(means[2]),

            # res_dict[f"divcurves_{metric}"]["silh_std"].append(means[0]),
            # res_dict[f"divcurves_{metric}"]["db_std"].append(means[1]),
            # res_dict[f"divcurves_{metric}"]["ch_std"].append(means[2]),

            # res_dict[f"divcurves_{metric}"]["ari_agglo_mean"].append(ari_agglo_mean)
            # res_dict[f"divcurves_{metric}"]["ari_agglo_std"].append(ari_agglo_std)
            # res_dict[f"divcurves_{metric}"]["ari_kmeans_mean"].append(ari_kmeans_mean)
            # res_dict[f"divcurves_{metric}"]["ari_kmeans_std"].append(ari_kmeans_std)

            # computing accuracies
            if n_graphs > 1:
                test_accs = knn_classifier(
                    np.array(mag_funs_interpolated),
                    np.array(y),
                    precomputed=False,
                    groups=np.array(n_nodes),
                )
            else:
                test_accs = knn_classifier(
                    np.array(mag_funs_interpolated), np.array(y), precomputed=False
                )
            accs_seeds[f"divcurves_{metric}"].append(test_accs)
            res_dict[f"divcurves_{metric}"]["acc_mean"].append(np.mean(test_accs))
            res_dict[f"divcurves_{metric}"]["acc_std"].append(np.std(test_accs))
            all_clustering_results_euclidean_append(f"divcurves_{metric}", res_dict, np.array(mag_funs_interpolated), y)

            # doing PCA with Euclidean distance between the diversity curves
            pca_embedding = PCA(n_components=2)
            X_pca = pca_embedding.fit_transform(np.array(mag_funs_interpolated))

            # computing clustering scores and accuracies on transformed data from PCA
            if n_graphs > 1:
                test_accs = knn_classifier(
                    X_pca, np.array(y), precomputed=False, groups=np.array(n_nodes)
                )
            else:
                test_accs = knn_classifier(X_pca, np.array(y), precomputed=False)
            accs_seeds[f"divcurves_pca_{metric}"].append(test_accs)
            res_dict[f"divcurves_pca_{metric}"]["acc_mean"].append(np.mean(test_accs))
            res_dict[f"divcurves_pca_{metric}"]["acc_std"].append(np.std(test_accs))
            all_clustering_results_euclidean_append(f"divcurves_pca_{metric}", res_dict, X_pca, y)

            # # creating plots
            # fig = sns.scatterplot(
            #     x=X_pca[:, 0], y=X_pca[:, 1], hue=hue_names, palette="tab10"
            # )
            # fig.legend(frameon=False, loc=(1.04, 0), bbox_to_anchor=(1, 0.5))
            # sns.despine()
            # plt.savefig(
            #     results_dir + f"divcurves_pca_{metric}_nreps_{n_rep}.pdf",
            #     dpi=300,
            #     format="pdf",
            #     bbox_inches="tight",
            # )
            # plt.savefig(
            #     results_dir + f"divcurves_pca_{metric}_nreps_{n_rep}.png",
            #     dpi=300,
            #     format="png",
            #     bbox_inches="tight",
            # )
            # plt.close()

        #compute mean curve over all repeats
        df_mag_funs_all_seeds = pd.concat([mag_funs_per_seed[i] for i in range(n_seeds)])
        df_mean_curve = df_mag_funs_all_seeds.groupby(df_mag_funs_all_seeds.index).mean()

        # mean and std (over all graphs) for l2 distance in difference seeds
        for seed_i in range(n_seeds):
            # getting a difference between the curves from random seed
            l2_differences = []
            for graph_index in df_mean_curve.index:
                this_curve, mean_curve = mag_funs_per_seed[seed_i].loc[graph_index], df_mean_curve.loc[graph_index]
                l2norm = np.linalg.norm([this_curve - mean_curve], ord=None)
                l2_differences.append(l2norm)
                all_diffs_meancurve.append(l2norm)
            avg_diffs.append(np.mean(l2_differences))
            std_diffs.append(np.std(l2_differences))

        #as a control we plot the mean distances between some randomly picked graphs

        # mean and std (over all graphs) for l2 distance in difference seeds to the mean curve
        for seed_i in range(n_seeds):
            # getting a difference between the curves from random seed
            l2_differences = []
            
            for graph_index in df_mean_curve.index:
                if comp_method == 'all_but_self':
                    other_graph = np.random.choice(np.delete(list(df_mean_curve.index),graph_index))
                elif comp_method == 'different_group':
                    this_label = y[graph_index]
                    other_indices_options = [i for i in df_mean_curve.index if y[i] != this_label]
                    other_graph = np.random.choice(other_indices_options)
                else:
                    print('Choose comparison method from `all_but_self` or `different_group`')
                this_curve, other_curve = mag_funs_per_seed[seed_i].loc[graph_index], df_mean_curve.loc[other_graph]
                l2norm = np.linalg.norm([this_curve - other_curve], ord=None)
                l2_differences.append(l2norm)
                all_rand_diffs_meancurve.append(l2norm)
            avg_rand_diffs.append(np.mean(l2_differences))
            std_rand_diffs.append(np.std(l2_differences))

        #create version of distance distribution without mean curve
        comp_method = 'different_group'

        for seed_i in range(n_seeds):
            for seed_j in range(seed_i+1, n_seeds):
                for graph_index in df_mean_curve.index:
                    curve_i, curve_j = mag_funs_per_seed[seed_i].loc[graph_index], mag_funs_per_seed[seed_j].loc[graph_index]
                    l2norm = np.linalg.norm([curve_i - curve_j], ord=None)
                    all_diffs.append(l2norm)

        #control with random curves
        for seed_i in range(n_seeds):
            for seed_j in range(seed_i+1, n_seeds):
                for graph_index in df_mean_curve.index:
                    if comp_method == 'all_but_self':
                        other_graphs = np.delete(list(df_mean_curve.index), graph_index)
                    elif comp_method == 'different_group':
                        this_label = y[graph_index]
                        other_graphs = [i for i in df_mean_curve.index if y[i] != this_label]
                    else:
                        print('Choose comparison method from `all_but_self` or `different_group`')
                    other_graph_index = np.random.choice(other_graphs)
                    this_curve, other_curve = mag_funs_per_seed[seed_i].loc[graph_index], mag_funs_per_seed[seed_j].loc[other_graph_index]
                    l2norm = np.linalg.norm([this_curve - other_curve], ord=None)
                    all_rand_diffs.append(l2norm)
        
    # creating dataframes and saving results
    res_tuples = {(i,j): res_dict[i][j] 
       for i in res_dict.keys() 
       for j in res_dict[i].keys()}
    res_df = pd.DataFrame(list(res_tuples.values()), index=pd.MultiIndex.from_tuples(res_tuples.keys()))

    all_diffs_mean_dict = {
        'same': all_diffs_meancurve,
        'rand': all_rand_diffs_meancurve
    }

    all_diffs_mean_df = pd.DataFrame.from_dict(all_diffs_mean_dict, orient='index')

    all_diffs_dict = {
            'same': all_diffs,
            'rand': all_rand_diffs 
        }
    
    all_diffs_df = pd.DataFrame.from_dict(all_diffs_dict, orient='index')

    seed_diffs_dict = {
        'same_mean': avg_diffs,
        'same_std': std_diffs,
        'rand_mean': avg_rand_diffs,
        'rand_std': std_rand_diffs,
    }

    seed_diffs_df = pd.DataFrame.from_dict(seed_diffs_dict, orient='index')

    accuracies_tuples = {(i,j): accs_seeds[i][j] 
       for i in accs_seeds.keys() 
       for j in range(len(accs_seeds[i]))}
    accs_seeds_df = pd.DataFrame(list(accuracies_tuples.values()), index=pd.MultiIndex.from_tuples(accuracies_tuples.keys()))

    df_n_nodes = pd.DataFrame({"n_nodes": n_nodes})
    res_df.to_csv(f"{results_dir}/res_df.csv")
    all_diffs_mean_df.to_csv(f'{results_dir}/all_diffs_mean.csv')
    all_diffs_df.to_csv(f'{results_dir}/all_diffs.csv')
    seed_diffs_df.to_csv(f'{results_dir}/seed_diffs.csv')
    accs_seeds_df.to_csv(f'{results_dir}/accs_seed.csv')
    df_n_nodes.to_csv(f"{results_dir}/n_nodes.csv")
