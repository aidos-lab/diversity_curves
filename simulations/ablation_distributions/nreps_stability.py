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
)
from diversity_curves.baselines.spectral_zoo import est_moment
from diversity_curves.baselines.ph import obtain_input_from_graphs

if __name__ == "__main__":
    np.random.seed(43)
    random.seed(314)

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
        nargs='+',
        default=[1,3,5],
        help='Specify the number of repetitions in the diversity curves computation'
    )

    argparser.add_argument(
        '--nseeds',
        type=int,
        default=10,
        help='Specify the number of seeds over which we repeat the experiment'
    )

    args = argparser.parse_args()
    results_dir = args.output_dir
    n_low = args.nrange[0]
    n_high = args.nrange[1]
    n_steps = args.nsteps
    n_graphs = args.ngraphs
    n_reps = args.nreps
    n_seeds = args.nseeds

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

            for i_seed in range(n_seeds):

                res_dict[(f"divcurves_{metric}", i_seed)] = {}
                res_dict[(f"divcurves_pca_{metric}", i_seed)] = {}

                for n_rep in n_reps:
                    
                    print(f"Computing MAG with metric: {metric} and nrep: {n_rep} seed: {i_seed}")

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



                    # compute clustering metrics based on euclidean distance of diversity curve vectors
                    ari_kmeans_mean, ari_kmeans_std = compute_rand_kmeans(
                        np.array(mag_funs_interpolated), y
                    )
                    ari_agglo_mean, ari_agglo_std = compute_rand_agglo(
                        np.array(mag_funs_interpolated), y, metric="euclidean"
                    )

                    means, stds = compute_clustering_scores_resample(
                        np.array(mag_funs_interpolated), y
                    )

                    (
                        res_dict[(f"divcurves_{metric}", i_seed)][("silh_mean", n_rep)],
                        res_dict[(f"divcurves_{metric}", i_seed)][("db_mean", n_rep, )],
                        res_dict[(f"divcurves_{metric}", i_seed)][("ch_mean", n_rep)],
                    ) = means
                    (
                        res_dict[(f"divcurves_{metric}", i_seed)][("silh_std", n_rep)],
                        res_dict[(f"divcurves_{metric}", i_seed)][("db_std", n_rep)],
                        res_dict[(f"divcurves_{metric}", i_seed)][("ch_std", n_rep)],
                    ) = stds
                    res_dict[(f"divcurves_{metric}", i_seed)][("ari_agglo_mean", n_rep)] = ari_agglo_mean
                    res_dict[(f"divcurves_{metric}", i_seed)][("ari_agglo_std", n_rep)] = ari_agglo_std
                    res_dict[(f"divcurves_{metric}", i_seed)][("ari_kmeans_mean", n_rep)] = ari_kmeans_mean
                    res_dict[(f"divcurves_{metric}", i_seed)][("ari_kmeans_std", n_rep)] = ari_kmeans_std

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

                    res_dict[(f"divcurves_{metric}", i_seed)][("acc_mean", n_rep)] = np.mean(test_accs)
                    res_dict[(f"divcurves_{metric}", i_seed)][("acc_std", n_rep)] = np.std(test_accs)

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

                    res_dict[(f"divcurves_pca_{metric}", i_seed)][("acc_mean", n_rep)] = np.mean(test_accs)
                    res_dict[(f"divcurves_pca_{metric}", i_seed)][("acc_std", n_rep)] = np.std(test_accs)
                    # all_clustering_results_euclidean(f"divcurves_pca_{metric}_nreps_{n_rep}", res_dict, X_pca, y)
                    ari_kmeans_mean, ari_kmeans_std = compute_rand_kmeans(
                        X_pca, y
                    )
                    ari_agglo_mean, ari_agglo_std = compute_rand_agglo(
                        X_pca, y, metric="euclidean"
                    )

                    means, stds = compute_clustering_scores_resample(
                        X_pca, y
                    )

                    (
                        res_dict[(f"divcurves_pca_{metric}", i_seed)][("silh_mean", n_rep)],
                        res_dict[(f"divcurves_pca_{metric}", i_seed)][("db_mean", n_rep, )],
                        res_dict[(f"divcurves_pca_{metric}", i_seed)][("ch_mean", n_rep)],
                    ) = means
                    (
                        res_dict[(f"divcurves_pca_{metric}", i_seed)][("silh_std", n_rep)],
                        res_dict[(f"divcurves_pca_{metric}", i_seed)][("db_std", n_rep)],
                        res_dict[(f"divcurves_pca_{metric}", i_seed)][("ch_std", n_rep)],
                    ) = stds
                    res_dict[(f"divcurves_pca_{metric}", i_seed)][("ari_agglo_mean", n_rep)] = ari_agglo_mean
                    res_dict[(f"divcurves_pca_{metric}", i_seed)][("ari_agglo_std", n_rep)] = ari_agglo_std
                    res_dict[(f"divcurves_pca_{metric}", i_seed)][("ari_kmeans_mean", n_rep)] = ari_kmeans_mean
                    res_dict[(f"divcurves_pca_{metric}", i_seed)][("ari_kmeans_std", n_rep)] = ari_kmeans_std
                    
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

        
    # creating dataframes and saving results
    res_df = pd.DataFrame.from_dict(res_dict, orient="index")

    df_n_nodes = pd.DataFrame({"n_nodes": n_nodes})
    res_df.to_csv(f"{results_dir}/res_df.csv")
    df_n_nodes.to_csv(f"{results_dir}/n_nodes.csv")
