import networkx as nx
from torch_geometric.datasets import TUDataset
import networkx as nx
from magnipy.magnitude.distances import get_dist
from diversity_curves.process import (
    get_mag_graphs,
    get_weight_diffs,
    get_funs_diff,
    get_mag_areas,
    get_mag_funs_graphs,
)
import argparse
import os
import seaborn as sns
from diversity_curves.process import get_funs_df
from diversity_curves.utils import graph_processing_torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="?",
        default="ENZYMES",
        help="Which data split to use",
    )
    args = parser.parse_args()

    dataset_name = args.dataset

    results_dir = f"./results/{dataset_name}/"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    root = "./../datasets/TUDataset"
    pyg_dataset = TUDataset(root=root, name=dataset_name)

    van_graph_list = graph_processing_torch(pyg_dataset)
    lables = [data.y.item() for data in pyg_dataset]

    for metric in ["shortest_path_distance"]:
        # , "diffusion_distance", "heat_kernel_distance"]:
        def dist_fn(G):
            return get_dist(G=G, X=None, X2=None, metric=metric, mode="structure")

        n_nodes = [graph.number_of_nodes() for graph in van_graph_list]
        upsample_to = min(max(n_nodes), 101)

        if max(n_nodes) > 101:
            compute_at_cardinality = [r for r in range(1, 101)]
        else:
            compute_at_cardinality = None

        ts = [1]

        all_spreads, weights_lists = get_mag_graphs(
            van_graph_list,
            dist_fn,
            ts=ts,
            upsample_to=upsample_to,
            n_repeats=3,
            mode="structure",
            compute_at_cardinality=compute_at_cardinality,
        )
        weight_diffs = get_weight_diffs(weights_lists)
        funs_diffs = get_funs_diff(all_spreads)
        mag_funs = get_funs_df(all_spreads)
        mag_funs = mag_funs.loc[:, :upsample_to]
        mag_funs.to_csv(results_dir + f"mag_funs_{metric}.csv")
