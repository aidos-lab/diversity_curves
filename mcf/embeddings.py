import argparse
import contextlib
import os
import signal
import time

import networkx as nx
import numpy as np
import pandas as pd
from magnipy.magnitude.distances import get_dist
from torch_geometric.datasets import TUDataset

from diversity_curves.classifiers import knn_classifier_cv
from diversity_curves.process import get_funs_df, get_mag_graphs
from diversity_curves.utils import graph_processing_torch

TIMEOUT_SECONDS = 7200

CLASSIFIER_KWARGS = dict(
    outer_k=10,
    holdout_test_size=0.1,
    stratify=True,
    tune=True,
    seed=42,
    scale=False,
)

KERNEL_NAMES = [
    "shortest_path",
    "pyramid_match",
    "neighborhood_hash",
    "weisfeiler_lehman",
    "core_framework",
    "weisfeiler_lehman_optimal_assignment",
]

TOPER_FILTS = ["degree", "popularity", "fricci", "oricci", "closeness"]

MAG_DISTANCE_METRICS = ["shortest_path_distance", "diffusion_distance"]

METHODS = (
    [f"mag_{m}" for m in MAG_DISTANCE_METRICS]
    + ["netlsd", "graph2", "feather"]
    + [f"kernel_{k}" for k in KERNEL_NAMES]
    + [f"toper_{f}" for f in TOPER_FILTS]
    + ["spectral_zoo"]
)


@contextlib.contextmanager
def timeout(seconds):
    def _handler(signum, frame):
        raise TimeoutError(f"computation timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def run_classification(name, X, y, precomputed, classification_path):
    test_accs, test_aurocs, test_accs_balanced, conf_matrices = knn_classifier_cv(
        X=X, y=y, precomputed=precomputed, **CLASSIFIER_KWARGS
    )
    pd.DataFrame(
        {
            "fold": range(len(test_accs)),
            "test_accuracy": test_accs,
            "test_balanced_accuracy": test_accs_balanced,
            "test_auroc": test_aurocs,
            "confusion_matrix": [cm.tolist() for cm in conf_matrices],
        }
    ).to_csv(f"{classification_path}{name}.csv", index=False)


def run_mag(metric, van_graph_list, results_path, y):
    def dist_fn(G):
        return get_dist(G=G, X=None, X2=None, metric=metric, mode="structure")

    n_nodes = [graph.number_of_nodes() for graph in van_graph_list]
    upsample_to = min(max(n_nodes), 50)

    if max(n_nodes) > 101:
        compute_at_cardinality = [r for r in range(5, 101, 5)]
    else:
        compute_at_cardinality = None

    all_spreads, weights_lists = get_mag_graphs(
        van_graph_list,
        dist_fn,
        ts=[1],
        upsample_to=upsample_to,
        n_repeats=3,
        mode="structure",
        compute_at_cardinality=compute_at_cardinality,
    )
    mag_funs = get_funs_df(all_spreads)
    mag_funs = mag_funs.loc[:, :upsample_to]

    num_cols = mag_funs.select_dtypes(include=[np.number]).columns
    mag_funs = mag_funs[num_cols].interpolate(method="linear", axis=0, limit_direction="both")

    mag_funs.to_csv(f"{results_path}mag_funs_{metric}.csv")

    y_aligned = y[mag_funs.index]
    return [
        {
            "name": f"mag_funs_{metric}",
            "X": mag_funs.values,
            "y": y_aligned,
            "precomputed": False,
        }
    ]


def run_karateclub_embedding(method, G_nx, results_path, y):
    from karateclub import FeatherGraph, Graph2Vec, NetLSD

    G_nx_fixed = [
        nx.convert_node_labels_to_integers(g, ordering="sorted") for g in G_nx
    ]

    if method == "netlsd":
        embedding = NetLSD()
    elif method == "graph2":
        embedding = Graph2Vec()
    elif method == "feather":
        embedding = FeatherGraph()

    embedding.fit([g.copy() for g in G_nx_fixed])
    array = embedding.get_embedding()

    pd.DataFrame(array).to_csv(f"{results_path}{method}.csv", index=False)
    return [{"name": method, "X": array, "y": y, "precomputed": False}]


def run_kernel(kernel_name, G_nx, results_path, y):
    from grakel import GraphKernel, graph_from_networkx

    Gs = []
    for graph in G_nx:
        nx.set_node_attributes(graph, 1, name="feature")  # dummy label for nx
        grakel_gen = graph_from_networkx(
            [graph], as_Graph=True, node_labels_tag="feature"
        )
        Gs.append(next(grakel_gen))

    if kernel_name in ["weisfeiler_lehman", "core_framework"]:
        gk = GraphKernel(
            kernel=[{"name": kernel_name}, {"name": "shortest_path"}],
            normalize=True,
        )
    else:
        gk = GraphKernel(kernel=[{"name": kernel_name}], normalize=True)

    K = gk.fit_transform(Gs)
    pd.DataFrame(K).to_csv(f"{results_path}kernel_{kernel_name}.csv", index=False)
    #if precomputed:
    K=-K+K.max()
    return [{"name": f"kernel_{kernel_name}", "X": K, "y": y, "precomputed": True}]


def run_toper(filt, G_nx, results_path, y):
    import toper

    F = toper.main(G_nx, [filt], num_segm=501)

    out_filt = filt
    if filt == "fricci":
        out_filt = "forricci"
    elif filt == "oricci":
        out_filt = "olricci"

    outputs = []
    for kind in ["super", "sub"]:
        array = np.array(F[f"{out_filt}_{kind}"])
        name = f"toper_{out_filt}_{kind}"
        pd.DataFrame(array).to_csv(f"{results_path}{name}.csv", index=False)
        outputs.append({"name": name, "X": array, "y": y, "precomputed": False})
    return outputs


def run_spectral_zoo(G_nx, results_path, y):
    from diversity_curves.baselines.spectral_zoo import est_moment

    moments = [est_moment(graph, 10000, 4) for graph in G_nx]
    array = np.array(moments)
    array = array[:, ~np.all(array == 0, axis=0)]

    pd.DataFrame(array).to_csv(f"{results_path}spectral_zoo_moments.csv", index=False)
    return [{"name": "spectral_zoo_moments", "X": array, "y": y, "precomputed": False}]


def run_method(method, van_graph_list, results_path, y):
    if method.startswith("mag_"):
        return run_mag(method[len("mag_") :], van_graph_list, results_path, y)
    elif method in ("netlsd", "graph2", "feather"):
        return run_karateclub_embedding(method, van_graph_list, results_path, y)
    elif method.startswith("kernel_"):
        return run_kernel(method[len("kernel_") :], van_graph_list, results_path, y)
    elif method.startswith("toper_"):
        return run_toper(method[len("toper_") :], van_graph_list, results_path, y)
    elif method == "spectral_zoo":
        return run_spectral_zoo(van_graph_list, results_path, y)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="?",
        default="MCF-7",
        help="Which data split to use",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=METHODS,
        help="Which embedding method to compute",
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    method = args.method

    results_dir = f"./results/{dataset_name}/"
    os.makedirs(results_dir, exist_ok=True)

    timings_dir = os.path.join(results_dir, "timings")
    os.makedirs(timings_dir, exist_ok=True)

    classification_dir = os.path.join(results_dir, "classification") + os.sep
    os.makedirs(classification_dir, exist_ok=True)

    root = "./../datasets/TUDataset"
    pyg_dataset = TUDataset(root=root, name=dataset_name)
    van_graph_list = graph_processing_torch(pyg_dataset)
    labels = np.array([data.y.item() for data in pyg_dataset])

    start_time = time.time()
    try:
        with timeout(TIMEOUT_SECONDS):
            outputs = run_method(method, van_graph_list, results_dir, labels)
        elapsed = time.time() - start_time
    except TimeoutError:
        print(f"{method} timed out after {TIMEOUT_SECONDS}s")
        elapsed = TIMEOUT_SECONDS
        outputs = []
    except Exception as e:
        print(f"{method} failed with error: {e}")
        elapsed = float("nan")
        outputs = []

    pd.DataFrame([{"method": method, "time_seconds": elapsed}]).to_csv(
        os.path.join(timings_dir, f"{method}.csv"), index=False
    )

    for output in outputs:
        name = output["name"]
        classification_start = time.time()
        try:
            run_classification(
                name, output["X"], output["y"], output["precomputed"], classification_dir
            )
            classification_elapsed = time.time() - classification_start
        except Exception as e:
            print(f"classification for {name} failed with error: {e}")
            classification_elapsed = float("nan")

        pd.DataFrame([{"method": name, "time_seconds": classification_elapsed}]).to_csv(
            os.path.join(timings_dir, f"{name}_classification.csv"), index=False
        )
