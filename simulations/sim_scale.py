import argparse
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from grakel import GraphKernel
from karateclub import NetLSD, FeatherGraph
import toper

from magnipy.magnitude.distances import get_dist
from diversity_curves.process import get_mag_graphs, get_funs_df
from diversity_curves.simulate import simulate_graphs_kwargs
from diversity_curves.classifiers import knn_classifier
from diversity_curves.clustering_performance import (
    compute_silhouette_score_resamples,
    compute_rand_agglo,
    all_clustering_results_euclidean,
)
from diversity_curves.baselines.spectral_zoo import est_moment
from diversity_curves.baselines.ph import obtain_input_from_graphs

ALL_METHODS = [
    "divcurves",
    "kernel_shortest_path",
    "kernel_wl_oa",
    "toper",
    "netlsd",
    "feather",
    "spectral_zoo",
    "ph_vietoris_rips",
]
EXPENSIVE_METHODS = {"kernel_shortest_path", "kernel_wl_oa", "toper", "ph_vietoris_rips"}

DISPLAY_NAMES = {
    "divcurves": "Diversity Curves (ours)",
    "kernel_wl_oa": "Kernel WL Opt. Assig.",
    "kernel_shortest_path": "Kernel Shortest Path",
    "toper": "TopER",
    "netlsd": "NetLSD",
    "feather": "FEATHER",
    "spectral_zoo": "Spectral Zoo",
    "ph_vietoris_rips": "PH Vietoris-Rips",
}

metric = "shortest_path_distance"


def dist_fn(G):
    return get_dist(G=G, X=None, X2=None, metric=metric, mode="structure")


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def resolve_n_jobs(n_jobs):
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        return os.cpu_count() or 1
    return n_jobs


def chunk_list(items, n_chunks):
    n_chunks = max(1, min(n_chunks, len(items)))
    chunk_size = math.ceil(len(items) / n_chunks)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def parallel_map_chunks(func, items, n_jobs, *extra_args):
    if n_jobs <= 1 or len(items) <= 1:
        return [items], [func(items, *extra_args)]
    chunks = chunk_list(items, n_jobs)
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [executor.submit(func, chunk, *extra_args) for chunk in chunks]
        results = [f.result() for f in futures]
    return chunks, results


def _generate_one_model(model, model_kwargs, n_graphs, node_sizes):
    start = time.perf_counter()
    result = simulate_graphs_kwargs(
        model=model, n_graphs=n_graphs, nodes_range=node_sizes, **model_kwargs
    )
    return model, result, time.perf_counter() - start


def _divcurves_chunk(graphs, ts, upsample_to, n_repeats, compute_at_cardinality):
    all_spreads, _ = get_mag_graphs(
        graphs,
        dist_fn,
        ts=ts,
        upsample_to=upsample_to,
        n_repeats=n_repeats,
        compute_at_cardinality=compute_at_cardinality,
    )
    return all_spreads


def _embedding_chunk(graphs, embedding_cls):
    embedding = embedding_cls()
    embedding.fit([g.copy() for g in graphs])
    return embedding.get_embedding()


def _spectral_zoo_chunk(graphs):
    return [est_moment(g, 10000, 4) for g in graphs]


KERNEL_CONFIGS = {
    "kernel_shortest_path": [{"name": "shortest_path"}],
    "kernel_wl_oa": [{"name": "weisfeiler_lehman_optimal_assignment"}],
}
TOPER_FILTRATIONS = ["degree"]


def kernel_compute(graphs_grakel, kernel_name):
    gk = GraphKernel(kernel=KERNEL_CONFIGS[kernel_name], normalize=True)
    return gk.fit_transform(graphs_grakel)


def toper_compute(graphs, filtrations=TOPER_FILTRATIONS, n_jobs=1, num_segm=1e9):
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(filtrations))) as executor:
            futures = {
                executor.submit(toper.main, graphs, [filt], num_segm): filt for filt in filtrations
            }
            filt_results = {futures[future]: future.result() for future in futures}
    else:
        filt_results = {filt: toper.main(graphs, [filt], num_segm) for filt in filtrations}

    all_sub = []
    for filt in filtrations:
        F = filt_results[filt]
        key = "forricci" if filt == "fricci" else "olricci" if filt == "oricci" else filt
        array_sub = np.array(F[f"{key}_sub"])
        if not np.isnan(array_sub).any():
            all_sub.append(array_sub)
        else:
            print(f"toper filtration {filt} produced nan, skipping it")
    if not all_sub:
        return None
    return np.concatenate(all_sub, axis=1)


def ph_vietoris_rips_compute(graphs, k=2):
    return obtain_input_from_graphs(graphs, filtration_fn_name="vietoris_rips", k=k)


def run_timed(compute_fn, n_runs):
    run_times = []
    result = None
    for _ in range(n_runs):
        start = time.perf_counter()
        result = compute_fn()
        run_times.append(time.perf_counter() - start)
    return result, run_times


def record_timing(name, res_dict, timings, run_times):
    timings[name] = float(np.mean(run_times))
    res_dict[name]["time_mean"] = float(np.mean(run_times))
    res_dict[name]["time_std"] = float(np.std(run_times))
    n_runs = len(run_times)
    if n_runs > 1:
        print(
            f"  {name}: {n_runs} run(s), "
            f"{format_duration(res_dict[name]['time_mean'])} ± "
            f"{format_duration(res_dict[name]['time_std'])}"
        )
    else:
        print(f"  {name} finished in {format_duration(res_dict[name]['time_mean'])}")


def kernel_to_distance(K, y_local, n_nodes_local, class_labels, method_label):
    bad = np.isnan(np.diag(K))
    if bad.any():
        bad_idx = np.flatnonzero(bad)
        K = np.nan_to_num(K, nan=0.0)
    D = -K + K.max()
    np.fill_diagonal(D, 0.0)
    remaining_nan = int(np.isnan(D).sum())
    if remaining_nan:
        raise ValueError(
            f"{method_label}: {remaining_nan} NaN entries remain in the distance matrix "
        )
    return D


def fit_and_score(name, res_dict, X, y, n_nodes, precomputed=False):
    res_dict[name] = {"n_graphs": len(y)}
    if precomputed:
        silh_mean, silh_std = compute_silhouette_score_resamples(X, y)
        #ari_agglo_mean, ari_agglo_std = compute_rand_agglo(X, y, metric="precomputed")
        res_dict[name]["silh_mean"], res_dict[name]["silh_std"] = silh_mean, silh_std
        #res_dict[name]["ari_agglo_mean"] = ari_agglo_mean
        #res_dict[name]["ari_agglo_std"] = ari_agglo_std
    else:
        all_clustering_results_euclidean(name, res_dict, X, y)

    outer_k = max(2, min(10, len(np.unique(n_nodes))))
    test_accs = knn_classifier(
        X, np.array(y), precomputed=precomputed, groups=np.array(n_nodes), outer_k=outer_k
    )
    res_dict[name]["acc_mean"] = np.mean(test_accs)
    res_dict[name]["acc_std"] = np.std(test_accs)


if __name__ == "__main__":
    script_start = time.perf_counter()
    np.random.seed(43)
    random.seed(314)

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--output_dir",
        type=str,
        default="./simulations/results/sim_scale/",
    )
    argparser.add_argument(
        "--node_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Explicit graph sizes to sample per model, e.g. --node_sizes 1000 ",
    )
    argparser.add_argument(
        "--node_range",
        type=int,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        default=None,
    )
    argparser.add_argument(
        "--node_powers",
        type=int,
        nargs=2,
        metavar=("START", "STOP"),
        default=None,
    )
    argparser.add_argument(
        "--ngraphs",
        type=int,
        default=2,
        help="Number of graphs generated per model per node size.",
    )
    argparser.add_argument(
        "--n_curve_checkpoints",
        type=int,
        default=None,
    )
    argparser.add_argument(
        "--curve_checkpoints",
        type=int,
        nargs="+",
        default=None,
    )
    argparser.add_argument("--n_repeats", type=int, default=3)
    argparser.add_argument(
        "--n_timing_runs",
        type=int,
        default=1,
    )
    argparser.add_argument(
        "--large_graph_cutoff",
        type=int,
        default=2000,
        help="Max node count for methods with worse-than-quadratic complexity "
    )
    argparser.add_argument(
        "--toper_num_segm",
        type=int,
        default=5000,
        help="Max distinct filtration thresholds per graph for TopER"
    )
    argparser.add_argument("--methods", type=str, nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    argparser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="CPU worker processes for graph generation / diversity curves"
    )
    args = argparser.parse_args()
    n_jobs = resolve_n_jobs(args.n_jobs)

    n_range_opts_given = sum(
        opt is not None for opt in (args.node_sizes, args.node_range, args.node_powers)
    )
    if n_range_opts_given > 1:
        argparser.error("--node_sizes, --node_range and --node_powers are mutually exclusive")
    elif args.node_range is not None:
        start, stop, step = args.node_range
        node_sizes = list(range(start, stop, step))
        if not node_sizes:
            argparser.error(f"--node_range {start} {stop} {step} produced no sizes")
    elif args.node_powers is not None:
        start, stop = args.node_powers
        node_sizes = [2**i for i in range(start, stop + 1)]
    elif args.node_sizes is not None:
        node_sizes = args.node_sizes
    else:
        node_sizes = [1000, 2500, 5000, 7500, 10000]

    if args.n_curve_checkpoints is not None and args.curve_checkpoints is not None:
        argparser.error("--n_curve_checkpoints and --curve_checkpoints are mutually exclusive")
    elif args.curve_checkpoints is not None:
        # Fixed, explicit cardinalities - independent of node size (e.g. the
        # paper's I = {10, 30, 50, 70, 90}), computed once below.
        fixed_curve_checkpoints = np.array(sorted(set(args.curve_checkpoints)))
    else:
        fixed_curve_checkpoints = None
        n_curve_checkpoints = args.n_curve_checkpoints if args.n_curve_checkpoints is not None else 30

    if (
        fixed_curve_checkpoints is not None
        and "divcurves" in args.methods
        and min(node_sizes) < fixed_curve_checkpoints.min()
    ):
        argparser.error(
            f"--curve_checkpoints' smallest value ({fixed_curve_checkpoints.min()}) is "
            f"larger than the smallest requested node size ({min(node_sizes)}). Graphs "
        )

    results_dir = args.output_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    model_params = {
        "ER": {"ER_probs": [0.75], "ER_nps": None},
        "RPG": {"RPG_probs": [[0.9, 0.1]], "RPG_props": [1 / 3, 1 / 3]},
        "SBM": {"SBM_probs": [0.8, 0.05], "SBM_blocks": [[1 / 4, 1 / 4, 1 / 4]]},
        "RG": {"RG_probs": [0.25]},
    }

    # generate the graphs of all models and create a label per model
    timings = {}
    G_nx, Gs, n_nodes, y = [], [], [], []
    class_labels = {}
    generation_start = time.perf_counter()

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(model_params))) as executor:
            futures = [
                executor.submit(
                    _generate_one_model, model, model_params[model], args.ngraphs, node_sizes
                )
                for model in model_params
            ]
            gen_results = {r[0]: r for r in (f.result() for f in futures)}
    else:
        gen_results = {
            model: _generate_one_model(model, model_params[model], args.ngraphs, node_sizes)
            for model in model_params
        }

    for i_m, model in enumerate(model_params):
        _, (G_nx_model, Gs_model, y_model, y_probs_model, n_nodes_model), model_elapsed = gen_results[
            model
        ]
        timings[f"generate_{model}"] = model_elapsed
        print(f"  generated {len(G_nx_model)} {model} graphs in {format_duration(model_elapsed)}")
        G_nx.extend(G_nx_model)
        Gs.extend(Gs_model)
        n_nodes.extend(n_nodes_model)
        y.extend([i_m] * len(G_nx_model))
        class_labels[i_m] = model

    timings["graph_generation_total"] = time.perf_counter() - generation_start
    n_nodes = np.array(n_nodes)
    upsample_to = int(n_nodes.max())
    print(
        f"Generated {len(G_nx)} graphs total in {format_duration(timings['graph_generation_total'])}, "
        f"{dict(zip(*np.unique(n_nodes, return_counts=True)))}"
    )

    small_mask = n_nodes <= args.large_graph_cutoff
    if small_mask.sum() < len(G_nx):
        print(
            f"{len(G_nx) - small_mask.sum()}/{len(G_nx)} graphs exceed "
        )
    G_nx_small = [g for g, keep in zip(G_nx, small_mask) if keep]
    Gs_small = [g for g, keep in zip(Gs, small_mask) if keep]
    y_small = [v for v, keep in zip(y, small_mask) if keep]
    n_nodes_small = n_nodes[small_mask]

    res_dict = {}

    ##################################
    ### diversity curves (checkpointed magnitude) ###
    ##################################
    if "divcurves" in args.methods:
        print(f"Computing diversity curves ({n_jobs} worker(s), {args.n_timing_runs} timing run(s))...")
        if fixed_curve_checkpoints is not None:
            # Fixed checkpoints are independent of dataset size: e.g. with
            # I = {10, 30, 50, 70, 90}, nmax=90 - graphs already bigger than
            compute_at_cardinality = fixed_curve_checkpoints
            divcurves_upsample_to = int(fixed_curve_checkpoints.max())
        else:
            compute_at_cardinality = np.unique(
                np.geomspace(1, upsample_to, num=n_curve_checkpoints).astype(int)
            )
            divcurves_upsample_to = upsample_to

        def compute_divcurves():
            chunks, chunk_results = parallel_map_chunks(
                _divcurves_chunk,
                G_nx,
                n_jobs,
                [1],
                divcurves_upsample_to,
                args.n_repeats,
                compute_at_cardinality,
            )
            offsets = np.cumsum([0] + [len(c) for c in chunks[:-1]])
            spread_dfs = []
            for offset, spread_df_chunk in zip(offsets, chunk_results):
                spread_df_chunk = spread_df_chunk.copy()
                spread_df_chunk["graph"] = spread_df_chunk["graph"] + offset
                spread_dfs.append(spread_df_chunk)
            all_spreads = pd.concat(spread_dfs, axis=0)
            mag_funs = get_funs_df(all_spreads)
            present_indices = list(mag_funs.index)
            return np.array(mag_funs.interpolate(method="linear", axis="columns")), present_indices

        (representation, present_indices), run_times = run_timed(compute_divcurves, args.n_timing_runs)

        if len(present_indices) < len(G_nx):
            missing = sorted(set(range(len(G_nx))) - set(present_indices))
            print(
                f"  WARNING: {len(missing)}/{len(G_nx)} graphs produced no diversity-curve "
            )
            y_divcurves = [y[i] for i in present_indices]
            n_nodes_divcurves = n_nodes[present_indices]
        else:
            y_divcurves = y
            n_nodes_divcurves = n_nodes

        fit_and_score("divcurves", res_dict, representation, y_divcurves, n_nodes_divcurves)
        record_timing("divcurves", res_dict, timings, run_times)

    ### baselines ###

    if "netlsd" in args.methods or "feather" in args.methods:
        embedding_classes = {}
        if "netlsd" in args.methods:
            embedding_classes["netlsd"] = NetLSD
        if "feather" in args.methods:
            embedding_classes["feather"] = FeatherGraph
        for name, embedding_cls in embedding_classes.items():
            print(f"Computing {name} ({n_jobs} worker(s), {args.n_timing_runs} timing run(s))...")

            def compute_embedding(embedding_cls=embedding_cls):
                _, chunk_results = parallel_map_chunks(_embedding_chunk, G_nx, n_jobs, embedding_cls)
                return np.vstack(chunk_results)

            representation, run_times = run_timed(compute_embedding, args.n_timing_runs)
            fit_and_score(name, res_dict, representation, y, n_nodes)
            record_timing(name, res_dict, timings, run_times)

    if "spectral_zoo" in args.methods:
        print(f"Computing spectral zoo ({n_jobs} worker(s), {args.n_timing_runs} timing run(s))...")

        def compute_spectral_zoo():
            _, chunk_results = parallel_map_chunks(_spectral_zoo_chunk, G_nx, n_jobs)
            moments = np.array([m for chunk in chunk_results for m in chunk])
            return moments[:, ~np.all(moments == 0, axis=0)]

        representation, run_times = run_timed(compute_spectral_zoo, args.n_timing_runs)
        fit_and_score("spectral_zoo", res_dict, representation, y, n_nodes)
        record_timing("spectral_zoo", res_dict, timings, run_times)

    if "kernel_shortest_path" in args.methods or "kernel_wl_oa" in args.methods:
        for name in ["kernel_shortest_path", "kernel_wl_oa"]:
            if name not in args.methods:
                continue
            print(
                f"Computing {name} on {len(Gs_small)} graphs (<= {args.large_graph_cutoff} nodes, "
                f"{args.n_timing_runs} timing run(s))..."
            )
            try:
                K, run_times = run_timed(lambda: kernel_compute(Gs_small, name), args.n_timing_runs)
            except Exception as e:
                print(f"Kernel {name} failed with error: {e}")
                continue
            D = kernel_to_distance(K, y_small, n_nodes_small, class_labels, name)
            fit_and_score(name, res_dict, D, y_small, n_nodes_small, precomputed=True)
            record_timing(name, res_dict, timings, run_times)

    if "toper" in args.methods:
        print(
            f"Computing TopER on {len(G_nx_small)} graphs (<= {args.large_graph_cutoff} nodes, "
            f"{n_jobs} worker(s), {args.n_timing_runs} timing run(s))..."
        )
        all_sub, run_times = run_timed(
            lambda: toper_compute(G_nx_small, n_jobs=n_jobs, num_segm=args.toper_num_segm),
            args.n_timing_runs,
        )
        if all_sub is not None:
            fit_and_score("toper", res_dict, all_sub, y_small, n_nodes_small)
            record_timing("toper", res_dict, timings, run_times)

    if "ph_vietoris_rips" in args.methods:
        print(
            f"Computing PH Vietoris-Rips on {len(G_nx_small)} graphs "
            f"(<= {args.large_graph_cutoff} nodes, {args.n_timing_runs} timing run(s))..."
        )
        try:
            array_ph, run_times = run_timed(
                lambda: ph_vietoris_rips_compute(G_nx_small), args.n_timing_runs
            )
            fit_and_score("ph_vietoris_rips", res_dict, array_ph, y_small, n_nodes_small)
            record_timing("ph_vietoris_rips", res_dict, timings, run_times)
        except MemoryError as e:
            print(f"PH Vietoris-Rips ran out of memory: {e}")

    timings["total"] = time.perf_counter() - script_start

    # creating dataframes and saving results
    res_df = pd.DataFrame.from_dict(res_dict, orient="index")
    res_df.to_csv(f"{results_dir}/res_df.csv")
    pd.DataFrame({"n_nodes": n_nodes}).to_csv(f"{results_dir}/n_nodes.csv")
    pd.DataFrame.from_dict(timings, orient="index", columns=["seconds"]).to_csv(
        f"{results_dir}/timings.csv"
    )

    summary_rows = []
    for key, name in DISPLAY_NAMES.items():
        if key not in res_dict:
            continue
        r = res_dict[key]
        if "time_mean" in r:
            time_str = format_duration(r["time_mean"])
            if args.n_timing_runs > 1:
                time_str += f" ± {format_duration(r['time_std'])}"
        else:
            time_str = "-"
        summary_rows.append(
            {
                "Method": name,
                "Accuracy": f"{r.get('acc_mean', float('nan')):.2f} ± {r.get('acc_std', float('nan')):.2f}",
                "Silhouette Score": f"{r.get('silh_mean', float('nan')):.2f} ± {r.get('silh_std', float('nan')):.2f}",
                "n_graphs": r.get("n_graphs", len(G_nx_small) if key in EXPENSIVE_METHODS else len(G_nx)),
                "Time": time_str,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{results_dir}/summary_table.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"\nTotal runtime: {format_duration(timings['total'])}")
