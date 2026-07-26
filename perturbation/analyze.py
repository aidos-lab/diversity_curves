"""
    uv run python analyze.py --root ./results --embedding mag
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="./results")
parser.add_argument("--embedding", type=str, default="mag")
parser.add_argument("--diff_baseline", action="store_true", default=False)
parser.add_argument("--output_dir", type=str, default="./results/summary")


def build_drift_df(root, embedding, diff_baseline):
    rows = {"dataset": [], "perturbation_type": [], "seed": [], "degree": [], "value": []}

    for dataset in sorted(os.listdir(root)):
        emb_dir = os.path.join(root, dataset, embedding)
        if not os.path.isdir(emb_dir):
            continue

        avg_baseline = None
        if diff_baseline:
            baseline_path = os.path.join(emb_dir, "baseline.npy")
            avg_baseline = np.mean(np.load(baseline_path), axis=0)

        for pert_type in sorted(os.listdir(emb_dir)):
            pert_dir = os.path.join(emb_dir, pert_type)
            if not os.path.isdir(pert_dir):
                continue

            for seed in sorted(os.listdir(pert_dir)):
                seed_dir = os.path.join(pert_dir, seed)
                if not os.path.isdir(seed_dir):
                    continue

                for pert_file in sorted(os.listdir(seed_dir)):
                    degree = int(pert_file.split(".")[0].split("_")[-1])
                    data = np.load(os.path.join(seed_dir, pert_file))
                    avg_graph = np.mean(data, axis=0)
                    if avg_baseline is not None:
                        avg_graph = avg_graph - avg_baseline

                    rows["dataset"].append(dataset)
                    rows["perturbation_type"].append(pert_type)
                    rows["seed"].append(seed)
                    rows["degree"].append(degree)
                    rows["value"].append(np.linalg.norm(avg_graph))

    return pd.DataFrame(rows)


def compute_correlations(drift_df):
    rows = {"dataset": [], "perturbation_type": [], "seed": [], "correlation": []}
    for dataset in drift_df["dataset"].unique():
        for pert_type in drift_df["perturbation_type"].unique():
            for seed in drift_df["seed"].unique():
                sub = drift_df[
                    (drift_df["dataset"] == dataset)
                    & (drift_df["perturbation_type"] == pert_type)
                    & (drift_df["seed"] == seed)
                ]
                if sub.empty:
                    continue
                corr = spearmanr(sub["value"], sub["degree"], nan_policy="omit").statistic
                rows["dataset"].append(dataset)
                rows["perturbation_type"].append(pert_type)
                rows["seed"].append(seed)
                rows["correlation"].append(corr)
    return pd.DataFrame(rows)


def plot_drift_curves(drift_df, output_dir):
    mean_df = (
        drift_df.groupby(["dataset", "perturbation_type", "degree"])["value"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    datasets = sorted(mean_df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        sub = mean_df[mean_df["dataset"] == dataset]
        sns.lineplot(data=sub, x="degree", y="mean", hue="perturbation_type", marker="o", ax=ax)
        ax.set_title(dataset)
        ax.set_xlabel("Perturbation degree (%)")
        ax.set_ylabel("Embedding drift from baseline")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "drift_curves.pdf"))
    plt.close(fig)


def plot_correlation_heatmap(corr_df, output_dir):
    summary = corr_df.groupby(["dataset", "perturbation_type"])["correlation"].mean().reset_index()
    matrix = summary.pivot(index="dataset", columns="perturbation_type", values="correlation")
    plt.figure(figsize=(1.5 * matrix.shape[1] + 2, 1.2 * matrix.shape[0] + 2))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0)
    plt.title("Spearman correlation: embedding drift vs. perturbation degree")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.pdf"))
    plt.close()


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    drift_df = build_drift_df(args.root, args.embedding, args.diff_baseline)
    if drift_df.empty:
        raise SystemExit(f"No precomputed embeddings found under {args.root}")

    corr_df = compute_correlations(drift_df)

    drift_df.to_csv(os.path.join(args.output_dir, "drift.csv"), index=False)
    corr_df.to_csv(os.path.join(args.output_dir, "correlations.csv"), index=False)

    plot_drift_curves(drift_df, args.output_dir)
    plot_correlation_heatmap(corr_df, args.output_dir)

    print(corr_df.groupby(["dataset", "perturbation_type"])["correlation"].agg(["mean", "std"]))
