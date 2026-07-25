import argparse
import glob
import os

import numpy as np
import pandas as pd
from torch_geometric.datasets import TUDataset

from embeddings import KERNEL_NAMES, run_classification

KERNEL_FILE_NAMES = {f"kernel_{k}" for k in KERNEL_NAMES}


def discover_names(results_dir):
    names = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        names.append(os.path.splitext(os.path.basename(path))[0])
    return names


def load_embedding(name, results_dir):
    path = os.path.join(results_dir, f"{name}.csv")
    if name.startswith("mag_funs_"):
        # mag embeddings can drop graphs (e.g. empty ones), so the CSV keeps
        # the original graph index in its first column to realign labels.
        df = pd.read_csv(path, index_col=0)
        return df.values, df.index.to_numpy(), False
    if name in KERNEL_FILE_NAMES:
        # the saved matrix is the raw similarity kernel; classification needs
        # the same distance-like transform applied in embeddings.run_kernel.
        K = pd.read_csv(path).values
        K = -K + K.max()
        return K, None, True
    return pd.read_csv(path).values, None, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run KNN classification on precomputed embedding CSVs."
    )
    parser.add_argument("--dataset", type=str, default="MCF-7")
    parser.add_argument(
        "--names",
        type=str,
        nargs="*",
        default=None,
        help="Embedding names to classify (default: all CSVs found in the results dir)",
    )
    args = parser.parse_args()

    results_dir = f"./results/{args.dataset}/"
    classification_dir = os.path.join(results_dir, "classification") + os.sep
    os.makedirs(classification_dir, exist_ok=True)

    root = "./../datasets/TUDataset"
    pyg_dataset = TUDataset(root=root, name=args.dataset)
    labels = np.array([data.y.item() for data in pyg_dataset])

    names = args.names if args.names else discover_names(results_dir)

    for name in names:
        print(f"Classifying '{name}'...")
        try:
            X, index, precomputed = load_embedding(name, results_dir)
        except FileNotFoundError:
            print(f"  embedding csv for '{name}' not found, skipping")
            continue

        y = labels[index] if index is not None else labels
        try:
            run_classification(name, X, y, precomputed, classification_dir)
        except Exception as e:
            print(f"  classification for '{name}' failed with error: {e}")

        if name.startswith("mag_funs_"):
            scaled_name = f"{name}_scaled"
            print(f"Classifying '{scaled_name}'...")
            try:
                run_classification(
                    scaled_name, X, y, precomputed, classification_dir, scale=True
                )
            except Exception as e:
                print(f"  classification for '{scaled_name}' failed with error: {e}")
