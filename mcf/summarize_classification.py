import argparse
import glob
import os

import pandas as pd

METRICS = ["test_accuracy", "test_balanced_accuracy", "test_auroc"]


def summarize(dataset):
    classification_dir = f"./results/{dataset}/classification/"
    rows = []
    for path in sorted(glob.glob(os.path.join(classification_dir, "*.csv"))):
        method = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        row = {"method": method}
        for metric in METRICS:
            row[metric] = f"{df[metric].mean():.3f} ± {df[metric].std(ddof=0):.3f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize per-method classification results as mean +/- std."
    )
    parser.add_argument("--dataset", type=str, default="MCF-7")
    args = parser.parse_args()

    summary = summarize(args.dataset)

    out_path = f"./results/{args.dataset}/classification_summary.csv"
    summary.to_csv(out_path)

    print(summary.to_string())
    print(f"\nSaved summary to {out_path}")
