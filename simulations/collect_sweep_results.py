
import argparse
import re
from pathlib import Path

import pandas as pd

METHOD_ORDER = [
    "Diversity Curves (ours)",
    "Kernel WL Opt. Assig.",
    "Kernel Shortest Path",
    "TopER",
    "NetLSD",
    "FEATHER",
    "Spectral Zoo",
    "PH Vietoris-Rips",
]


def config_sort_key(name):
    m = re.match(r"^(\d+(?:\.\d+)?)([KMG]?)$", name, re.IGNORECASE)
    if not m:
        return (1, name, 0.0)
    value, suffix = m.groups()
    multiplier = {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}[suffix.upper()]
    return (0, "", float(value) * multiplier)


def method_sort_key(name):
    return (METHOD_ORDER.index(name) if name in METHOD_ORDER else len(METHOD_ORDER), name)


def to_markdown_table(df):
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def collect(results_dir):
    results_dir = Path(results_dir)
    rows = []
    missing = []
    for config_dir in sorted(results_dir.iterdir(), key=lambda p: config_sort_key(p.name)):
        if not config_dir.is_dir():
            continue
        csv_path = config_dir / "summary_table.csv"
        if not csv_path.exists():
            missing.append(config_dir.name)
            continue
        df = pd.read_csv(csv_path)
        df.insert(0, "Config", config_dir.name)
        rows.append(df)

    if not rows:
        raise SystemExit(f"No summary_table.csv found under {results_dir}")

    combined = pd.concat(rows, axis=0, ignore_index=True)
    combined["_method_sort"] = combined["Method"].map(method_sort_key)
    combined = combined.sort_values(["_method_sort", "Config"]).drop(columns="_method_sort")
    return combined, missing


def pivot(combined, value_col, config_order):
    table = combined.pivot_table(
        index="Method", columns="Config", values=value_col, aggfunc="first"
    )
    table = table.reindex(index=sorted(table.index, key=method_sort_key))
    table = table.reindex(columns=[c for c in config_order if c in table.columns])
    table = table.reset_index()
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="simulations/results/sim_scale")
    parser.add_argument(
        "--output", type=str, default=None, help="Markdown file to write; defaults to stdout."
    )
    args = parser.parse_args()

    combined, missing = collect(args.results_dir)
    config_order = sorted(combined["Config"].unique(), key=config_sort_key)

    sections = []
    sections.append("# sim_scale.py sweep results\n")
    if missing:
        sections.append(
            "> **Note:** no `summary_table.csv` found yet for: "
            + ", ".join(missing)
            + " (run still in progress, or that sweep point failed)\n"
        )

    sections.append("## All results\n")
    sections.append(to_markdown_table(combined))

    for value_col in ["Accuracy", "Silhouette Score", "Time", "n_graphs"]:
        sections.append(f"\n\n## {value_col} by method × config\n")
        sections.append(to_markdown_table(pivot(combined, value_col, config_order)))

    report = "\n".join(sections) + "\n"

    if args.output:
        Path(args.output).write_text(report)
        print(f"Wrote {args.output}")
    else:
        print(report)
