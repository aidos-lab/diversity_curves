"""
    uv run python run_perturbation.py -d sbm -e mag -p \
        --perturb_func remove_edge --perturb_degree 50 --root ./results
"""

import argparse
import os
import random

import numpy as np
import tqdm
from torch_geometric.utils import to_networkx

from datasets import SpectreGraphDataset
from perturb import PERTURBATION_FUNCTIONS, embed

parser = argparse.ArgumentParser(
    prog="Perturbation robustness precomputation",
    description="Compute baseline and perturbed embeddings for a graph dataset",
)

parser.add_argument("--root", type=str, default="./results")
parser.add_argument(
    "-d",
    "--dataset",
    choices=["sbm", "planar", "comm20", "ego", "lobster"],
    required=True,
)
parser.add_argument("-p", "--perturb", action="store_true", default=False)
parser.add_argument("-perturb_degree", "--perturb_degree", type=int)
parser.add_argument("--perturb_func", choices=list(PERTURBATION_FUNCTIONS.keys()))
parser.add_argument("-e", "--embedding", choices=["mag", "color"], required=True)
parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])


def load_dataset(name):
    """Loads a dataset as a list of undirected NetworkX graphs."""
    if name in ("sbm", "planar", "comm20"):
        ds = SpectreGraphDataset(dataset_name=name, split="train", root=f"./data/{name}")
        return [to_networkx(G, to_undirected=True) for G in ds]
    elif name == "ego":
        from polygraph.datasets import SmallEgoGraphDataset

        return list(SmallEgoGraphDataset(split="train").to_nx())
    elif name == "lobster":
        from polygraph.datasets import LobsterGraphDataset

        return list(LobsterGraphDataset(split="train").to_nx())
    raise ValueError(f"Unknown dataset {name}")


if __name__ == "__main__":
    args = parser.parse_args()

    save_dir = os.path.join(args.root, args.dataset, args.embedding)
    os.makedirs(save_dir, exist_ok=True)

    graph_list_dataset = load_dataset(args.dataset)

    if args.dataset == "sbm":
        graph_list_dataset = sorted(graph_list_dataset, key=lambda G: G.number_of_nodes())[:20]
        print([G.number_of_nodes() for G in graph_list_dataset])

    # A `baseline` embedding is needed to compare the perturbations against.
    file_embedd_base = os.path.join(save_dir, "baseline.npy")
    if os.path.exists(file_embedd_base):
        print("Skipping base embedding calculation...")
    else:
        print("Calculating base embedding....")
        G_list_embedded = embed(G_list=graph_list_dataset, embedding_type=args.embedding)
        with open(file_embedd_base, "wb") as f:
            np.save(file=f, arr=G_list_embedded)

    if args.perturb:
        assert args.perturb_func is not None, "Need to specify a perturb_function"

        save_dir = os.path.join(save_dir, args.perturb_func)
        os.makedirs(save_dir, exist_ok=True)

        for seed in args.seeds:
            random.seed(seed)
            seed_dir = os.path.join(save_dir, str(seed))
            os.makedirs(seed_dir, exist_ok=True)

            G_original = [G.copy() for G in graph_list_dataset]
            pert_fn = PERTURBATION_FUNCTIONS[args.perturb_func]

            print(f"Perturbing with {args.perturb_func} and seed={seed} for {args.perturb_degree}% ...")
            for G in tqdm.tqdm(G_original, desc="G"):
                pert_fn(G, args.perturb_degree)

            print(f"Calculating {args.embedding} embeddings...")
            G_perm_list_embedded = embed(G_list=G_original, embedding_type=args.embedding)
            G_perm_array = np.array(G_perm_list_embedded)

            pert_path = os.path.join(seed_dir, f"pert_{args.perturb_degree}.npy")
            with open(pert_path, "wb") as f:
                np.save(file=f, arr=G_perm_array)
