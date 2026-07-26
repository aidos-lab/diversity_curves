#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

DATASET="${1:-sbm}"
EMBEDDING="${2:-mag}"
ROOT="./results"
PERTURB_FUNCS=(add_edge remove_edge rewire_edge swap_edge)
PERTURB_DEGREES=(10 20 30 40 50 60 70 80 90 100)

for pert_fn in "${PERTURB_FUNCS[@]}"; do
    for pert_degree in "${PERTURB_DEGREES[@]}"; do
        uv run python run_perturbation.py \
            -d "$DATASET" -e "$EMBEDDING" -p \
            --perturb_func "$pert_fn" --perturb_degree "$pert_degree" \
            --root "$ROOT"
    done
done
