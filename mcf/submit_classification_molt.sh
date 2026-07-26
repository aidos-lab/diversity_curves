#!/bin/bash
#SBATCH --job-name=classify
#SBATCH --output=logs/classify_%A_%a.out
#SBATCH --error=logs/classify_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --qos=cpu_normal
#SBATCH --partition=cpu_p
#SBATCH --mem=64G
#SBATCH --array=0-1
# 2 methods (see METHODS below) -> array indices 0-1.
# If you add/remove methods below, update both the array bound above and the
# METHODS list to match.

set -euo pipefail

DATASET=${1:-MOLT-4}

METHODS=(
    kernel_shortest_path
    kernel_core_framework
)

METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

mkdir -p logs

echo "Classifying '$METHOD' on dataset '$DATASET' (array task $SLURM_ARRAY_TASK_ID)"
uv run python classify_embeddings.py --dataset "$DATASET" --names "$METHOD"
