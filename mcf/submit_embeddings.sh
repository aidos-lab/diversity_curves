#!/bin/bash
#SBATCH --job-name=embeddings
#SBATCH --output=logs/embeddings_%A_%a.out
#SBATCH --error=logs/embeddings_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-16
# 17 methods (see METHODS below) -> array indices 0-16.
# If you add/remove methods in embeddings.py, update both the array bound
# above and the METHODS list below to match.

set -euo pipefail

DATASET=${1:-MCF-7}

METHODS=(
    mag_shortest_path_distance
    mag_diffusion_distance
    netlsd
    graph2
    feather
    kernel_shortest_path
    kernel_pyramid_match
    kernel_neighborhood_hash
    kernel_weisfeiler_lehman
    kernel_core_framework
    kernel_weisfeiler_lehman_optimal_assignment
    toper_degree
    toper_popularity
    toper_fricci
    toper_oricci
    toper_closeness
    spectral_zoo
)

METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

mkdir -p logs

echo "Running method '$METHOD' on dataset '$DATASET' (array task $SLURM_ARRAY_TASK_ID)"
python embeddings.py --dataset "$DATASET" --method "$METHOD"
