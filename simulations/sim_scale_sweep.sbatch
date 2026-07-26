#!/bin/bash -l
#SBATCH --job-name=sim_scale
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --nodelist=diufrd201
#SBATCH --partition=CPU

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CONFIGS=(
    "--node_powers 3 8 --methods divcurves toper feather netlsd spectral_zoo --curve_checkpoints 8 16 32 64 128 --ngraphs 10 --n_timing_runs 2 --large_graph_cutoff 2000 --toper_num_segm 1000000000 --output_dir simulations/results/sim_scale/pow08"
    "--node_powers 3 9 --methods divcurves toper feather netlsd spectral_zoo --curve_checkpoints 8 32 128 256  --ngraphs 10 --n_timing_runs 2 --large_graph_cutoff 2000 --toper_num_segm 1000000000 --output_dir simulations/results/sim_scale/pow10"
    "--node_powers 3 10 --methods divcurves toper feather netlsd spectral_zoo --curve_checkpoints 8 32 128 256  --ngraphs 10 --n_timing_runs 2 --large_graph_cutoff 2000 --toper_num_segm 1000000000 --output_dir simulations/results/sim_scale/pow10"
)

if (( SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]} )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} has no matching entry in CONFIGS (len=${#CONFIGS[@]})" >&2
    exit 1
fi
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"


echo "task ${SLURM_ARRAY_TASK_ID} on $(hostname), ${SLURM_CPUS_PER_TASK} cpus:"
echo "  uv run -m simulations.sim_scale ${CONFIG} --n_jobs ${SLURM_CPUS_PER_TASK}"

uv run -m simulations.sim_scale $CONFIG --n_jobs "$SLURM_CPUS_PER_TASK"
