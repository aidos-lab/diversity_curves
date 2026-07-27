#!/bin/bash -l
#SBATCH --job-name=sim_scale
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-4
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --nodelist=diufrd201
#SBATCH --partition=CPU

# source /HOME/carrasco/miniconda3/etc/profile.d/conda.sh
#
# conda activate test_new_gpu

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CONFIGS=(
    "--node_powers 4 8 --methods divcurves feather netlsd spectral_zoo --curve_checkpoints 16 30 50 70 90 --ngraphs 5 --n_timing_runs 2 --large_graph_cutoff 2000 --toper_num_segm 1000000000 --output_dir simulations/results/sim_scale/pow8"
    "--node_powers 4 9 --methods divcurves feather netlsd spectral_zoo --curve_checkpoints 16 30 50 70 90  --ngraphs 5 --n_timing_runs 2 --large_graph_cutoff 2000 --toper_num_segm 1000000000 --output_dir simulations/results/sim_scale/pow9"
    "--node_powers 4 10 --methods divcurves feather netlsd spectral_zoo --curve_checkpoints 16 30 50 70 90  --ngraphs 5 --n_timing_runs 2 --large_graph_cutoff 2000 --toper_num_segm 1000000000 --output_dir simulations/results/sim_scale/pow10"
    "--node_powers 4 11 --methods divcurves feather netlsd spectral_zoo --curve_checkpoints 16 30 50 70 90  --ngraphs 5 --n_timing_runs 2 --large_graph_cutoff 2000 --toper_num_segm 1000000000 --output_dir simulations/results/sim_scale/pow11"
    "--node_powers 4 12 --methods divcurves feather netlsd spectral_zoo --curve_checkpoints 16 30 50 70 90  --ngraphs 5 --n_timing_runs 2 --large_graph_cutoff 2000 --toper_num_segm 1000000000 --output_dir simulations/results/sim_scale/pow12"
)

if (( SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]} )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} has no matching entry in CONFIGS (len=${#CONFIGS[@]})" >&2
    exit 1
fi
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"


echo "task ${SLURM_ARRAY_TASK_ID} on $(hostname), ${SLURM_CPUS_PER_TASK} cpus:"
echo "  uv run -m simulations.sim_scale ${CONFIG} --n_jobs ${SLURM_CPUS_PER_TASK}"

uv run -m simulations.sim_scale $CONFIG --n_jobs "$SLURM_CPUS_PER_TASK"
