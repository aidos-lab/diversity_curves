#!/bin/bash
#SBATCH --job-name=divcurves_simulations
#SBATCH --time=12:00:00
#SBATCH --output=logs/output_%j_%A.txt
#SBATCH --error=logs/error_%j_%A.txt
#SBATCH --ntasks=1
#SBATCH --partition=GPU
#SBATCH --nodelist=diufrd204
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nadja.haeusermann@unifr.ch
#SBATCH --mem=32G
#SBATCH --array=0-4

CONFIGS=(
 "--nrange 10 200 --nsteps 5 --ngraphs 1 --max_compute_card 100 --step_compute_card 5 --output_dir /simulations/results_cluster/sim_distributions/nrange10_200nsteps_5/max_comcard_100step_compute_card_5/ "
 "--nrange 10 200 --nsteps 10 --ngraphs 2 --max_compute_card 100 --step_compute_card 5 --output_dir /simulations/results_cluster/sim_distributions/nrange10_300nsteps_10/max_comcard_100step_compute_card_5/"
 "--nrange 10 200 --nsteps 5 --ngraphs 1 --max_compute_card 150 --step_compute_card 5 --output_dir /simulations/results_cluster/sim_distributions/nrange10_200nsteps_5/max_comcard_100step_compute_card_5/ "
 "--nrange 10 200 --nsteps 10 --ngraphs 2 --max_compute_card 150 --step_compute_card 5 --output_dir /simulations/results_cluster/sim_distributions/nrange10_300nsteps_10/max_comcard_100step_compute_card_5/"
 "--nrange 10 300 --nsteps 10 --ngraph 1 --max_compute_card 100 --step_compute_card 5 --output_dir /simulations/results_cluster/sim_distributions/nrange10_300nsteps_10/max_comcard_100step_compute_card_5/"   
)

task_index=$((SLURM_ARRAY_TASK_ID))
config=${CONFIGS[$task_index]}

echo "Display the host name Start"
pwd; hostname; date

echo "Just checking the version of Python on the server"
python --version

echo "task ${task_index}, sim_distributions with ${config}."

uv run -m simulations.sim_distributions $CONFIG


echo "Display the host name End"
pwd; hostname; date