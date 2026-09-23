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
#SBATCH --array=0-3
#SBATCH --chdir=/HOME/haeuserm/diversity_curves

CONFIGS=(
    # "--nrange 10 200 --nsteps 5 --ngraphs 1 --max_compute_card 200 --step_compute_card 4 --output_dir ./simulations/results_cluster/sim_distributions/nrange10_200nsteps_5ngraphs_1/max_comcard_200step_compute_card_4/"
    # "--nrange 10 200 --nsteps 5 --ngraphs 1 --max_compute_card 200 --step_compute_card 3 --output_dir ./simulations/results_cluster/sim_distributions/nrange10_200nsteps_5ngraphs_1/max_comcard_200step_compute_card_3/"
    # "--nrange 10 200 --nsteps 5 --ngraphs 1 --max_compute_card 200 --step_compute_card 2 --output_dir ./simulations/results_cluster/sim_distributions/nrange10_200nsteps_5ngraphs_1/max_comcard_200step_compute_card_2/"
#  "--nrange 10 30 100 130 200 230 --nsteps 1 --ngraphs 1 --max_compute_card 230 --step_compute_card 5 --output_dir ./simulations/results_cluster/sim_distributions/nrange10_30_100_130_200_230nsteps_1ngraphs_1/max_comcard_230step_compute_card_5/ "
#  "--nrange 10 30 100 130 200 230 --nsteps 1 --ngraphs 2 --max_compute_card 230 --step_compute_card 4 --output_dir ./simulations/results_cluster/sim_distributions/nrange10_30_100_130_200_230nsteps_1ngraphs_2/max_comcard_230step_compute_card_4/"
#  "--nrange 10 30 100 130 200 230 --nsteps 1 --ngraphs 1 --max_compute_card 230 --step_compute_card 3 --output_dir ./simulations/results_cluster/sim_distributions/nrange10_30_100_130_200_230nsteps_1ngraphs_1/max_comcard_230step_compute_card_3/ "
#  "--nrange 10 30 100 130 200 230 --nsteps 1 --ngraphs 2 --max_compute_card 230 --step_compute_card 2 --output_dir ./simulations/results_cluster/sim_distributions/nrange10_30_100_130_200_230nsteps_1ngraphs_2/max_comcard_230step_compute_card_2/"
 "--nrange 10 70 70 140 140 200 --nsteps 1 3 5 --ngraphs 1 --max_compute_card 200 --step_compute_card 2 --output_dir ./simulations/results_cluster/sim_distributions/nrange10_70_70_140_140_200nsteps_1_3_5ngraphs_1/max_comcard_200step_compute_card_2/ "
 "--nrange 10 70 70 140 140 200 --nsteps 1 5 10 --ngraphs 1 --max_compute_card 200 --step_compute_card 2 --output_dir ./simulations/results_cluster/sim_distributions/nrange10_70_70_140_140_200nsteps_1_5_10ngraphs_1/max_comcard_200step_compute_card_2/ "
 "--nrange 10 40 40 100 100 200 --nsteps 1 3 5 --ngraphs 1 --max_compute_card 200 --step_compute_card 2 --output_dir ./simulations/results_cluster/sim_distributions/nrange10_40_40_100_100_200nsteps_1_3_5ngraphs_1/max_comcard_200step_compute_card_2/ "
 "--nrange 10 40 40 100 100 200 --nsteps 1 5 10 --ngraphs 2 --max_compute_card 200 --step_compute_card 2 --output_dir ./simulations/results_cluster/sim_distributions/nrange10_40_40_100_100_200nsteps_1_5_10ngraphs_2/max_comcard_200step_compute_card_2/ "
#  "--nrange 10 30 100 130 200 230 300 330 --nsteps 1 --ngraph 1 --max_compute_card 330 --step_compute_card 5 --output_dir ./simulations/results_cluster/sim_distributions/nrange10_30_100_130_200_230_300_330nsteps_1ngraphs_1/max_comcard_150step_compute_card_5/"   
)

task_index=$((SLURM_ARRAY_TASK_ID))
config=${CONFIGS[$task_index]}

echo "Display the host name Start"
pwd; hostname; date

echo "Just checking the version of Python on the server"
python --version

echo "task ${task_index}, sim_distributions with ${config}."

uv run -m simulations.sim_distributions $config


echo "Display the host name End"
pwd; hostname; date