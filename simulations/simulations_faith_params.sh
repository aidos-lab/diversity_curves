#!/bin/bash
#SBATCH --job-name=divcurves_simulations
#SBATCH --time=12:00:00
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --ntasks=1
#SBATCH --partition=CPU
#SBATCH --nodelist=diufrd201
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nadja.haeusermann@unifr.ch

echo "Display the host name Start"
pwd; hostname; date

echo "Just checking the version of Python on the server"
python --version

# uv run -m simulations.sim_params --output_dir ./simulations/results/sim_params/nrange10_30_100_130_200_230/ --step_compute_card 5 
uv run -m simulations.sim_params --output_dir ./simulations/results/sim_params/nrange10_30_100_130/ --step_compute_card 5 


echo "Display the host name End"
pwd; hostname; date