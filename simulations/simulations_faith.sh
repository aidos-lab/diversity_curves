#!/bin/bash
#SBATCH --job-name=divcurves_simulations
#SBATCH --time=1:00:00
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --ntasks=1
#SBATCH --partition=CPU
#SBATCH --nodelist=diufrd204
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nadja.haeusermann@unifr.ch

echo "Display the host name Start"
pwd; hostname; date

echo "Just checking the version of Python on the server"
python --version

uv run -m simulations.sim_distributions --output_dir --output_dir ./simulations/results/sim_distributions/nrange10_30/ --nrange 10 30 --step_compute_card 5 


echo "Display the host name End"
pwd; hostname; date