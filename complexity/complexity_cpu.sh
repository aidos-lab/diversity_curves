#!/bin/bash
#SBATCH --job-name=complexity
#SBATCH --time=96:00:00
#SBATCH --output=logs/output_%j.out
#SBATCH --error=logs/error_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=CPU
#SBATCH --nodelist=diufrd201
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nadja.haeusermann@unifr.ch


echo "Display the host name Start"
pwd; hostname; date

echo "Just checking the version of Python on the server"
python --version

# echo "task $((SLURM_ARRAY_TASK_ID)), sim_distributions with ${config}."

uv run -m complexity.complexity --method fast_divcurve

echo "Display the host name End"
pwd; hostname; date