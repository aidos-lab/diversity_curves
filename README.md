# Diversity Curves for Graph Representation Learning

This repository provides some code examples for the computation of Diversity
Curves. In particular, we provide the code to reproduce the experiments on
distinguishing and clustering graphs from different distributions. 


## Dependencies

Our dependencies are managed with `uv`. To get started, we recommend you to
install `uv` and then execute the following steps:
1. Clone this repository locally.
2. Specify which Python version to use via `uv venv --python 3.10`.
3. Run the command `uv sync` to install all neccessairy dependencies.

## Experiments

The experiment to distinguish graph distributions is in the script 
`sim_distributions.py`, and the experiment for distinguishing parameter
choices is in the scripts `sim_params.py`.

To reproduce the first experiment, run the script with the command 
`uv run -m simulations.sim_distributions`.
This will use the configurations as described in the paper and create a 
.csv file with all results and the PCA visualisations in `./simulations/results/sim_distributions`.
After running the script, you can run the notebook `sim_distributions.ipynb`
to reproduce the table with the full results.

To reproduce the experiment distinguishing parameter choices, run the command
`uv run -m simulations.sim_params`.
Again, all the configurations will be used as described in the paper. 
A .csv file will be created with all results. For each random graph model a PCA plot of the
diversity curves will also be created. After running the script, you can run the notebook
`sim_params.ipynb` to reproduce the full tables and the plot of the silhouette 
scores. 


