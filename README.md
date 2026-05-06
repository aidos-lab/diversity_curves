# Diversity Curves for Graph Representation Learning

This repository provides some code examples for the computation of Diversity
Curves. In particular, we provide the code to reproduce the simulated graph 
distribution experiments. 

The experiment to distinguish graph distributions is in the script 
`sim_distributions.py`, and the experiment for distinguishing parameter
choices is in the scripts `sim_params.py`.

To reproduce the experiment distinguishing graph distributions, run the script
with the command 
`uv run -m simulations.sim_distributions`.
By running the script like this, all the configurations will be used as 
described in the paper. This will create a .csv file with all results and
create the PCA visualisations in `./simulations/results/sim_distributions`.
After running the script, you can run the notebook `sim_distributions.ipynb`,
in which the table with the full results are created.

To reproduce the experiment distinguishing parameter choices, run the command
`uv run -m simulations.sim_params`.
Again, By running the script like this, all the configurations will be used as 
described in the paper. Running the script will create a .csv file with all
results and for each random graph model it will create the PCA plot of the
diversity curves. After running the script, you can run the notebook
`sim_params.ipynb` to create the full tables and the plot of the silhouette 
scores. 

## Dependencies

Our dependencies are managed with uv. To get started, we recommend you to
install `uv` and then execute the following steps:
1. Clone this repository locally.
2. Run the command `uv sync` to install all neccessairy dependencies.

