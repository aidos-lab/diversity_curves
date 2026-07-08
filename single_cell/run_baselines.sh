
method_names=(
    "spread"
    "netlsd"
    "toper"
    "graph2"
    "feather"
    "spectral_zoo"
    "kernel"
)

for method in "${method_names[@]}"; do
    echo "Running method: ${method}"
    uv run ./baselines_sc.py --method "${method}"
done
