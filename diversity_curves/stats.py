import numpy as np
import pandas as pd


def two_sample_test_matrix(
    df,
    class_col="class",
    drop_cols=("class", "dataset"),
    n_permutations=1000,
    seed=42,
):
    """
    Perform pairwise permutation tests between all classes in `df[class_col]`,
    comparing mean vectors via Euclidean distance.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the class column and feature columns.
    class_col : str
        Name of the column that identifies the class/group.
    drop_cols : tuple/list of str
        Columns to drop before computing feature means (typically the
        class column and any non-feature columns).
    n_permutations : int
        Number of permutations to run per pair.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    p_values : pd.DataFrame
        Symmetric matrix of p-values, indexed/columns by class name.
    stats : pd.DataFrame
        Symmetric matrix of standardized test statistics, indexed/columns by class name.
    """
    np.random.seed(seed)

    class_names = df[class_col].unique()
    n_classes = len(class_names)

    test_results = np.zeros((n_classes, n_classes))
    test_results_stats = np.zeros((n_classes, n_classes))

    # Precompute feature arrays per class to avoid repeated filtering
    class_data = {
        name: df[df[class_col] == name].drop(columns=list(drop_cols)).values
        for name in class_names
    }

    for i, name1 in enumerate(class_names):
        for j, name2 in enumerate(class_names):
            if j <= i:
                data1 = class_data[name1]
                data2 = class_data[name2]

                mean1 = data1.mean(axis=0)
                mean2 = data2.mean(axis=0)
                obs_diff = np.linalg.norm(mean1 - mean2)

                combined = np.concatenate([data1, data2], axis=0)
                n1 = data1.shape[0]

                perm_diffs = np.empty(n_permutations)
                for perm in range(n_permutations):
                    perm_data = combined[np.random.permutation(combined.shape[0])]
                    perm_mean1 = perm_data[:n1].mean(axis=0)
                    perm_mean2 = perm_data[n1:].mean(axis=0)
                    perm_diffs[perm] = np.linalg.norm(perm_mean1 - perm_mean2)

                stat = (obs_diff - perm_diffs.mean()) / perm_diffs.std()
                p_value = (np.sum(perm_diffs >= obs_diff) + 1) / (n_permutations + 1)

                test_results[i, j] = p_value
                test_results_stats[i, j] = stat

    # Mirror the lower triangle to the upper triangle
    test_results = test_results + test_results.T - np.diag(test_results.diagonal())
    test_results_stats = (
        test_results_stats
        + test_results_stats.T
        - np.diag(test_results_stats.diagonal())
    )

    p_values = pd.DataFrame(test_results, index=class_names, columns=class_names)
    stats = pd.DataFrame(test_results_stats, index=class_names, columns=class_names)

    return p_values, stats
