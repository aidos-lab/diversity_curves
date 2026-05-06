from diversity_curves.coarsen import repeat_edge_dropping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from diversity_curves.weight_diff import compute_magnitude_graph

def get_mag_graphs(graphs, dist_fn, ts, upsample_to=None, n_repeats=5, compute_at_cardinality=None, mode="full", scores_method="random", upsample_method="nodes", downsample_to=None, upsample_mode = 'node_duplicating'):
    spread_dfs = []
    weights_lists = []

    for i, g in enumerate(graphs):
        
        spreads_1 = []
        weight_dict = {}
        if g.number_of_nodes() > 0:
            first_node = list(g.nodes())[0]
            if g.nodes[first_node].get("feature") is None:
                for n in g.nodes():
                    g.nodes[n]["feature"] = np.array([1.0])
     
        for seed in range(n_repeats):
            np.random.seed(seed)
            if downsample_to:
                n_steps = g.number_of_nodes() - downsample_to
            else:
                n_steps = g.number_of_nodes()
            this_graph, S, spread, n_nodes_spread, weight_list = repeat_edge_dropping(g, ts, dist_fn, original_magni=None, n_steps=n_steps, method="spread", scores_method = scores_method, get_weights=True, 
                                                                                      upsample_to=upsample_to, upsample_mode = upsample_mode, compute_at_cardinality=compute_at_cardinality, upsample_method=upsample_method)
            this_col_2 = pd.Series(spread)
            this_col_2.index = pd.Series(n_nodes_spread)
            this_col_2 = pd.DataFrame(this_col_2)
            this_col_2["repeat"] = seed 
            spreads_1.append(this_col_2)
            for nodes, weights in zip(n_nodes_spread, weight_list):
                if nodes not in weight_dict:
                    weight_dict[nodes] = []
                weight_dict[nodes].append(weights)
        for key in weight_dict:
            weight_dict[key] = np.mean(weight_dict[key], axis=0)
        weights_lists.append(weight_dict)
        spread_df = pd.DataFrame()
        for df in spreads_1:
            spread_df = pd.concat([spread_df, df], axis=0)
        
        spread_df.columns = range(spread_df.shape[1])
        spread_df["graph"] = i
        spread_dfs.append(spread_df)

    all_spreads = pd.concat(spread_dfs, axis=0)
    return all_spreads, weights_lists

def get_mag_funs_graphs(graphs, dist_fn, ts, method="spread", get_weights=False):
    magnitudes = []
    weight_list = []
    for i, g in enumerate(graphs):
        mag, ts = compute_magnitude_graph(g, dist_fn, ts=ts,method=method, get_weights=get_weights)
        if get_weights:
            this_magnitude = np.sum(mag, axis=0)
            magnitudes.append(this_magnitude)
            weight_list.append(mag)
        else:
            this_magnitude = mag
            magnitudes.append(this_magnitude)
            weight_list.append(None)
    return magnitudes, weight_list

def get_weight_diffs(weight_lists):
    diff_matrix = np.zeros((len(weight_lists), len(weight_lists)))
    for i in range(len(weight_lists)):
        for j in range(i+1, len(weight_lists)):
            g1_weights = weight_lists[i]
            g2_weights = weight_lists[j]
            common_keys = set(g1_weights.keys()).intersection(set(g2_weights.keys()))
            diffs = [(1, 0.0)]  # Dummy entry to ensure non-empty
            for key in common_keys:
                w1 = np.array(g1_weights[key])
                w2 = np.array(g2_weights[key])
                diff = np.sum(np.abs(w1 - w2))
                diffs.append((key, diff))
            diffs.sort(key=lambda x: x[0])
            diff_matrix[i, j] = np.trapz([d[1] for d in diffs], x=[d[0] for d in diffs])
    return diff_matrix + diff_matrix.T

def get_funs_diff(all_spreads):
    sp_df = all_spreads.reset_index()
    n_graphs  = sp_df.graph.nunique()
    sp_df = sp_df.groupby(["graph", "index"]).mean()
    sp_df = sp_df.pivot_table(index="index", columns="graph", values=sp_df.columns[0])
    mag_funs = sp_df.T
    mag_funs.loc[:,1] = 1
    mag_funs.sort_index(axis=1, inplace=True)

    diff_matrix_mag = np.zeros((n_graphs, n_graphs))
    for i in range(n_graphs):
        for j in range(i+1, n_graphs):
            diff = np.abs(mag_funs.loc[i,:] - mag_funs.loc[j,:])
            diff.dropna(inplace=True)
            diff_matrix_mag[i, j] = np.trapz(y=diff.values, x=diff.index)

    diff_matrix_mag = diff_matrix_mag + diff_matrix_mag.T
    return diff_matrix_mag

def plot_ordered_heatmap(diff_matrix, ys):
    order = np.argsort(ys)
    sorted_matrix = diff_matrix[np.ix_(order, order)]
    sorted_labels = [ys[i] for i in order]
    plt.clf()
    sns.heatmap(sorted_matrix, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="viridis")
    plt.xlabel("y")
    plt.ylabel("y")
    plt.tight_layout()

def get_mag_areas(all_spreads, y=None):
    areas = []
    n_graphs = all_spreads.graph.nunique()
    for i in range(n_graphs):
        sub = all_spreads[all_spreads["graph"]==i]
        #sub = sub[sub.index<10]
        sub.reset_index(inplace=True)
        sub = sub.groupby("index")[0].mean()
        sub[1] = 1
        #if 
        sub = sub.sort_index()
        area = np.trapz(sub, x=sub.index)
        areas.append(area)
    
    if y is None:
        df_areas = pd.DataFrame({"area": areas, "graph": [i for i in range(n_graphs)]})
    else:
        df_areas = pd.DataFrame({"area": areas, "graph": [i for i in range(n_graphs)], "y": y})
    return df_areas

def get_funs_df(all_spreads):
    sp_df = all_spreads.reset_index()
    #n_graphs  = sp_df.graph.nunique()
    sp_df = sp_df.groupby(["graph", "index"]).mean()
    sp_df = sp_df.pivot_table(index="index", columns="graph", values=sp_df.columns[0])
    mag_funs = sp_df.T
    mag_funs.loc[:,1] = 1
    mag_funs.sort_index(axis=1, inplace=True)

    return mag_funs