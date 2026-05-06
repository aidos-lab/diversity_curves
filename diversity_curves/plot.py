import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_heatmap(diff_matrix, ys, path=None):
    order = np.argsort(ys)
    #sorted_matrix = diff_feature[np.ix_(order, order)]
    #sorted_labels = [ys[i] for i in order]

    #diff_combined = diff_matrix/np.quantile(diff_matrix, 0.95) 
    sorted_matrix = diff_matrix[np.ix_(order, order)]
    sorted_labels = [ys[i] for i in order]
    plt.clf()
    sns.heatmap(sorted_matrix, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="viridis")
    plt.xlabel("y")
    plt.ylabel("y")
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    plt.close()

def plot_all_spread(all_spreads, path=None):
    sns.lineplot(data=all_spreads, x=all_spreads.index, y=all_spreads[all_spreads.columns[0]], hue="graph")
    if path is not None:
        plt.savefig(path)
    plt.show()
    # plt.close()

def plot_areas(df_areas, path=None):
    sns.violinplot(data=df_areas, x="area", hue="y")
    if path is not None:
        plt.savefig(path)
    #plt.show()
    plt.close()