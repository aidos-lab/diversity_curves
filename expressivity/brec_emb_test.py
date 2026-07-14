import networkx as nx
from brec.dataset import BRECDataset
import numpy as np
import pandas as pd
import numpy as np
from karateclub import NetLSD, Graph2Vec, FeatherGraph


gm_train = BRECDataset("../../notebooks/Data")
extracted_pairs = []

for i in range(400):
    # Each pair i starts at index (i * 64)
    # The first version of Graph A is at that start index
    # The first version of Graph B is at the very next index due to interlacing
    
    idx_a = i * 64
    idx_b = i * 64 + 1
    
    graph_a = gm_train[idx_a]
    graph_b = gm_train[idx_b]
    extracted_pairs.append(graph_a)
    extracted_pairs.append(graph_b)

extracted_pairs_test = []

for i in range(400, 800):
    idx_a = i * 64
    idx_b = i * 64 + 1
    
    graph_a = gm_train[idx_a]
    graph_b = gm_train[idx_b]
    extracted_pairs_test.append(graph_a)
    extracted_pairs_test.append(graph_b)

def pyg_dataset_to_networkx(gm_train):
    labels = []
    nx_graphs = []
    for data in gm_train:
        edges_list = data.edge_index.numpy().T.tolist()
        G = nx.Graph(edges_list)
        nx_graphs.append(G)
    return labels, nx_graphs

labels_train, nx_graphs_train = pyg_dataset_to_networkx(extracted_pairs)
labels_test, nx_graphs_test = pyg_dataset_to_networkx(extracted_pairs_test)
y_train = labels_train #+ y_train
data_train = nx_graphs_train 
G_nx = nx_graphs_train

n_nodes = [g.number_of_nodes() for g in G_nx]


for metric in ["netlsd", "feather"]:
    G_nx = nx_graphs_train
    method=metric
    test_results=[]
    for i in range(0,800,2):
        #print(i)
        j=i+1
        
        Gs=[G_nx[i], G_nx[j]]
        G_nx_fixed = [
            nx.convert_node_labels_to_integers(g, ordering="sorted")
            for g in Gs
        ]
        
        if method == "netlsd":
            embedding = NetLSD()
        elif method == "graph2":
            embedding = Graph2Vec()
        elif method == "feather":
            embedding = FeatherGraph()
        embedding.fit([g.copy() for g in G_nx_fixed])
        array = embedding.get_embedding()

        spread_i = array[0,:]
        spread_j = array[1,:]

        stat = np.sum(np.abs(spread_i-spread_j))
        p_value = stat >10**(-5)
        test_results.append(p_value)
    test_results_df = pd.DataFrame({"pval": pd.Series(test_results)})
    lables = ["Basic" for r in range(60)]+["Regular" for r in range(100)]+["Extension" for r in range(100)]+["CFI" for r in range(100)]+["4V" for r in range(20)]+["DR" for r in range(20)]
    test_results_df.columns=["pval"]
    test_results_df["lables"]=lables
    test_results_df["sig"] = test_results_df["pval"]
    results = test_results_df[["lables", "sig"]].groupby(["lables"]).sum().T
    ordered = results[["Basic", "Regular", "Extension", "CFI", "4V", "DR"]]
    ordered.loc["all",:] = [60, 100, 100, 100, 20, 20]
    ordered.loc["prop",:] = (ordered.loc["sig",:] / ordered.loc["all",:]).values
    ordered["sum"] = ordered.sum(axis=1)
    ordered["sum"]["prop"] = ordered["sum"]["sig"] / ordered["sum"][ "all"]
    ordered_v2 = ordered.copy()
    ordered_v2["Regular"] = ordered_v2["Regular"]+ordered_v2["4V"]+ordered_v2["DR"]
    ordered_v2=ordered_v2.drop(columns=["4V", "DR"])
    ordered_v2.loc["prop",:] = (ordered_v2.loc["sig",:] / ordered_v2.loc["all",:]).values
    print(metric)
    print(ordered_v2)
    ordered_v2.to_csv(f"./brec_{metric}.csv")



for metric in ["netlsd", "feather"]:
    method=metric
    G_nx = nx_graphs_test
    test_results=[]
    for i in range(0,800,2):
        #print(i)
        j=i+1
        
        Gs=[G_nx[i], G_nx[j]]
        G_nx_fixed = [
            nx.convert_node_labels_to_integers(g, ordering="sorted")
            for g in Gs
        ]
        
        if method == "netlsd":
            embedding = NetLSD()
        elif method == "graph2":
            embedding = Graph2Vec()
        elif method == "feather":
            embedding = FeatherGraph()
        embedding.fit([g.copy() for g in G_nx_fixed])
        array = embedding.get_embedding()

        spread_i = array[0,:]
        spread_j = array[1,:]
        
        
        stat = np.sum(np.abs(spread_i-spread_j))
        p_value = stat >10**(-5)  
        test_results.append(p_value)
    test_results_df = pd.DataFrame({"pval": pd.Series(test_results)})
    lables = ["Basic" for r in range(60)]+["Regular" for r in range(100)]+["Extension" for r in range(100)]+["CFI" for r in range(100)]+["4V" for r in range(20)]+["DR" for r in range(20)]
    test_results_df.columns=["pval"]
    test_results_df["lables"]=lables
    test_results_df["sig"] = test_results_df["pval"]
    results = test_results_df[["lables", "sig"]].groupby(["lables"]).sum().T
    ordered = results[["Basic", "Regular", "Extension", "CFI", "4V", "DR"]]
    ordered.loc["all",:] = [60, 100, 100, 100, 20, 20]
    ordered.loc["prop",:] = (ordered.loc["sig",:] / ordered.loc["all",:]).values
    ordered["sum"] = ordered.sum(axis=1)
    ordered["sum"]["prop"] = ordered["sum"]["sig"] / ordered["sum"][ "all"]
    ordered_v2 = ordered.copy()
    ordered_v2["Regular"] = ordered_v2["Regular"]+ordered_v2["4V"]+ordered_v2["DR"]
    ordered_v2=ordered_v2.drop(columns=["4V", "DR"])
    ordered_v2.loc["prop",:] = (ordered_v2.loc["sig",:] / ordered_v2.loc["all",:]).values
    print(metric)
    print(ordered_v2)
    ordered_v2.to_csv(f"./brec_{metric}_reliability.csv")

