from __future__ import print_function
import numpy as np
import random
from grakel import graph_from_networkx
import networkx as nx


def simulate_graphs_kwargs(nodes_range = range(5, 35), n_graphs = 1, model="ER", 
                           ER_probs=[0.25, 0.5, 0.75], ER_nps = None,
                           RG_probs=[0.25, 0.5, 0.75],
                           BA_ms=[3.0, 5.0, 8.0], BA_initial=None,
                           DBA_m=[2, 5], DBA_ps=[0.25, 0.5, 0.75], DBA_initial=None,
                           SBM_probs=[0.6, 0.01], SBM_blocks=[[1/2], [1/3, 1/3], [1/4, 1/4, 1/4]],
                           WS_probs=[0.01, 0.25, 0.5], WS_k=6,
                           NWS_probs=[0.01, 0.25, 0.5], NWS_k=6,
                           CWS_probs=[0.01, 0.25, 0.5], CWS_k=6,
                           RPT_gammas = [2.0, 3.0, 4.0], 
                           LFR_mu=0.1, LFR_avdeg_prop=0.02, LFR_mincom=20, LFR_taus=[[2, 2], [3,2], [2, 1.1], [3, 1.1]],
                           RL_probs= [[0.5, 0.5]], RL_node_var = 5, 
                           PLC_probs = [0.25, 0.5, 0.75], PLC_m=2,
                           RPG_probs =  [[0.9, 0.1], [0.7, 0.1], [0.5, 0.1]], RPG_props = [1/3, 1/3],
                           GRP_mean_probs = [[0.9, 0.1], [0.7, 0.1], [0.5, 0.1]], GRP_mean_prop = 1/3, GRP_shape=3, 
                           W_alphas=[0.1, 0.2, 0.3], W_beta=0.4,
                           RR_ds = [2, 4, 6],
                           ):

    np.random.seed(43)
    random.seed(314)
    if model == 'ER':
        if ER_nps is None:
            y_probs = ER_probs
        else:
            y_probs = ER_nps
    elif model == "BA":
        y_probs = BA_ms
    elif model == 'DBA':
        y_probs = DBA_ps
    elif model == "WS":
        y_probs = WS_probs
    elif model == "CWS":
        y_probs = CWS_probs
    elif model == "SBM":
        y_probs = SBM_blocks
    elif model == 'RPT':
        y_probs = RPT_gammas
    elif model == 'LFR':
        y_probs = LFR_taus
    elif model == 'RL':
        y_probs = RL_probs
    elif model == 'PLC':
        y_probs = PLC_probs
    elif model == 'NWS':
        y_probs = NWS_probs
    elif model == 'RPG':
        y_probs = RPG_probs
    elif model == 'W':
        y_probs = W_alphas
    elif model == 'RG':
        y_probs = RG_probs
    elif model == 'RR':
        y_probs = RR_ds
    else:
        raise ValueError("Model not recognized")
    Gs = []
    y = []
    G_nx = []
    n_nodes = []
    y_probs_multi = []
    
    graph_success = True

    for i, prob in enumerate(y_probs):
        for j in nodes_range:
            for k in range(n_graphs):
                if model == "ER":
                    if ER_nps is None:
                        graph = nx.erdos_renyi_graph(n=j, p=prob)
                    else:
                        graph = nx.erdos_renyi_graph(n=j, p=prob/j)
                        
                elif model == 'BA':
                    m=prob
                    if BA_initial =='2star':
                        star1 = nx.star_graph(int(np.ceil(m/2)))
                        initial_graph = nx.disjoint_union(star1, star1)
                        nx.contracted_nodes(initial_graph, 1, 2*star1.number_of_nodes()-1, copy=False)
                    elif BA_initial == '2star_edge':
                        star1 = nx.star_graph(int(np.ceil(m/2)))
                        initial_graph = nx.disjoint_union(star1, star1)
                        initial_graph.add_edge(1, int(np.ceil(m/2))+2)
                    elif not BA_initial:
                        initial_graph = None
                    else:
                        raise ValueError('BA_initial not recognized')
                    graph = nx.barabasi_albert_graph(n=j, m=m, initial_graph=initial_graph)
                elif model == 'DBA':
                    m = max(DBA_m)
                    if BA_initial =='2star':
                        star1 = nx.star_graph(int(np.ceil(m/2)))
                        initial_graph = nx.disjoint_union(star1, star1)
                        nx.contracted_nodes(initial_graph, 1, 2*star1.number_of_nodes()-1, copy=False)
                    elif BA_initial == '2star_edge':
                        star1 = nx.star_graph(int(np.ceil(m/2)))
                        initial_graph = nx.disjoint_union(star1, star1)
                        initial_graph.add_edge(1, int(np.ceil(m/2))+2)
                    elif not BA_initial:
                        initial_graph = None
                    else:
                        raise ValueError('BA_initial not recognized')
                    graph = nx.dual_barabasi_albert_graph(n=j, m1=DBA_m[0], m2=DBA_m[1], p=prob, initial_graph=initial_graph)
                elif model == "WS":
                    graph = nx.watts_strogatz_graph(n=j, k=WS_k, p=prob)
                elif model == "CWS":
                    graph = nx.connected_watts_strogatz_graph(n=j, k=CWS_k, p=prob)
                elif model == 'NWS':
                    graph = nx.newman_watts_strogatz_graph(n=j, k=NWS_k, p=prob)
                elif model == "RG":
                    graph = nx.random_geometric_graph(n=j, radius=prob)
                elif model == "SBM":
                    blocklist = prob
                    sizes = [int(np.floor(j*size)) for size in blocklist]
                    remainder = j - np.sum(sizes)
                    sizes.append(remainder)
                    p_in = SBM_probs[0]
                    p_out = SBM_probs[1]
                    probs_matrix = np.diag([p_in]*(len(blocklist)+1))
                    probs_matrix[probs_matrix==0] = p_out
                    graph = nx.stochastic_block_model(sizes, probs_matrix.tolist())
                elif model == 'RPG':
                    prop_sizes = RPG_props
                    sizes = [int(np.floor(j*prop)) for prop in prop_sizes]
                    remainder = j - np.sum(sizes)
                    sizes.append(remainder)
                    graph = nx.random_partition_graph(sizes=sizes, p_in=prob[0], p_out=prob[1])
                elif model == 'RPT':
                    try:
                        graph = nx.random_powerlaw_tree(n=j, gamma=prob, tries=100000)
                        graph_success = True
                    except Exception as e:
                        print(f'graph could not be constructed within max number tries')
                        graph_success = False
                elif model == 'LFR':
                    print(f'trying to generate graph with {j} nodes and combination {prob}')
                    try:
                        graph = nx.LFR_benchmark_graph(n=j, tau1=prob[0], tau2=prob[1], mu=LFR_mu, average_degree=LFR_avdeg_prop*j, min_community=LFR_mincom, seed=314)
                        print(f'success.')
                        graph_success = True
                    except Exception as e:
                        print(f'graph could not be constructed with error {e}') 
                        graph_success = False
                elif model == 'RL':
                    graph_success = False
                    while not graph_success:
                        graph = nx.random_lobster_graph(n=j, p1=prob[0], p2=prob[1])
                        if len(graph.nodes()) >= max(j-RL_node_var, 1) and len(graph.nodes()) <= j+RL_node_var and graph.number_of_edges() > 1:            
                            graph_success = True
                elif model == 'PLC':
                    graph = nx.powerlaw_cluster_graph(n=j, m=PLC_m, p=prob)
                elif model == 'W':
                    graph = nx.waxman_graph(n=j, beta=W_beta, alpha=prob)
                elif model == 'RR':
                    graph_success=False
                    try:
                        graph = nx.random_regular_graph(d=prob, n=j)
                        graph_success=True
                    except Exception as e:
                        continue
                else:
                    raise ValueError("Model not recognized")

                if graph_success:
                    edges = list(graph.edges())
                    adj = nx.adjacency_matrix(graph)
                    G_nx.append(graph)
                    nx.set_node_attributes(G_nx[-1], 1, name='feature') #dummy label for nx
                    node_labels = {u: 1 for u in G_nx[-1].nodes()}  # dummy label
                    grakel_gen = graph_from_networkx([graph], as_Graph=True, node_labels_tag='feature')
                    Gs.append(next(grakel_gen))
                    y.append(i)
                    n_nodes.append(graph.number_of_nodes())
                    y_probs_multi.append(prob)

    return  G_nx, Gs, y, y_probs_multi, n_nodes

