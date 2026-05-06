from __future__ import print_function
import numpy as np
import scipy

import random
# from random import random

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from grakel import Graph, graph_from_networkx
import networkx as nx

# Generates 3 sets of Erdos-Renyi graphs. Each edge is included in the graph with probability p
# independent from every other edge. The probability p is set equal to 0.25, 0.5 and 0.75 for
# the graphs of the 1st, 2nd and 3rd set, respectively

def simulate_graphs(probs = [0.25, 0.5, 0.75], nodes_range = range(5, 35), model="ER"):

    np.random.seed(43)
    
    Gs = list()
    y = list()
    G_nx = []
    n_nodes = []
    y_probs = []
    #[0.3, 0.5, 0.7, 0.9]
    #[0.1, 0.25,0.33, 0.5,0.66, 0.75, 0.9]
    for i in range(len(probs)):
        for j in nodes_range:
                if model == "ER":
                    graph = nx.erdos_renyi_graph(n=j, p=probs[i])
                elif model == "BA":
                    graph = nx.barabasi_albert_graph(n=j, m=max(1, int(probs[i]*j/2)))
                elif model == "WS":
                    graph = nx.watts_strogatz_graph(n=j, k=5, p=probs[i])
                elif model == "CWS":
                    graph = nx.connected_watts_strogatz_graph(n=j, k=5, p=probs[i])
                elif model == "RG":
                    graph = nx.random_geometric_graph(n=j, radius=probs[i])
                elif model == "GNM":
                    m = int(probs[i] * j * (j - 1) / 2)
                    graph = nx.gnm_random_graph(n=j, m=m)
                elif model == "SBM":
                    sizes = [j // 2, j - j // 2]
                    p_in = probs[i]
                    p_out = probs[i] / 10
                    probs_matrix = [[p_in, p_out], [p_out, p_in]]
                    graph = nx.stochastic_block_model(sizes, probs_matrix)
                else:
                    raise ValueError("Model not recognized")

                edges = list(graph.edges())
                G_nx.append(nx.from_edgelist(edges))
                nx.set_node_attributes(G_nx[-1], 1, name='feature') #dummy label for nx
                node_labels = {u: 1 for u in G_nx[-1].nodes()}  # dummy label
                Gs.append(Graph(edges, node_labels=node_labels))
                y.append(i)
                n_nodes.append(j)
                y_probs.append(probs[i])

    return  G_nx, Gs, y, y_probs, n_nodes

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
        # print(str(y_probs))
    elif model == "BA":
        y_probs = BA_ms
        # print(y_probs).
    elif model == 'DBA':
        y_probs = DBA_ps
    elif model == "WS":
        y_probs = WS_probs
        # print(y_probs)
    elif model == "CWS":
        y_probs = CWS_probs
        # print(y_probs)
    # elif model == "RG":
    #
    # elif model == "GNM":
    #
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
            # print(f'nodes {j} and prob {prob}')
            for k in range(n_graphs):
                if model == "ER":
                    # print(prob)
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
                    # print(prob, WS_k)
                    graph = nx.watts_strogatz_graph(n=j, k=WS_k, p=prob)
                elif model == "CWS":
                    graph = nx.connected_watts_strogatz_graph(n=j, k=CWS_k, p=prob)
                elif model == 'NWS':
                    graph = nx.newman_watts_strogatz_graph(n=j, k=NWS_k, p=prob)
                elif model == "RG":
                    graph = nx.random_geometric_graph(n=j, radius=prob)
                # elif model == "GNM":
                #     m = int(probs[i] * j * (j - 1) / 2)
                #     graph = nx.gnm_random_graph(n=j, m=m)
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
                    # edges = list(graph.edges())
                    # G_nx.append(nx.from_edgelist(edges))
                    # nx.set_node_attributes(G_nx[-1], 1, name='feature') #dummy label for nx
                    # node_labels = {u: i for i, u in enumerate(G_nx[-1].nodes())}  # dummy label
                    # Gs.append(Graph(edges, node_labels=node_labels))
                    # y.append(i)
                    # n_nodes.append(graph.number_of_nodes())


                    # alternative keeping the isolated vertices
                    # print(f'print generating graph with {j} nodes and prob {prob}')
                    edges = list(graph.edges())
                    adj = nx.adjacency_matrix(graph)
                    # G_nx.append(nx.from_edgelist(edges))
                    G_nx.append(graph)
                    nx.set_node_attributes(G_nx[-1], 1, name='feature') #dummy label for nx
                    node_labels = {u: 1 for u in G_nx[-1].nodes()}  # dummy label
                    # Gs.append(Graph(edges, node_labels=node_labels))
                    # Gs.append(Graph(scipy.sparse.csr_matrix(adj), node_labels=node_labels))
                    grakel_gen = graph_from_networkx([graph], as_Graph=True, node_labels_tag='feature')
                    Gs.append(next(grakel_gen))
                    y.append(i)
                    n_nodes.append(graph.number_of_nodes())
                    y_probs_multi.append(prob)

    return  G_nx, Gs, y, y_probs_multi, n_nodes

# adapted https://github.com/lrjconan/GRAN Lobster dataset
def simulate_random_lobster(nodes_range = range(5, 35), probs = [[0.5, 0.5]], node_var = 5, seed=1234):
    Gs = []
    y = []
    G_nx = []
    n_nodes = []


    for i, prob_pair in enumerate(probs):
        for j in nodes_range:
            # seed_tmp = seed
            valid = False
            while not valid:
                graph = nx.random_lobster_graph(j, prob_pair[0], prob_pair[1])#, seed=seed_tmp)
                if len(graph.nodes()) >= max(j-node_var, 1) and len(graph.nodes()) <= j+node_var:
                    # G_nx.append(nx.from_edgelist(G))
                    edges = list(graph.edges())
                    if len(edges) > 1:
                        G_nx.append(nx.from_edgelist(edges))
                        nx.set_node_attributes(G_nx[-1], 1, name='feature') #dummy label for nx
                        node_labels = {u: 1 for u in G_nx[-1].nodes()}  # dummy label
                        Gs.append(Graph(edges, node_labels=node_labels))
                        y.append(i)
                        n_nodes.append(graph.number_of_nodes())
                    
                    valid = True

                # seed_tmp += 1

    print('max # nodes = {} || mean # nodes = {} || min # nodes = {}'.format(max(n_nodes), np.mean(n_nodes), min(n_nodes)))
    # print('max # edges = {} || mean # edges = {}'.format(max(num_edges), np.mean(num_edges)))

    return  G_nx, Gs, y, probs, n_nodes

def simulate_graphs_SBM(probin = 0.75, probout= 0.1, blocklists=[[1/2], [1/3, 1/3], [1/4, 1/4, 1/4]], nodes_range = range(15, 50), model="SBM"):
    if model != 'SBM':
         raise ValueError('Model not recognized')
    np.random.seed(43)
    
    Gs = list()
    y = list()
    G_nx = []
    n_nodes = []
    y_probs = []

    for i, blocklist in enumerate(blocklists):
        for j in nodes_range:

                sizes = [int(np.floor(j*size)) for size in blocklist]
                remainder = j - np.sum(sizes)
                sizes.append(remainder)
                p_in = probin
                p_out = probout
                probs_matrix = np.diag([p_in]*(len(blocklist)+1))
                probs_matrix[probs_matrix==0] = p_out
                # probs_matrix = [[p_in, p_out], [p_out, p_in]]
                graph = nx.stochastic_block_model(sizes, probs_matrix.tolist())

                edges = list(graph.edges())
                G_nx.append(nx.from_edgelist(edges))
                nx.set_node_attributes(G_nx[-1], 1, name='feature') #dummy label for nx
                node_labels = {u: 1 for u in G_nx[-1].nodes()}  # dummy label
                Gs.append(Graph(edges, node_labels=node_labels))
                y.append(i)
                n_nodes.append(j)
                y_probs.append(blocklist)

    return  G_nx, Gs, y, y_probs, n_nodes

def simulate_graphs_BA(ms, initial = None, nodes_range = range(15,50), model='BA'):
    np.random.seed(43)
    if model != 'BA':
         raise ValueError('Model not recognized')
        
    Gs = list()
    y = list()
    G_nx = []
    G_nx_alt = []
    n_nodes = []
    y_probs = []

    for i in range(len(ms)):
        for j in nodes_range:

            m = ms[i]
            if initial =='2star':
                star1 = nx.star_graph(int(np.ceil(m/2)))
                initial_graph = nx.disjoint_union(star1, star1)
                nx.contracted_nodes(initial_graph, 1, int(np.ceil(m/2))+2, copy=False)
            else:
                initial_graph = None
            graph = nx.barabasi_albert_graph(n=j, m=m, initial_graph=initial_graph)
    

            edges = list(graph.edges())
            G_nx.append(nx.from_edgelist(edges))
            G_nx_alt.append(graph)
            nx.set_node_attributes(G_nx[-1], 1, name='feature') #dummy label for nx
            node_labels = {u: 1 for u in G_nx[-1].nodes()}  # dummy label
            Gs.append(Graph(edges, node_labels=node_labels))
            y.append(i)
            n_nodes.append(j)
            y_probs.append(ms[i])
    return  G_nx, Gs, y, y_probs, n_nodes

def simulate_graphs_WS(probs, k=5, nodes_range = range(15,50), model='WS'):
    np.random.seed(43)
    if model != 'WS':
         raise ValueError('Model not recognized')
        
    Gs = list()
    y = list()
    G_nx = []
    G_nx_alt = []
    n_nodes = []
    y_probs = []

    for i in range(len(probs)):
        for j in nodes_range:

            graph = nx.watts_strogatz_graph(n=j, k=k, p=probs[i])

            edges = list(graph.edges())
            G_nx.append(nx.from_edgelist(edges))
            G_nx_alt.append(graph)
            nx.set_node_attributes(G_nx[-1], 1, name='feature') #dummy label for nx
            node_labels = {u: 1 for u in G_nx[-1].nodes()}  # dummy label
            Gs.append(Graph(edges, node_labels=node_labels))
            y.append(i)
            n_nodes.append(j)
            y_probs.append(probs[i])
    return  G_nx, Gs, y, y_probs, n_nodes

def simulate_graphs_param_sweep(param_values, nodes_range = range(5, 35), model="ER", param_name="probs", **fixed_params):
    """
    Flexible function to vary a different parameter of random graph models.
    
    Parameters:
    -----------
    param_values : list
        Values to iterate through for the main parameter to vary
    nodes_range : range, default=range(5, 35)
        Range of node counts for graph generation
    model : str, default="ER"
        Graph model to use: "ER", "BA", "WS", "CWS", "SBM", "RG", "GNM"
    param_name : str, default="probs"
        Name of the parameter to vary (e.g., "WS_k", "WS_probs", "BA_m", etc.)
    **fixed_params : dict
        Fixed parameters for the model (e.g., WS_p=0.5, BA_initial=None)
        
    Returns:
    --------
    G_nx : list
        List of NetworkX graphs
    Gs : list
        List of GraKeL graphs
    y : list
        Class labels (index of param_values)
    param_values_used : list
        The actual parameter values used
    n_nodes : list
        Number of nodes for each graph
    """
    np.random.seed(43)
    
    Gs = []
    y = []
    G_nx = []
    n_nodes = []
    
    for i, param_val in enumerate(param_values):
        for j in nodes_range:
            # Prepare parameters based on model type
            params = {'n': j}
            
            if model == "ER":
                params['p'] = param_val if param_name == "ER_p" else fixed_params.get('ER_p', 0.5)
                graph = nx.erdos_renyi_graph(**params)
                
            elif model == "BA":
                params['m'] = param_val if param_name == "BA_m" else fixed_params.get('BA_m', 3)
                initial_graph = fixed_params.get('BA_initial_graph', None)
                if initial_graph is not None:
                    params['initial_graph'] = initial_graph
                graph = nx.barabasi_albert_graph(**params)
                
            elif model == "WS":
                params['k'] = param_val if param_name == "WS_k" else fixed_params.get('WS_k', 5)
                params['p'] = param_val if param_name == "WS_p" else fixed_params.get('WS_p', 0.5)
                graph = nx.watts_strogatz_graph(**params)
                
            elif model == "CWS":
                params['k'] = param_val if param_name == "WS_k" else fixed_params.get('WS_k', 5)
                params['p'] = param_val if param_name == "WS_p" else fixed_params.get('WS_p', 0.5)
                graph = nx.connected_watts_strogatz_graph(**params)
                
            elif model == "RG":
                params['radius'] = param_val if param_name == "RG_radius" else fixed_params.get('RG_radius', 0.3)
                graph = nx.random_geometric_graph(**params)
                
            elif model == "GNM":
                m = param_val if param_name == "GNM_m" else fixed_params.get('GNM_m', int(0.3 * j * (j - 1) / 2))
                graph = nx.gnm_random_graph(n=j, m=m)
                
            elif model == "SBM":
                blocklist = param_val
                sizes = [int(np.floor(j*size)) for size in blocklist]
                remainder = j - np.sum(sizes)
                sizes.append(remainder)
                p_in = fixed_params.get('SBM_p_in', 0.75)
                p_out = fixed_params.get('SBM_p_out', 0.1)
                probs_matrix = np.diag([p_in]*(len(blocklist)+1))
                probs_matrix[probs_matrix==0] = p_out
                graph = nx.stochastic_block_model(sizes, probs_matrix.tolist())
                
            else:
                raise ValueError(f"Model '{model}' not recognized")

            edges = list(graph.edges())
            G_nx.append(nx.from_edgelist(edges))
            nx.set_node_attributes(G_nx[-1], 1, name='feature')
            node_labels = {u: 1 for u in G_nx[-1].nodes()}
            Gs.append(Graph(edges, node_labels=node_labels))
            y.append(i)
            n_nodes.append(j)

    return G_nx, Gs, y, param_values, n_nodes