"Coarsen the graphs in a hierarchical manner to get a filtration of graphs."
import random
import numpy as np
from magnipy.magnitude.compute import compute_magnitude_subgraphs_with_dist, compute_magnitude_from_distances #, compute_magnitude_graph
from diversity_curves.weight_diff import compute_magnitude_subgraphs, compute_magnitude_graph
#from diversity_curves.magnitude.scales import guess_convergence_scale, median_heuristic, get_scales
import networkx as nx


#  ╭──────────────────────────────────────────────────────────╮
#  │ Edge Scores used for Edge Pooling                        │
#  ╰──────────────────────────────────────────────────────────╯


def get_scores_edge(this_graph, edges, dist_fn, ts, original_magni, subgraphs, mags, method="cholesky"):
    scores = []

    for edge in edges:
        node_a, node_b = edge
        mags_steps = []
        for i, subgraph in enumerate(subgraphs):
            if edge[0] not in subgraph.nodes() or edge[1] not in subgraph.nodes():
                #continue
                mags_steps.append(mags[i])
            else:
                step_graph = subgraph.copy()
                nodes = [m for m in step_graph.nodes()]
                node_a, node_b = edge

                # Merge node_a and node_b into node_a
                neighbors_a = set(step_graph.neighbors(node_a)) - {node_b}
                neighbors_b = set(step_graph.neighbors(node_b)) - {node_a}
                merged_neighbors = neighbors_b.difference(neighbors_a)

                # Connect node_a to the neighbors of both nodes
                for neighbor in merged_neighbors:
                    step_graph.add_edge(node_a, neighbor)

                # Remove node_b and its edges
                step_graph.remove_node(node_b)

                # Compute the magnitude difference
                step_magni, _ = compute_magnitude_subgraphs(step_graph, subgraphs=[step_graph], dist_fn=dist_fn, ts=ts, get_weights=False, method=method)
                mags_steps.append(step_magni[0])
        
        mag_diff_this = abs(original_magni[0] - sum(mags_steps))
        scores.append(mag_diff_this)
    return scores


def get_scores_edge_approx_dist(this_graph, edges, dist_fn, ts, original_magni, subgraphs, Ds, mags, method="cholesky"):
    scores = []

    for edge in edges:
        node_a, node_b = edge
        mags_steps = []
        for i, subgraph in enumerate(subgraphs):
            if edge[0] not in subgraph.nodes() or edge[1] not in subgraph.nodes():
                #continue
                mags_steps.append(mags[i])
            else:
                step_graph = subgraph.copy()
                nodes = [m for m in step_graph.nodes()]
                
                node_a, node_b = edge

                # Merge node_a and node_b into node_a
                neighbors_a = set(step_graph.neighbors(node_a)) - {node_b}
                neighbors_b = set(step_graph.neighbors(node_b)) - {node_a}
                merged_neighbors = neighbors_b.difference(neighbors_a)

                D =  Ds[i].copy()
                D_sub = Ds[i].copy()

                for neighbor in merged_neighbors:
                    #print(nodes.index(node_a), nodes.index(neighbor))
                    weight = min(D[nodes.index(node_a),:][nodes.index(neighbor)],
                                D[nodes.index(node_b),:][nodes.index(neighbor)])
                    D_sub[nodes.index(node_a), nodes.index(neighbor)] = weight
                    #step_graph.add_edge(node_a, neighbor, weight=weight)

                index_b = nodes.index(node_b)

                #D_sub = D[-index_b, :]
                D_sub = np.delete(D_sub, index_b, axis=0)
                D_sub = np.delete(D_sub, index_b, axis=1)

                mag_step = compute_magnitude_from_distances(
                    D_sub,
                    ts=ts,
                    method=method,
                    get_weights=False,
                    one_point_property=True,
                    perturb_singularities=True,
                    positive_magnitude=False,
                    input_distances=True
                )
                mags_steps.append(mag_step[0])
        step_magni = sum(mags_steps)
        mag_diff_this = abs(original_magni[0] - step_magni)
        scores.append(mag_diff_this)
    return scores


#  ╭──────────────────────────────────────────────────────────╮
#  │ Upsampling Graphs or Distances                           │
#  ╰──────────────────────────────────────────────────────────╯


def upsample_distances(D, n_samples):
    """Upsample a distance matrix D to a larger size n_samples using nearest neighbor interpolation."""
    from scipy.spatial import cKDTree

    n_original = D.shape[0]
    if n_samples <= n_original:
        raise ValueError("n_samples must be greater than the size of D.")

    # Create original points
    original_points = np.arange(n_original).reshape(-1, 1)

    # Create new points
    new_points = np.linspace(0, n_original - 1, n_samples).reshape(-1, 1)

    # Build a KDTree for fast nearest neighbor search
    tree = cKDTree(original_points)

    # Find nearest neighbors in the original points for each new point
    _, indices = tree.query(new_points, k=1)

    # Create the upsampled distance matrix
    D_upsampled = D[indices.flatten()][:, indices.flatten()]

    return D_upsampled

def upsample_graph_random(G, n_samples):
    "Upsample a graph G to n_samples nodes by randomly resampling nodes and "
    "adding edges to all the neighbours of the original node as well as the chosen node."
    n_original = G.number_of_nodes()
    if n_samples == n_original:
        return G.copy()
    elif n_samples < n_original:
        raise ValueError("n_samples must be greater than the number of nodes in G.")
    n_samples = n_samples - n_original
    # Randomly sample nodes with replacement
    sampled_nodes = np.random.choice(list(G.nodes()), size=n_samples, replace=True)
    G_upsampled = G.copy()
    for j, node in enumerate(sampled_nodes):
        G_upsampled.add_node(str(j)+"_new", feature=G.nodes[node]['feature'])
        for neighbor in G.neighbors(node):
            G_upsampled.add_edge(str(j)+"_new", neighbor)
        G_upsampled.add_edge(str(j)+"_new", node)
    return G_upsampled

def upsample_graph(G, sampled_nodes,lables=None):
    "Upsample a graph G to n_samples nodes by randomly resampling nodes and "
    "adding edges to all the neighbours of the original node as well as the chosen node."
    G_upsampled = G.copy()
    for j, node in enumerate(sampled_nodes):
        new_name = str(node)+"_new" if lables is None else str(lables[j])+"_new"
        G_upsampled.add_node(new_name, feature=G.nodes[node]['feature'])
        for neighbor in G.neighbors(node):
            G_upsampled.add_edge(new_name, neighbor)
        G_upsampled.add_edge(new_name, node)
    return G_upsampled

def upsample_graph_subdiv(G, upsample_to):
    """Upsample a graph G to n_samples nodes by subdividing all edges in sampled_edges
    """
    n_original = G.number_of_nodes()
    m_original = G.number_of_edges()
    n_samples = upsample_to - n_original
    if n_samples < 0:
        raise ValueError('upsample to must be larger thatn number of nodes')
    if n_samples == 0:
        return None
    G_upsampled = G.copy()
    while G_upsampled.number_of_edges() <= upsample_to-G_upsampled.number_of_nodes():
        sampled_edges_i = list(np.random.permutation(range(G_upsampled.number_of_edges()))) 
        for j, edge_i in enumerate(sampled_edges_i):
            edge = list(G_upsampled.edges())[j]
            u = edge[0]
            v = edge[1]
            G_upsampled.remove_edge(u, v)
            new_node = 'div_'+str(u)+'_'+str(v)
            G_upsampled.add_node(new_node, feature=np.mean([G_upsampled.nodes[u]['feature'], G_upsampled.nodes[v]['feature']]) )

            G_upsampled.add_edges_from([(u, new_node), (new_node, v)])
            yield G_upsampled
    sampled_edges_i = list(np.random.choice(range(G_upsampled.number_of_edges()), size=upsample_to-G_upsampled.number_of_nodes(), replace=False))
    for j, edge in enumerate(sampled_edges_i):
        edge = list(G_upsampled.edges())[j]
        u = edge[0]
        v = edge[1]
        G_upsampled.remove_edge(u, v)
        new_node = 'div_'+str(u)+'_'+str(v)
        G_upsampled.add_node(new_node, feature=np.mean([G_upsampled.nodes[u]['feature'], G_upsampled.nodes[v]['feature']]) )

        G_upsampled.add_edges_from([(u, new_node), (new_node, v)])
        yield G_upsampled

def get_upsample_nodes(G, n_samples):
    "Get a list of nodes to upsample a graph G to n_samples nodes by randomly resampling nodes."
    n_original = G.number_of_nodes()
    if n_samples == n_original:
        return []
    elif n_samples < n_original:
        raise ValueError("n_samples must be greater than the number of nodes in G.")
    n_samples = n_samples - n_original

    n_mod = n_samples % n_original
    n_rep = n_samples // n_original

    sampled_nodes = []
    for j in range(n_rep):
        sampled_nodes += list(np.random.permutation(list(G.nodes())))  # Add all original nodes multiple times
    # Randomly sample nodes with replacement
    sampled_nodes += list(np.random.choice(list(G.nodes()), size=n_mod, replace=False))
    #all_nodes = list(G.nodes()) + [str(j)+"_new" for j in range(n_samples)]
    return sampled_nodes

def upsample_triangles(G, n_samples):
    n_original = G.number_of_nodes()
    triangles = [list(c) for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    n_triangles = len(triangles)
    if n_triangles == 0:
        raise ValueError("No triangles to upsample.")
    
    if n_samples == n_original:
        return G #[]
    elif n_samples < n_original:
        raise ValueError("n_samples must be greater than the number of nodes in G.")
    n_samples = n_samples - n_original

    #n_mod = n_samples % n_triangles
    #n_rep = n_samples // n_triangles
    n_nodes_upsampled = 0
    n_rep=0
    n_mod=0
    while n_nodes_upsampled < n_samples:
        if n_nodes_upsampled + n_triangles > n_samples:
            n_mod = (n_samples - n_nodes_upsampled) #// n_triangles
            n_nodes_upsampled = n_samples
            #n_rep = 1
        else:
            n_nodes_upsampled += n_triangles 
            n_triangles = n_triangles * 3
            n_rep += 1

    n_upsample = 0
    for j in range(n_rep):
        perm_triangles = list(np.random.permutation(triangles)) 
        for t in perm_triangles:
            try:
                G.add_node(str(n_upsample)+"_new", feature=np.mean([G.nodes[node]['feature'] for node in t], axis=0))
            except:
                G.add_node(str(n_upsample)+"_new")
            for node in t:
                G.add_edge(str(n_upsample)+"_new", node)
            n_upsample += 1
        triangles = [list(c) for c in nx.enumerate_all_cliques(G) if len(c) == 3]

    #for j in range(n_mod):
    #print(triangles)
    sampled_t_indices = list(np.random.choice(list(range(len(triangles))), size=n_mod, replace=False))
    sampled_t = [triangles[i] for i in sampled_t_indices]
    for t in sampled_t:
        try:
            G.add_node(str(n_upsample)+"_new", feature=np.mean([G.nodes[node]['feature'] for node in t], axis=0))
        except:
            G.add_node(str(n_upsample)+"_new")
        for node in t:
            G.add_edge(str(n_upsample)+"_new", node)
        n_upsample += 1
    #triangles = [list(c) for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    #if n_mod 
    return G

def upsample_distances_linear(D, n_samples):
    """Upsample a distance matrix D to a larger size n_samples using linear interpolation."""
    from scipy.interpolate import interp2d

    n_original = D.shape[0]
    if n_samples <= n_original:
        raise ValueError("n_samples must be greater than the size of D.")

    # Create original grid
    x_original = np.linspace(0, n_original - 1, n_original)
    y_original = np.linspace(0, n_original - 1, n_original)

    # Create new grid
    x_new = np.linspace(0, n_original - 1, n_samples)
    y_new = np.linspace(0, n_original - 1, n_samples)

    # Interpolate
    interp_func = interp2d(x_original, y_original, D, kind='linear')
    D_upsampled = interp_func(x_new, y_new)

    return D_upsampled

def upsample_distances_random(D, n_samples, method="noise"):
    "Upsample distance matrix D to n_samples by randomly resampling rows and adding noise."
    
    n_original = D.shape[0]
    if n_samples <= n_original:
        raise ValueError("n_samples must be greater than the size of D.")

    # Randomly sample indices with replacement
    sampled_indices = np.random.choice(n_original, size=n_samples, replace=True)

    # Create the upsampled distance matrix
    D_upsampled = D[sampled_indices][:, sampled_indices]

    # Add small random noise to distances to avoid duplicates
    noise = np.random.normal(loc=0.0, scale=np.std(D), size=D_upsampled.shape)
    D_upsampled += noise

    # Ensure the distance matrix remains symmetric and has zeros on the diagonal
    D_upsampled = (D_upsampled + D_upsampled.T) / 2
    np.fill_diagonal(D_upsampled, 0)

    return D_upsampled


#  ╭──────────────────────────────────────────────────────────╮
#  │ Edge Pooling based Graph Coarsening / Upsampling         │
#  ╰──────────────────────────────────────────────────────────╯


def mag_edge_pooling(g, ts, dist_fn, original_magni=None, n_steps=None, method="cholesky", scores_method ="full", 
                     get_weights=False, upsample_to=None, upsample_mode = 'node_duplicating', compute_at_cardinality=None, mode="structure", upsample_method="nodes"):
    ### This function is used to drop nodes from a graph in order to minimise the magnitude difference after removal.

    magnitudes = []
    n_nodes = []
    weight_list = []

    n = g.number_of_nodes()


    if compute_at_cardinality is None:
        compute_here  = True
        min_cardinality = 1
    elif n in compute_at_cardinality:
        compute_here  = True
        min_cardinality = min(compute_at_cardinality)
    else:
        compute_here  = False
        min_cardinality = min(compute_at_cardinality)

    if ts == "conv":
        try:
            n_sub = nx.number_connected_components(g)
            target_value = max(n_sub , g.number_of_nodes()*0.5)# / 2)
            mag0,ts=compute_magnitude_graph(g, dist_fn, ts=None,target_value=target_value,n_ts=2, scale_finding="convergence", method=method, get_weights=get_weights)
            ts = [ts[-1]]
        except:
            ts = [1]
    elif ts == "median":
        mag0,ts=compute_magnitude_graph(g, dist_fn, ts=None,target_value=None,n_ts=2, scale_finding="median", method=method, get_weights=get_weights)
        ts = [ts[-1]]
    #else:
    #    if compute_here:
    #        mag0,_=compute_magnitude_graph(g, dist_fn, ts=ts, method=method, get_weights=get_weights)

    #print(ts)
    ### only compute mag at this scale if relevant!
    #if upsample_to is not None:
    #    if g.number_of_nodes() <= upsample_to:
    #        upsampled_nodes = get_upsample_nodes(g, upsample_to)
    #        g = upsample_graph(g, upsampled_nodes)

    if (scores_method in ["full", "approx_all"]) or (compute_here):
        if original_magni is None:
            original_magni, _, subgraphs, Ds, mags = compute_magnitude_subgraphs_with_dist(g, dist_fn=dist_fn, ts=ts, get_weights=False, method=method)
            #if get_weights:
            #    original_magni = np.sum(mag0)
            #else:
            #    original_magni = mag0
            if compute_here:
                mag0,_ = compute_magnitude_graph(g, dist_fn, ts=ts,method=method, get_weights=get_weights)
    this_graph = g.copy()

    if n_steps is None:
        n_steps = n-1

    edges = list(this_graph.edges()) 
    X = np.array(
                [this_graph.nodes[node]["feature"] for node in this_graph.nodes()]
            )
    
    #can_upsample = n<upsample_to

    if (upsample_to is not None):
        # and original_magni is not None:
        can_upsample = n<upsample_to

        if can_upsample:

            if upsample_method == "nodes":
                upsampled_nodes = get_upsample_nodes(this_graph, upsample_to)

                G_upsampled = this_graph.copy()
                for k in range(len(upsampled_nodes)):
                    G_upsampled = upsample_graph(G_upsampled, [upsampled_nodes[k]], lables=[str(k)+ "_upsample"])

                    if compute_at_cardinality is None:
                        compute_here  = True
                    elif G_upsampled.number_of_nodes() in compute_at_cardinality:
                        compute_here  = True
                    else:
                        compute_here  = False
                    
                    if compute_here:
                        mag, ts =compute_magnitude_graph(G_upsampled, dist_fn, ts=ts,method=method, get_weights=get_weights)
                        #mags.append(mag)
                        if False:
                            for this_upsample in range(n, upsample_to+1):
                                mags = []
                                upsample_ratio = this_upsample / n
                                n_upsample = 0
                                #Ds_upsampled = []
                                if False:
                                    for D in Ds:
                                        upsample_this = int(round(D.shape[0] * upsample_ratio))
                                        D_upsampled = upsample_distances(D, upsample_this)
                                        #Ds_upsampled.append(D_upsampled)
                                        mag = compute_magnitude_from_distances(
                                            D_upsampled,
                                            ts=ts,
                                            method=method,
                                            get_weights=get_weights,
                                            input_distances=True,
                                        )
                                        mags.append(mag)
                                        n_upsample += D_upsampled.shape[0]
                                else:
                                    for subgraph in subgraphs:
                                        upsample_this = int(round(subgraph.number_of_nodes() * upsample_ratio))
                                        G_upsampled = upsample_graph(subgraph, upsample_this)
                                        D_upsampled = dist_fn(G=G_upsampled)
                                        #Ds_upsampled.append(D_upsampled)
                                        mag = compute_magnitude_from_distances(
                                            D_upsampled,
                                            ts=ts,
                                            method=method,
                                            get_weights=get_weights,
                                            input_distances=True,
                                        )
                                        mags.append(mag)
                                        n_upsample += D_upsampled.shape[0]
                        if get_weights:
                            this_magnitude = np.sum(mag)
                            magnitudes.append(this_magnitude)
                            weight_list.append(mag)
                        else:
                            #total_magnitude = np.sum([mag for mag in mags], axis=0)
                            this_magnitude = mag[0]
                            magnitudes.append(this_magnitude)
                            weight_list.append(None)
                        n_nodes.append(G_upsampled.number_of_nodes())

            elif upsample_method == "triangles":
                n_upsampled_nodes = upsample_to - n
            
                G_upsampled = upsample_triangles(this_graph.copy(), upsample_to)
                #print(f"Upsampled to {G_upsampled.number_of_nodes()} nodes by adding {n_upsampled_nodes} nodes through triangle upsampling.")
                    
                #G_upsampled = this_graph.copy()
                for k in range(n_upsampled_nodes):
                    #print(f"Processing node {k}")
                    if k>0:
                        G_upsampled.remove_node(str(n_upsampled_nodes-k)+"_new")

                    #print(f"{G_upsampled.number_of_nodes()} nodes")
                    #= G_upsampled.copy()
                    #print()
                    if compute_at_cardinality is None:
                        compute_here  = True
                    elif G_upsampled.number_of_nodes() in compute_at_cardinality:
                        compute_here  = True
                    else:
                        compute_here  = False

                    if compute_here:
                        #G_upsampled = upsample_triangles(this_graph.copy(), upsample_to)
                        mag, ts =compute_magnitude_graph(G_upsampled, dist_fn, ts=ts,method=method, get_weights=get_weights)

                        if get_weights:
                            this_magnitude = np.sum(mag)
                            magnitudes.append(this_magnitude)
                            weight_list.append(mag)
                        else:
                            #total_magnitude = np.sum([mag for mag in mags], axis=0)
                            this_magnitude = mag[0]
                            magnitudes.append(this_magnitude)
                            weight_list.append(None)
                        n_nodes.append(G_upsampled.number_of_nodes())
                #Ds = Ds_upsampled

    
    if compute_at_cardinality is None:
        compute_here  = True
        min_cardinality = 1
    elif n in compute_at_cardinality:
        compute_here  = True
        min_cardinality = min(compute_at_cardinality)
    else:
        compute_here  = False
        min_cardinality = min(compute_at_cardinality)

    if compute_here:
        if get_weights:
            #weights, this_ts = compute_magnitude_subgraphs(this_graph, dist_fn=dist_fn, ts=ts, get_weights=True, method=method)
            this_magnitude = np.sum(mag0)
            magnitudes.append(this_magnitude)
            weight_list.append(mag0)

        else:
            #this_magnitude, this_ts = compute_magnitude_subgraphs(this_graph, dist_fn=dist_fn, ts=ts, get_weights=False, method=method)
            this_magnitude = mag0[0]
            magnitudes.append(this_magnitude)
            weight_list.append(None)
        n_nodes.append(n)
    
    ## get_scores_edge(this_graph, edges, dist_fn, ts, original_magni, subgraphs, mags, method="cholesky")

    if scores_method == "full":
        scores = get_scores_edge(this_graph, edges, dist_fn, ts, original_magni, subgraphs, mags, method=method)
    elif scores_method == "approx_all":
        scores = get_scores_edge_approx_dist(this_graph, edges, dist_fn, ts, original_magni, subgraphs, Ds, mags,  method=method)
    elif scores_method == "random":
        scores = np.random.rand(len(edges)).tolist()

    n_steps = min(n_steps, len(scores))
    order_this = np.argsort(scores).flatten().astype(int).tolist()
    scores = [scores[i] for i in order_this]
    edges = [edges[i] for i in order_this]
    
    if mode != "structure":
        S = np.eye(n)

    ### Edge Pooling ####
    i = 0
    for l in range(0, n_steps):
        min_score_indices = np.flatnonzero(np.isclose(scores, scores[0], rtol=1e-8, atol=1e-12))
        chosen_index = random.choice(min_score_indices)  # Randomly select one of the indices
        chosen_edge = edges[chosen_index]
        node_a, node_b = chosen_edge

        this_nodes = list(this_graph.nodes())
        #cluster[all_nodes.index(node_b)] = cluster[all_nodes.index(node_a)]
        indx_a = this_nodes.index(node_a)
        indx_b = this_nodes.index(node_b)
        # Merge node_a and node_b into node_a
        
        if mode != "structure":
            if 0 <= indx_b < S.shape[0]:  # Ensure indx_b is within bounds
                S[indx_a, :] += S[indx_b, :]
                S = np.delete(S, (indx_b), axis=0)
            else:
                raise IndexError(f"Index indx_b={indx_b} is out of bounds for array S with shape {S.shape}")

        #this_graph = nx.contracted_edge(this_graph, chosen_edge, self_loops=False)
        scores.pop(chosen_index)
        edges.pop(chosen_index)

        # Remove the edge from the graph
        neighbors_a = set(this_graph.neighbors(node_a)) - {node_b}
        neighbors_b = set(this_graph.neighbors(node_b)) - {node_a}
        merged_neighbors = neighbors_b.difference(neighbors_a)

        # Connect node_a to the neighbors of both nodes
        for neighbor in merged_neighbors:
            this_graph.add_edge(node_a, neighbor)

        # Remove node_b and its edges
        this_graph.remove_node(node_b)

        ### Track Magnitude ####
        if compute_at_cardinality is None:
            compute_here  = True
        elif this_graph.number_of_nodes() in compute_at_cardinality:
            compute_here  = True
        else:
            compute_here  = False
        
        if compute_here:
            if mode != "structure":
                row_sums = S.sum(axis=1, keepdims=True)
                sel = S / row_sums
                X_new = sel @ X
                for n in range(len(this_graph.nodes())):
                    these_nodes = list(this_graph.nodes())
                    this_graph.nodes[these_nodes[n]]["feature"] = X_new[n]
            
            if get_weights:
                weights, this_ts = compute_magnitude_subgraphs(this_graph, dist_fn=dist_fn, ts=ts, get_weights=True, method=method)
                this_magnitude = np.sum(weights)
                magnitudes.append(this_magnitude)
            else:
                weights = None
                this_magnitude, this_ts = compute_magnitude_subgraphs(this_graph, dist_fn=dist_fn, ts=ts, get_weights=False, method=method)
                magnitudes.append(this_magnitude[0])
            n_nodes.append(this_graph.number_of_nodes())
            weight_list.append(weights)

        #### Remove edges involving node_b from the list of edges ####
        if this_graph.number_of_edges() == 0:
            break

        if this_graph.number_of_edges() < min_cardinality:
            break

        new_edges = [] # updating edgelist to draw new edge to pool, we do not want to collapse an edge involving a vertex adjacent to an already pooled edge?
        new_scores = []
        for i, edge in enumerate(edges):
            if (node_b in edge) | (node_a in edge):
                continue
            else:
                new_edge = edge
                new_edges.append(new_edge)
                new_scores.append(scores[i])

        #magnitudes.append(this_magnitude)
        #weight_list.append(mag0)
        #else:
        #    #this_magnitude, this_ts = compute_magnitude_subgraphs(this_graph, dist_fn=dist_fn, ts=ts, get_weights=False, method=method)
        #    this_magnitude = mag0[0]
        #    #magnitudes.append(this_magnitude)
        #    #weight_list.append(None)
        # n_nodes.append(n)
        edges = new_edges
        scores = new_scores

        if (len(edges) == 0) & (l < n_steps - 1):
            break

    # Normalize each row in S by its sum
    if mode != "structure":
        row_sums = S.sum(axis=1, keepdims=True)
        S = S / row_sums
    else:
        S = np.eye(this_graph.number_of_nodes())

    return this_graph, S, magnitudes, n_nodes, weight_list

def repeat_edge_dropping(g, ts, dist_fn, original_magni=None, n_steps=None, method="cholesky", scores_method ="full", get_weights=False, upsample_to=None, upsample_mode = 'node_duplicating', compute_at_cardinality=None, mode="structure", upsample_method="nodes"):
    n_nodes = g.number_of_nodes()
    n_nodes_list = []
    weight_list = []

    if n_steps is None:
        n_steps = n_nodes - 1

    S = np.eye(n_nodes)
    this_graph = g.copy()
    magnitudes = []

    if compute_at_cardinality is not None:
        min_cardinality = min(compute_at_cardinality)
    else:
        min_cardinality = 2

    step=0
    n_nodes_now = this_graph.number_of_nodes()+1
    # print(f'n nodes now_{n_nodes_now}')
    while (this_graph.number_of_nodes() > (g.number_of_nodes()+1-n_steps)) and (this_graph.number_of_edges() > 0) and (this_graph.number_of_nodes() >= min_cardinality) and (this_graph.number_of_nodes() < n_nodes_now):
        n_nodes_now = this_graph.number_of_nodes()
        to_do = n_steps - (n_nodes- n_nodes_now) #n_steps = n_nodes_now-1 if not otherwise specifies, n_nodes is fix number of nodes of orig graph, so usually this is just n_nodes_now - 1
        # print("nodes: ",this_graph.number_of_nodes(),f'\nto do: {to_do}\nn_steps: {n_steps}')

        if step == 0:
            step += 1

            # print(f'First time calling mag_edge_pooling, doing upsampling')
            this_graph, S_new, magnitudes_new, n_nodes_new, weights_new = mag_edge_pooling(this_graph, ts, dist_fn, original_magni=original_magni, n_steps=to_do, method=method, scores_method=scores_method, get_weights=get_weights, upsample_to=upsample_to, upsample_mode = upsample_mode, compute_at_cardinality=compute_at_cardinality, mode=mode, upsample_method=upsample_method)

        else:
            # print(f'Calling mag_edge_pooling in step {step}, number of nodes are now {n_nodes_now} and number of edges at the moment are now {this_graph.number_of_edges()}')
            this_graph, S_new, magnitudes_new, n_nodes_new, weights_new = mag_edge_pooling(this_graph, ts, dist_fn, n_steps=to_do, method=method, scores_method=scores_method, get_weights=get_weights, compute_at_cardinality=compute_at_cardinality, mode=mode)
            step += 1
            # print(f'After mag_edge_pooling in step {step}, number of nodes are now {this_graph.number_of_nodes()} and number of edges at the moment are now {this_graph.number_of_edges()}')
            # print(f'this_graph.number_of_nodes() > (g.number_of_nodes()+1-n_steps): {this_graph.number_of_nodes()} > {(g.number_of_nodes()+1-n_steps)} {this_graph.number_of_nodes() > (g.number_of_nodes()+1-n_steps)}\n(this_graph.number_of_edges() > 0): {(this_graph.number_of_edges() > 0)}\n(this_graph.number_of_nodes() >= min_cardinality): {(this_graph.number_of_nodes() >= min_cardinality)}\n(this_graph.number_of_nodes() < n_nodes_now): {(this_graph.number_of_nodes() < n_nodes_now)}')
        magnitudes = magnitudes + magnitudes_new
        n_nodes_list = n_nodes_list + n_nodes_new
        weight_list = weight_list + weights_new
        if mode != "structure":
            S = np.dot(S_new,S)
        

    return this_graph, S, magnitudes, n_nodes_list, weight_list


#  ╭──────────────────────────────────────────────────────────╮
#  │ Magnitude Filtration                                     │
#  ╰──────────────────────────────────────────────────────────╯

#def magnitude_filtration(g, dist_fn, ts=None, method="cholesky", get_weights=False, upsample_to=None, downsample_to=None, compute_at_cardinality=None):#
#
#    magnitudes = []
#    n_nodes = []
#    weight_list = []
