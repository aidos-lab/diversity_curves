import networkx as nx
import random
import numpy as np


def est_moment(graph, number_of_walks, number_of_steps):
    node_list = list(nx.nodes(graph))
    moments = np.zeros(number_of_steps)
    for i in range(0, number_of_walks):
        node = random.choice(node_list)
        w = node
        for step in range(0, number_of_steps):
            nbr_list = list(nx.all_neighbors(graph, w))
            if len(nbr_list) == 0:
                # W is an isolated nodes
                break
            w = random.choice(nbr_list)
            if w == node:
                moments[step] = moments[step] + 1
    moments = moments / number_of_walks
    return moments
