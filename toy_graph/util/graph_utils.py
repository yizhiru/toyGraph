import networkx as nx
import random
from typing import List


def biased_random_walk(g: nx.Graph,
                       p=1.0,
                       q=1.0,
                       num_walks=10,
                       walk_length=20,
                       weight_attr_name=None) -> List[List[int]]:
    """node2vec 随机游走"""
    walk_paths = []
    for _ in range(num_walks):
        graph_nodes = list(g.nodes())
        random.shuffle(graph_nodes)
        for node in graph_nodes:
            source_node, current_node = None, node
            # 单个起始节点随机游走路径
            walks_per = [current_node]
            for _ in range(walk_length - 1):
                neighbors, probabilities = [], []
                for v, e in g.adj[current_node].items():
                    weight = 1.0 if weight_attr_name is None else e[weight_attr_name]
                    if v == source_node:
                        weight /= p
                    elif g.has_edge(source_node, v):
                        weight = weight
                    else:
                        weight /= q
                    neighbors.append(v)
                    probabilities.append(weight)
                norm_total = sum(probabilities)
                probabilities = [float(p) / norm_total for p in probabilities]
                next_node = random.choices(neighbors, weights=probabilities)[0]
                walks_per.append(next_node)
                source_node = current_node
                current_node = next_node
            walk_paths.append(walks_per)
    return walk_paths



