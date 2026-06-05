import random
from typing import List, Optional

import networkx as nx


def _transition_weight(
    g: nx.Graph,
    source_node,
    target,
    p: float,
    q: float,
    weight: float,
) -> float:
    if source_node is None:
        return weight
    if target == source_node:
        return weight / p
    if g.has_edge(source_node, target):
        return weight
    return weight / q


def biased_random_walk(
    g: nx.Graph,
    p: float = 1.0,
    q: float = 1.0,
    num_walks: int = 10,
    walk_length: int = 20,
    weight_attr_name: Optional[str] = None,
    random_state: Optional[int] = None,
) -> List[List]:
    """Biased random walk for node2vec / DeepWalk."""
    if random_state is not None:
        random.seed(random_state)

    walk_paths = []
    for _ in range(num_walks):
        graph_nodes = list(g.nodes())
        random.shuffle(graph_nodes)
        for node in graph_nodes:
            source_node, current_node = None, node
            walks_per = [current_node]
            for _ in range(walk_length - 1):
                neighbors, probabilities = [], []
                for v, edge_data in g.adj[current_node].items():
                    weight = 1.0 if weight_attr_name is None else edge_data[weight_attr_name]
                    weight = _transition_weight(g, source_node, v, p, q, weight)
                    neighbors.append(v)
                    probabilities.append(weight)
                if not neighbors:
                    break
                norm_total = sum(probabilities)
                probabilities = [prob / norm_total for prob in probabilities]
                next_node = random.choices(neighbors, weights=probabilities)[0]
                walks_per.append(next_node)
                source_node = current_node
                current_node = next_node
            walk_paths.append(walks_per)
    return walk_paths
