import functools
from typing import Dict

import networkx as nx
from gensim.models import Word2Vec
from joblib import Parallel
from joblib import delayed

from toy_graph.util import biased_random_walk
from toy_graph.util import partition


class Node2vec:

    def __init__(self,
                 p=0.5,
                 q=2.0,
                 num_walks=10,
                 walk_length=20,
                 epochs=10,
                 window_size=5,
                 embed_size=128,
                 weight_attr_name=None):
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.epochs = epochs
        self.window_size = window_size
        self.embed_size = embed_size
        self.weight_attr_name = weight_attr_name
        self.model: Word2Vec = None

    def _random_walk(self,
                     graph: nx.Graph,
                     n_jobs=1):
        walk_paths = Parallel(n_jobs=n_jobs)(
            delayed(biased_random_walk)(graph,
                                        self.p,
                                        self.q,
                                        parti_num_walks,
                                        self.walk_length,
                                        self.weight_attr_name)
            for parti_num_walks in partition(self.num_walks, n_jobs)
        )

        flatten_paths = functools.reduce(lambda x, y: x + y, walk_paths)
        return flatten_paths

    def fit(self,
            graph: nx.Graph,
            n_jobs=1):
        walk_paths = [[str(node) for node in walk] for walk in self._random_walk(graph, n_jobs)]
        self.model = Word2Vec(walk_paths,
                              vector_size=self.embed_size,
                              window=self.window_size,
                              min_count=1,
                              workers=4,
                              epochs=self.epochs)

    def save_model(self,
                   model_path: str):
        if self.model is None:
            raise Exception('the model has not been trained.')
        self.model.save(model_path)

    def get_node_embedding(self) -> Dict:
        if self.model is None:
            raise Exception('the model has not been trained.')
        index2node = self.model.wv.index_to_key
        index2embedding = self.model.wv.vectors
        node_embedding = {}
        for idx, node in enumerate(index2node):
            embedding = index2embedding[idx]
            node_embedding[node] = embedding
        return node_embedding
