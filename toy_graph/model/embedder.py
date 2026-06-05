import itertools
from typing import Dict, Optional

import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from joblib import Parallel, delayed

from toy_graph.util import biased_random_walk, partition


class RandomWalkEmbedder:
    """Random-walk graph embedder using Word2Vec (DeepWalk / node2vec)."""

    def __init__(
        self,
        p: float = 1.0,
        q: float = 1.0,
        num_walks: int = 10,
        walk_length: int = 20,
        epochs: int = 10,
        window_size: int = 5,
        embed_size: int = 128,
        weight_attr_name: Optional[str] = None,
        random_state: Optional[int] = None,
    ):
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.epochs = epochs
        self.window_size = window_size
        self.embed_size = embed_size
        self.weight_attr_name = weight_attr_name
        self.random_state = random_state
        self.model: Optional[Word2Vec] = None

    def _random_walk(self, graph: nx.Graph, n_jobs: int = 1) -> list:
        parts = partition(self.num_walks, n_jobs)
        n_workers = min(n_jobs, len(parts))

        walk_paths = Parallel(n_jobs=n_workers)(
            delayed(biased_random_walk)(
                graph,
                self.p,
                self.q,
                parti_num_walks,
                self.walk_length,
                self.weight_attr_name,
                self.random_state + i if self.random_state is not None else None,
            )
            for i, parti_num_walks in enumerate(parts)
        )
        return list(itertools.chain.from_iterable(walk_paths))

    def fit(self, graph: nx.Graph, n_jobs: int = 1) -> "RandomWalkEmbedder":
        walk_paths = [
            [str(node) for node in walk]
            for walk in self._random_walk(graph, n_jobs)
        ]
        workers = max(1, n_jobs)
        self.model = Word2Vec(
            walk_paths,
            vector_size=self.embed_size,
            window=self.window_size,
            min_count=1,
            workers=workers,
            epochs=self.epochs,
            seed=self.random_state if self.random_state is not None else 0,
        )
        return self

    def save_model(self, model_path: str) -> None:
        if self.model is None:
            raise RuntimeError("the model has not been trained.")
        self.model.save(model_path)

    def load_model(self, model_path: str) -> "RandomWalkEmbedder":
        self.model = Word2Vec.load(model_path)
        return self

    def get_node_embedding(self) -> Dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("the model has not been trained.")
        return {
            node: self.model.wv.vectors[idx]
            for idx, node in enumerate(self.model.wv.index_to_key)
        }
