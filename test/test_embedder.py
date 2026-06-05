from unittest import TestCase

import networkx as nx

from toy_graph import DeepWalk, Node2vec


class TestEmbedder(TestCase):

    def setUp(self) -> None:
        self.graph = nx.Graph([
            ("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c"),
        ])

    def test_deepwalk_fit_and_embedding(self):
        model = DeepWalk(
            num_walks=2,
            walk_length=5,
            epochs=1,
            embed_size=8,
            random_state=42,
        )
        model.fit(self.graph, n_jobs=1)
        embeddings = model.get_node_embedding()
        self.assertEqual(len(embeddings), self.graph.number_of_nodes())
        for node in self.graph.nodes():
            self.assertIn(str(node), embeddings)
            self.assertEqual(embeddings[str(node)].shape, (8,))

    def test_node2vec_reproducible_with_seed(self):
        params = dict(
            num_walks=2,
            walk_length=5,
            epochs=1,
            embed_size=8,
            random_state=7,
        )
        emb_a = Node2vec(**params).fit(self.graph.copy(), n_jobs=1).get_node_embedding()
        emb_b = Node2vec(**params).fit(self.graph.copy(), n_jobs=1).get_node_embedding()
        for node in emb_a:
            self.assertTrue((emb_a[node] == emb_b[node]).all())

    def test_untrained_raises(self):
        model = DeepWalk()
        with self.assertRaises(RuntimeError):
            model.get_node_embedding()
