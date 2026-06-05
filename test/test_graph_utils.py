from unittest import TestCase

import networkx as nx

from toy_graph.util import biased_random_walk
from toy_graph.util.graph_utils import _transition_weight


class TestBiasedRandomWalk(TestCase):

    def setUp(self) -> None:
        self.graph = nx.Graph([
            (1, 2), (1, 3), (1, 4), (2, 3), (2, 5),
        ])

    def test_first_step_ignores_q_bias(self):
        weight = _transition_weight(self.graph, None, 2, p=1.0, q=100.0, weight=1.0)
        self.assertEqual(weight, 1.0)

    def test_return_to_source_divides_by_p(self):
        weight = _transition_weight(self.graph, 1, 1, p=2.0, q=1.0, weight=1.0)
        self.assertEqual(weight, 0.5)

    def test_distant_neighbor_divides_by_q(self):
        weight = _transition_weight(self.graph, 1, 5, p=1.0, q=4.0, weight=1.0)
        self.assertEqual(weight, 0.25)

    def test_walk_length_and_nodes(self):
        walks = biased_random_walk(
            self.graph,
            num_walks=2,
            walk_length=4,
            random_state=42,
        )
        self.assertEqual(len(walks), 2 * self.graph.number_of_nodes())
        for walk in walks:
            self.assertGreaterEqual(len(walk), 1)
            self.assertLessEqual(len(walk), 4)
            for node in walk:
                self.assertIn(node, self.graph)

    def test_isolated_node_produces_single_node_walk(self):
        graph = nx.Graph([(1, 2)])
        graph.add_node(3)
        walks = biased_random_walk(graph, num_walks=1, walk_length=5, random_state=0)
        isolated_walks = [w for w in walks if w[0] == 3]
        self.assertEqual(len(isolated_walks), 1)
        self.assertEqual(isolated_walks[0], [3])
