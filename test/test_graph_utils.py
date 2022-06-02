from unittest import TestCase
import networkx as nx
from toy_graph import util


class Test(TestCase):

    def setUp(self) -> None:
        self.graph = nx.Graph([(1, 2),
                               (1, 3),
                               (1, 4),
                               (2, 3),
                               (2, 5)])

    def test_biased_random_walk(self):
        walk_path = util.biased_random_walk(self.graph, p=2.0, num_walks=2, walk_length=3)
        print(walk_path)

