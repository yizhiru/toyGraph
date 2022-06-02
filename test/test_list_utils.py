from unittest import TestCase
from toy_graph.util import partition


class Test(TestCase):

    def test_partition(self):
        print(partition(6, 3))
        self.assertEqual(partition(11, 3), [3, 4, 4])
        self.assertEqual(partition(2, 3), [1, 1])
        self.assertEqual(partition(6, 3), [2, 2, 2])
