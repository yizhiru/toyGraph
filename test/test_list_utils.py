from unittest import TestCase

from toy_graph.util import partition


class TestPartition(TestCase):

    def test_even_split(self):
        self.assertEqual(partition(6, 3), [2, 2, 2])

    def test_uneven_split(self):
        self.assertEqual(partition(11, 3), [4, 4, 3])

    def test_fewer_items_than_partitions(self):
        self.assertEqual(partition(2, 3), [1, 1])

    def test_single_partition(self):
        self.assertEqual(partition(10, 1), [10])

    def test_sum_equals_total(self):
        for num, parts in [(11, 3), (2, 3), (6, 3), (7, 4)]:
            self.assertEqual(sum(partition(num, parts)), num)
