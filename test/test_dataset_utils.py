from pathlib import Path
from unittest import TestCase

from toy_graph.util import read_content_file

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestReadContentFile(TestCase):

    def test_read_content_file(self):
        indices, labels, features = read_content_file(DATA_DIR / "cora.content")
        self.assertEqual(indices[1], "1061127")
        self.assertEqual(labels[1], "Rule_Learning")
        self.assertEqual(len(features[1]), 1433)
