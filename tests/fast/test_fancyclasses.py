import unittest

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from config import EDGES
from theseus.fancy_classes import Graph, defaultValues

class TestFancyClassesModule(unittest.TestCase):

    def test_defaultValues_not_imaginary(self):
        actual = defaultValues(87, False)
        self.assertEqual([True] * 87, actual)

    def test_defaultValues_cartesian(self):
        actual = defaultValues(10, 'cartesian')
        self.assertEqual([True] * 10, actual)

    def test_defaultValues_polar(self):
        actual = defaultValues(2, 'polar')
        self.assertEqual([(True, False), (True, False)], actual)

    def test_defaultValues_raises_exception_on_invalid_input(self):
        with self.assertRaises(ValueError):
            defaultValues(10, 'blabla')


class TestGraph(unittest.TestCase):

    def test_init_graph_edges_given_not_imaginary_rest_default(self):
        graph = Graph(EDGES, imaginary=False)

        expected_graph_attribute = {(0, 1, 0, 0): True, (0, 1, 0, 1): True, (0, 1, 0, 2): True, (0, 1, 0, 3): True,
                                    (0, 1, 1, 0): True, (0, 1, 1, 1): True, (0, 1, 1, 2): True, (0, 1, 1, 3): True,
                                    (0, 1, 2, 0): True, (0, 1, 2, 1): True, (0, 1, 2, 2): True, (0, 1, 2, 3): True,
                                    (0, 1, 3, 0): True, (0, 1, 3, 1): True, (0, 1, 3, 2): True, (0, 1, 3, 3): True,
                                    (0, 2, 0, 0): True, (0, 2, 0, 1): True, (0, 2, 0, 2): True, (0, 2, 0, 3): True,
                                    (0, 2, 1, 0): True, (0, 2, 1, 1): True, (0, 2, 1, 2): True, (0, 2, 1, 3): True,
                                    (0, 2, 2, 0): True, (0, 2, 2, 1): True, (0, 2, 2, 2): True, (0, 2, 2, 3): True,
                                    (0, 2, 3, 0): True, (0, 2, 3, 1): True, (0, 2, 3, 2): True, (0, 2, 3, 3): True,
                                    (0, 3, 0, 0): True, (0, 3, 1, 0): True, (0, 3, 2, 0): True, (0, 3, 3, 0): True,
                                    (0, 4, 0, 0): True, (0, 4, 1, 0): True, (0, 4, 2, 0): True, (0, 4, 3, 0): True,
                                    (0, 5, 0, 0): True, (0, 5, 1, 0): True, (0, 5, 2, 0): True, (0, 5, 3, 0): True,
                                    (1, 2, 0, 0): True, (1, 2, 0, 1): True, (1, 2, 0, 2): True, (1, 2, 0, 3): True,
                                    (1, 2, 1, 0): True, (1, 2, 1, 1): True, (1, 2, 1, 2): True, (1, 2, 1, 3): True,
                                    (1, 2, 2, 0): True, (1, 2, 2, 1): True, (1, 2, 2, 2): True, (1, 2, 2, 3): True,
                                    (1, 2, 3, 0): True, (1, 2, 3, 1): True, (1, 2, 3, 2): True, (1, 2, 3, 3): True,
                                    (1, 3, 0, 0): True, (1, 3, 1, 0): True, (1, 3, 2, 0): True, (1, 3, 3, 0): True,
                                    (1, 4, 0, 0): True, (1, 4, 1, 0): True, (1, 4, 2, 0): True, (1, 4, 3, 0): True,
                                    (1, 5, 0, 0): True, (1, 5, 1, 0): True, (1, 5, 2, 0): True, (1, 5, 3, 0): True,
                                    (2, 3, 0, 0): True, (2, 3, 1, 0): True, (2, 3, 2, 0): True, (2, 3, 3, 0): True,
                                    (2, 4, 0, 0): True, (2, 4, 1, 0): True, (2, 4, 2, 0): True, (2, 4, 3, 0): True,
                                    (2, 5, 0, 0): True, (2, 5, 1, 0): True, (2, 5, 2, 0): True, (2, 5, 3, 0): True,
                                    (3, 4, 0, 0): True, (3, 5, 0, 0): True, (4, 5, 0, 0): True}
