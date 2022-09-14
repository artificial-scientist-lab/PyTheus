import unittest
from pathlib import Path

from tests.fast.config import GHZ_346, BELL
from theseus.fancy_classes import Graph, defaultValues
from theseus.main import run_main, read_config


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
        graph = Graph(BELL['edges'], imaginary=False)

        expected_graph_attribute = {(0, 1, 0, 0): True, (0, 1, 1, 1): True}
        self.assertEqual(expected_graph_attribute, graph.graph)
        self.assertEqual([((0, 1, 0, 0),), ((0, 1, 1, 1),)], graph.perfect_matchings)
        self.assertEqual({
            ((0, 0), (1, 0)): [((0, 1, 0, 0),)], ((0, 1), (1, 1)): [((0, 1, 1, 1),)]
        }, graph.state_catalog)

    def test_graphStarter(self):
        cnfg, filename = read_config(is_example=True, filename='cnot_22.json')
        input = {(0, 1, 0, 0): True, (0, 1, 0, 1): True, (0, 1, 1, 0): True,
                 (0, 1, 1, 1): True, (0, 2, 0, 0): True, (0, 2, 0, 1): True,
                 (0, 2, 1, 0): True, (0, 2, 1, 1): True, (0, 3, 0, 0): True,
                 (0, 3, 0, 1): True, (0, 3, 1, 0): True, (0, 3, 1, 1): True,
                 (1, 2, 0, 0): True, (1, 2, 0, 1): True, (1, 2, 1, 0): True,
                 (1, 2, 1, 1): True, (1, 3, 0, 0): True, (1, 3, 0, 1): True,
                 (1, 3, 1, 0): True, (1, 3, 1, 1): True, (2, 3, 0, 0): True,
                 (2, 3, 0, 1): True, (2, 3, 1, 0): True, (2, 3, 1, 1): True}
        expected_sorted_edge = [(0, 1, 0, 0), (0, 1, 1, 1), (0, 1, 2, 2), (0, 1, 3, 3), (0, 2, 0, 0), (0, 2, 1, 1),
                                (0, 2, 2, 2), (0, 2, 3, 3), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (1, 2, 0, 0),
                                (1, 2, 1, 1), (1, 2, 2, 2), (1, 2, 3, 3), (1, 3, 0, 0), (1, 4, 0, 0), (1, 5, 0, 0),
                                (2, 3, 0, 0), (2, 4, 0, 0), (2, 5, 0, 0), (3, 4, 0, 0), (3, 5, 0, 0), (4, 5, 0, 0)]
        #actual = Graph.graphStarter(self,expected_sorted_edge,len(expected_sorted_edge))
        #print(actual)
        input_val = [(0, 1, 0, 0), (0, 1, 1, 1)]