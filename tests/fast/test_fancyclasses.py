import unittest
from pathlib import Path

from tests.fast.config import GHZ_346, BELL
from pytheus.fancy_classes import Graph, defaultValues
from pytheus.main import run_main, read_config


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



