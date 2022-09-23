import unittest
from random import random

import numpy as np
from numpy.random import RandomState

from build.lib.theseus.main import read_config
from tests.fast.config import GHZ_346
from theseus.theseus import stateDimensions, buildAllEdges, graphDimensions, findPerfectMatchings, stateCatalog, \
    stringEdges, allPerfectMatchings, allEdgeCovers, allColorGraphs, buildRandomGraph, nodeDegrees, edgeBleach


class TestTheseusModule(unittest.TestCase):

    def test_stateDimensions_happy_path(self):
        # TODO: Add more tests, e.g. with various invalid input
        kets = [((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)), ((0, 1), (1, 1), (2, 1), (3, 0), (4, 0), (5, 0)),
                ((0, 2), (1, 2), (2, 2), (3, 0), (4, 0), (5, 0)), ((0, 3), (1, 3), (2, 3), (3, 0), (4, 0), (5, 0))]
        expected = [4, 4, 4, 1, 1, 1]
        self.assertEqual(expected, stateDimensions(kets))

    def test_buildAllEdges(self):
        # TODO: Add more tests cases, in particular simple ones where we can exactly enumerate all edges by hand
        dimensions = [4, 4, 4, 1, 1, 1]
        all_edges = buildAllEdges(dimensions, string=False, imaginary=False)
        self.assertEqual(87, len(all_edges))

    def test_buildAllEdges_bell_state(self):
        dimensions = [2, 2]
        all_edges = buildAllEdges(dimensions, string=False, imaginary=False)
        self.assertEqual([
            (0, 1, 0, 0), (0, 1, 0, 1),
            (0, 1, 1, 0), (0, 1, 1, 1)], all_edges)

    def test_graphDimensions(self):
        dimensions = graphDimensions(GHZ_346['edges'])
        expected = [4, 4, 4, 1, 1, 1]
        self.assertEqual(expected, dimensions)

    def test_findPerfectMatchings(self):
        actual = findPerfectMatchings(GHZ_346['edges'])
        self.assertEqual(((0, 1, 0, 0), (2, 3, 0, 0), (4, 5, 0, 0)), actual[0])
        self.assertEqual(((0, 1, 0, 2), (2, 3, 0, 0), (4, 5, 0, 0)), actual[8])
        self.assertEqual(960, len(actual))

    def test_stateCatalog(self):
        graph_list = findPerfectMatchings(GHZ_346['edges'])
        actual = stateCatalog(graph_list)
        self.assertEqual(64, len(actual))
        key = ((0, 0), (1, 0), (2, 3), (3, 0), (4, 0), (5, 0))
        value = [((0, 1, 0, 0), (2, 3, 3, 0), (4, 5, 0, 0)), ((0, 1, 0, 0), (2, 4, 3, 0), (3, 5, 0, 0)),
                 ((0, 1, 0, 0), (2, 5, 3, 0), (3, 4, 0, 0)), ((0, 2, 0, 3), (1, 3, 0, 0), (4, 5, 0, 0)),
                 ((0, 2, 0, 3), (1, 4, 0, 0), (3, 5, 0, 0)), ((0, 2, 0, 3), (1, 5, 0, 0), (3, 4, 0, 0)),
                 ((0, 3, 0, 0), (1, 2, 0, 3), (4, 5, 0, 0)), ((0, 3, 0, 0), (1, 4, 0, 0), (2, 5, 3, 0)),
                 ((0, 3, 0, 0), (1, 5, 0, 0), (2, 4, 3, 0)), ((0, 4, 0, 0), (1, 2, 0, 3), (3, 5, 0, 0)),
                 ((0, 4, 0, 0), (1, 3, 0, 0), (2, 5, 3, 0)), ((0, 4, 0, 0), (1, 5, 0, 0), (2, 3, 3, 0)),
                 ((0, 5, 0, 0), (1, 2, 0, 3), (3, 4, 0, 0)), ((0, 5, 0, 0), (1, 3, 0, 0), (2, 4, 3, 0)),
                 ((0, 5, 0, 0), (1, 4, 0, 0), (2, 3, 3, 0))]
        self.assertIn(key, actual)
        self.assertEqual(value, actual[key])

    def test_stringEdges_withImaginary_False(self):
        actual = stringEdges(GHZ_346['edges'], imaginary=False)
        self.assertIn('w_0_3_2_0', actual)
        self.assertEqual('w_4_5_0_0', actual[86])
        self.assertEqual(87, len(actual))

    def test_stringEdges_withImaginary_True(self):
        actual = stringEdges(GHZ_346['edges'], imaginary=True)
        self.assertIn('r_0_1_0_1', actual)
        self.assertEqual('r_4_5_0_0', actual[86])
        self.assertIn('th_0_1_0_0', actual)
        self.assertEqual('th_4_5_0_0', actual[173])
        self.assertEqual(174, len(actual))

    def test_allPerfectMatchings(self):
        actual = allPerfectMatchings([4, 4, 4, 1, 1, 1])
        exp_value = [((0, 1, 0, 0), (2, 3, 0, 0), (4, 5, 0, 0)), ((0, 1, 0, 0), (2, 4, 0, 0), (3, 5, 0, 0)),
                     ((0, 1, 0, 0), (2, 5, 0, 0), (3, 4, 0, 0)), ((0, 2, 0, 0), (1, 3, 0, 0), (4, 5, 0, 0)),
                     ((0, 2, 0, 0), (1, 4, 0, 0), (3, 5, 0, 0)), ((0, 2, 0, 0), (1, 5, 0, 0), (3, 4, 0, 0)),
                     ((0, 3, 0, 0), (1, 2, 0, 0), (4, 5, 0, 0)), ((0, 3, 0, 0), (1, 4, 0, 0), (2, 5, 0, 0)),
                     ((0, 3, 0, 0), (1, 5, 0, 0), (2, 4, 0, 0)), ((0, 4, 0, 0), (1, 2, 0, 0), (3, 5, 0, 0)),
                     ((0, 4, 0, 0), (1, 3, 0, 0), (2, 5, 0, 0)), ((0, 4, 0, 0), (1, 5, 0, 0), (2, 3, 0, 0)),
                     ((0, 5, 0, 0), (1, 2, 0, 0), (3, 4, 0, 0)), ((0, 5, 0, 0), (1, 3, 0, 0), (2, 4, 0, 0)),
                     ((0, 5, 0, 0), (1, 4, 0, 0), (2, 3, 0, 0))]
        self.assertIn(((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)), actual.keys())
        self.assertIn(((0, 3), (1, 3), (2, 2), (3, 0), (4, 0), (5, 0)), actual.keys())
        self.assertIn(exp_value, actual.values())
        self.assertEqual(64, len(actual))

    def test_allEdgeCovers(self):
        actual = allEdgeCovers([2, 2])
        val = {((0, 0), (1, 0)): [((0, 1, 0, 0),)], ((0, 0), (1, 1)): [((0, 1, 0, 1),)],
               ((0, 1), (1, 0)): [((0, 1, 1, 0),)], ((0, 1), (1, 1)): [((0, 1, 1, 1),)]}
        self.assertEqual(list(val.keys()), list(actual.keys()))
        self.assertEqual(list(val.values()), list(actual.values()))

    def test_allColorGraphs_Falseloop(self):
        actual = allColorGraphs([(0, 0), (0, 1), (1, 1), (1, 1)], loops=False)
        self.assertEqual([((0, 1, 0, 1), (0, 1, 1, 1))], actual)

    def test_allColorGraphs_Trueloop(self):
        actual = allColorGraphs([(0, 0), (0, 1), (1, 1), (1, 1)], loops=True)
        self.assertEqual(((0, 0, 0, 1), (1, 1, 1, 1)), actual[0])
        self.assertEqual(((0, 1, 0, 1), (0, 1, 1, 1)), actual[1])

    def test_buildRandomGraph(self):
        dim = [4, 4, 4, 1, 1, 1]
        np.random.seed(0)
        actual = buildRandomGraph(dim, 87)
        self.assertIn((2, 4, 1, 0), actual)
        self.assertEqual(87, len(actual))
        self.assertEqual((0, 1, 1, 0), actual[4])
        self.assertEqual((0, 1, 3, 3), actual[15])

    def test_nodeDegrees(self):
        actual = nodeDegrees([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        self.assertEqual([(0, 3), (1, 3), (2, 3), (3, 3)], actual)
        self.assertEqual(4, len(actual))

    def test_edgeBleach(self):
        input = [(0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1), (0, 2, 0, 0), (0, 2, 0, 1), (0, 2, 0, 2),
                 (0, 2, 0, 3), (0, 2, 0, 4), (0, 2, 1, 0), (0, 2, 1, 1), (0, 2, 1, 2), (0, 2, 1, 3), (0, 2, 1, 4),
                 (0, 3, 0, 0), (0, 3, 1, 0), (1, 2, 0, 0), (1, 2, 0, 1), (1, 2, 0, 2), (1, 2, 0, 3), (1, 2, 0, 4),
                 (1, 2, 1, 0), (1, 2, 1, 1), (1, 2, 1, 2), (1, 2, 1, 3), (1, 2, 1, 4), (1, 3, 0, 0), (1, 3, 1, 0),
                 (2, 3, 0, 0), (2, 3, 1, 0), (2, 3, 2, 0), (2, 3, 3, 0), (2, 3, 4, 0)]
        exp_out = {(0, 1): [(0, 0), (0, 1), (1, 0), (1, 1)],
                   (0, 2): [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
                   (0, 3): [(0, 0), (1, 0)],
                   (1, 2): [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
                   (1, 3): [(0, 0), (1, 0)], (2, 3): [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]}
        actual = edgeBleach(input)
        self.assertEqual(exp_out, actual)
        self.assertEqual(6, len(actual))
