import json
import unittest
from random import random

import numpy as np
from numpy.random import RandomState

from build.lib.theseus.main import read_config
from tests.fast.config import GHZ_346
from pytheus.theseus import stateDimensions, buildAllEdges, graphDimensions, findPerfectMatchings, stateCatalog, \
    stringEdges, allPerfectMatchings, allEdgeCovers, allColorGraphs, buildRandomGraph, nodeDegrees, edgeBleach, \
    targetEdges, removeNodes, recursiveEdgeCover, findEdgeCovers, edgeWeight, weightProduct, writeNorm, targetEquation, \
    compute_entanglement, buildLossString


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
        exp_value = [((0, 1, 0, 0), (2, 3, 3, 0), (4, 5, 0, 0)), ((0, 1, 0, 0), (2, 4, 3, 0), (3, 5, 0, 0)),
                     ((0, 1, 0, 0), (2, 5, 3, 0), (3, 4, 0, 0)), ((0, 2, 0, 3), (1, 3, 0, 0), (4, 5, 0, 0)),
                     ((0, 2, 0, 3), (1, 4, 0, 0), (3, 5, 0, 0)), ((0, 2, 0, 3), (1, 5, 0, 0), (3, 4, 0, 0)),
                     ((0, 3, 0, 0), (1, 2, 0, 3), (4, 5, 0, 0)), ((0, 3, 0, 0), (1, 4, 0, 0), (2, 5, 3, 0)),
                     ((0, 3, 0, 0), (1, 5, 0, 0), (2, 4, 3, 0)), ((0, 4, 0, 0), (1, 2, 0, 3), (3, 5, 0, 0)),
                     ((0, 4, 0, 0), (1, 3, 0, 0), (2, 5, 3, 0)), ((0, 4, 0, 0), (1, 5, 0, 0), (2, 3, 3, 0)),
                     ((0, 5, 0, 0), (1, 2, 0, 3), (3, 4, 0, 0)), ((0, 5, 0, 0), (1, 3, 0, 0), (2, 4, 3, 0)),
                     ((0, 5, 0, 0), (1, 4, 0, 0), (2, 3, 3, 0))]
        self.assertIn(key, actual)
        self.assertEqual(exp_value, actual[key])

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

    def test_targetEdges(self):
        input = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        actual = targetEdges([0], input)
        print(actual)
        print(len(actual))
        self.assertEqual([(0, 1), (0, 2), (0, 3)], actual)
        self.assertEqual(3, len(actual))

    def test_removeNodes(self):
        node = (0, 2)
        graph_input = [(0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5),
                       (3, 4), (3, 5), (4, 5)]
        actual = removeNodes(node, graph_input)
        self.assertEqual([(1, 3), (1, 4), (1, 5), (3, 4), (3, 5), (4, 5)], actual)
        self.assertEqual(6, len(actual))

    def test_recursiveEdgeCover(self):
        graph_input = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        possible_edge_covers = []
        actual = recursiveEdgeCover(graph_input, possible_edge_covers)
        self.assertIn([(0, 3), (1, 2), (1, 2)], possible_edge_covers)
        self.assertEqual([(0, 1), (0, 2), (1, 3)], possible_edge_covers[2])
        self.assertEqual([(0, 2), (1, 2), (1, 3)], possible_edge_covers[12])
        self.assertEqual(22, len(possible_edge_covers))

    def test_findEdgeCovers(self):
        graph_input = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        actual = findEdgeCovers(graph_input, order=1)
        self.assertIn([(0, 3), (1, 2), (1, 2)], actual)
        self.assertEqual([(0, 1), (0, 2), (1, 3)], actual[2])
        self.assertEqual([(0, 2), (0, 3), (1, 3)], actual[12])
        self.assertEqual(22, len(actual))

    def test_edgeWeight_withimaginary_false(self):
        actual = edgeWeight(((0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1)), imaginary=False)
        self.assertIn('(0, 1, 1, 0)', actual)
        self.assertIn('w_(0, 1, 0, 0)', actual)

    def test_edgeWeight_withimaginary_cartesian(self):
        actual = edgeWeight(((0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1)), imaginary='cartesian')
        self.assertIn('r_(0, 1, 0, 0)', actual)
        self.assertIn('(0, 1, 1, 1)+1j*th_(0, 1, 0, 0)', actual)

    def test_edgeWeight_withimaginary_polar(self):
        actual = edgeWeight(((0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1)), imaginary='polar')
        self.assertIn('r_(0, 1, 0, 0)', actual)
        self.assertIn('(0, 1, 1, 1)*np.exp(1j*th_(0, 1, 0, 0)', actual)

    def test_weightProduct_withimaginary_false(self):
        actual = weightProduct(((0, 3, 0, 0), (1, 2, 0, 1)), imaginary=False)
        self.assertEqual('w_0_3_0_0*w_1_2_0_1', actual)

    def test_weightProduct_withimaginary_polar(self):
        actual = weightProduct(((0, 3, 0, 0), (1, 2, 0, 1)), imaginary='polar')
        self.assertEqual('r_0_3_0_0*np.exp(1j*th_0_3_0_0)*r_1_2_0_1*np.exp(1j*th_1_2_0_1)', actual)

    def test_weightProduct_withimaginary_cartesian(self):
        actual = weightProduct(((0, 3, 0, 0), (1, 2, 0, 1)), imaginary='cartesian')
        self.assertEqual('(r_0_3_0_0+1j*th_0_3_0_0)*(r_1_2_0_1+1j*th_1_2_0_1)', actual)

    def test_writeNorm_withimaginary_false(self):
        actual = writeNorm({((0, 0), (1, 0)): [((0, 1, 0, 0),)], ((0, 1), (1, 1)): [((0, 1, 1, 1),)]}, imaginary=False)
        exp_out = '1*((w_0_1_0_0)**2) + ((w_0_1_1_1)**2)'
        self.assertEqual(exp_out, actual)

    def test_writeNorm_withimaginary_cartesian(self):
        actual = writeNorm({((0, 0), (1, 0)): [((0, 1, 0, 0),)], ((0, 1), (1, 1)): [((0, 1, 1, 1),)]},
                           imaginary='cartesian')
        exp_out = '1*(abs((r_0_1_0_0+1j*th_0_1_0_0))**2) + (abs((r_0_1_1_1+1j*th_0_1_1_1))**2)'
        self.assertEqual(exp_out, actual)

    def test_writeNorm_withimaginary_polar(self):
        actual = writeNorm({((0, 0), (1, 0)): [((0, 1, 0, 0),)], ((0, 1), (1, 1)): [((0, 1, 1, 1),)]},
                           imaginary='polar')
        exp_out = '1*(abs(r_0_1_0_0*np.exp(1j*th_0_1_0_0))**2) + (abs(r_0_1_1_1*np.exp(1j*th_0_1_1_1))**2)'
        self.assertEqual(exp_out, actual)

    def test_targetEquation_withimaginary_false(self):
        ket_input = [((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)), ((0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 0)),
                     ((0, 1), (1, 0), (2, 1), (3, 1), (4, 0), (5, 0)), ((0, 1), (1, 1), (2, 1), (3, 0), (4, 0), (5, 0))]
        amp = [True, True, True, True]
        actual = targetEquation(ket_input, imaginary=False)
        self.assertIn('w_0_1_0_0*w_2_5_0_0*w_3_4_0_0', actual)
        self.assertIn('w_0_1_1_0*w_2_4_1_0*w_3_5_1_0', actual)
        self.assertIn('w_0_5_1_0*w_1_4_1_0*w_2_3_1_0))**2)/4', actual)
        self.assertEqual(1934, len(actual))
        print(actual)
        print(len(actual))
        print(type(actual))

    def test_targetEquation_withimaginary_cartesian(self):
        ket_input = [((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)), ((0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 0)),
                     ((0, 1), (1, 0), (2, 1), (3, 1), (4, 0), (5, 0)), ((0, 1), (1, 1), (2, 1), (3, 0), (4, 0), (5, 0))]
        actual = targetEquation(ket_input, None, state_catalog=None, imaginary='cartesian')
        self.assertIn('(abs(((r_0_1_0_0+1j*th_0_1_0_0)*(r_2_3_0_0+1j*th_2_3_0_0)*(r_4_5_0_0+1j*th_4_5_0_0)', actual)
        self.assertIn('(r_0_3_1_1+1j*th_0_3_1_1)*(r_1_5_0_0+1j*th_1_5_0_0)*(r_2_4_1_0+1j*th_2_4_1_0)', actual)
        self.assertIn('(r_0_5_1_0+1j*th_0_5_1_0)*(r_1_4_1_0+1j*th_1_4_1_0)*(r_2_3_1_0+1j*th_2_3_1_0)))**2)/4', actual)
        self.assertEqual(4817, len(actual))

