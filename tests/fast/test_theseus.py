import unittest

from build.lib.theseus.main import read_config
from tests.fast.config import GHZ_346
from theseus.theseus import stateDimensions, buildAllEdges, graphDimensions, findPerfectMatchings, stateCatalog, \
    stringEdges


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
        actual = stringEdges(GHZ_346['edges'], imaginary = False)
        exp = ['w_0_1_0_0', 'w_0_1_0_1', 'w_0_1_0_2', 'w_0_1_0_3', 'w_0_1_1_0', 'w_0_1_1_1', 'w_0_1_1_2', 'w_0_1_1_3',
               'w_0_1_2_0', 'w_0_1_2_1', 'w_0_1_2_2', 'w_0_1_2_3', 'w_0_1_3_0', 'w_0_1_3_1', 'w_0_1_3_2', 'w_0_1_3_3',
               'w_0_2_0_0', 'w_0_2_0_1', 'w_0_2_0_2', 'w_0_2_0_3', 'w_0_2_1_0', 'w_0_2_1_1', 'w_0_2_1_2', 'w_0_2_1_3',
               'w_0_2_2_0', 'w_0_2_2_1', 'w_0_2_2_2', 'w_0_2_2_3', 'w_0_2_3_0', 'w_0_2_3_1', 'w_0_2_3_2', 'w_0_2_3_3',
               'w_0_3_0_0', 'w_0_3_1_0', 'w_0_3_2_0', 'w_0_3_3_0', 'w_0_4_0_0', 'w_0_4_1_0', 'w_0_4_2_0', 'w_0_4_3_0',
               'w_0_5_0_0', 'w_0_5_1_0', 'w_0_5_2_0', 'w_0_5_3_0', 'w_1_2_0_0', 'w_1_2_0_1', 'w_1_2_0_2', 'w_1_2_0_3',
               'w_1_2_1_0', 'w_1_2_1_1', 'w_1_2_1_2', 'w_1_2_1_3', 'w_1_2_2_0', 'w_1_2_2_1', 'w_1_2_2_2', 'w_1_2_2_3',
               'w_1_2_3_0', 'w_1_2_3_1', 'w_1_2_3_2', 'w_1_2_3_3', 'w_1_3_0_0', 'w_1_3_1_0', 'w_1_3_2_0', 'w_1_3_3_0',
               'w_1_4_0_0', 'w_1_4_1_0', 'w_1_4_2_0', 'w_1_4_3_0', 'w_1_5_0_0', 'w_1_5_1_0', 'w_1_5_2_0', 'w_1_5_3_0',
               'w_2_3_0_0', 'w_2_3_1_0', 'w_2_3_2_0', 'w_2_3_3_0', 'w_2_4_0_0', 'w_2_4_1_0', 'w_2_4_2_0', 'w_2_4_3_0',
               'w_2_5_0_0', 'w_2_5_1_0', 'w_2_5_2_0', 'w_2_5_3_0', 'w_3_4_0_0', 'w_3_5_0_0', 'w_4_5_0_0']
        self.assertEqual(exp,actual)
        self.assertEqual(87, len(actual))

    def test_stringEdges_withImaginary_True(self):
        actual = stringEdges(GHZ_346['edges'], imaginary=True)
        exp_out = ['r_0_1_0_0', 'r_0_1_0_1', 'r_0_1_0_2', 'r_0_1_0_3', 'r_0_1_1_0', 'r_0_1_1_1', 'r_0_1_1_2',
                    'r_0_1_1_3', 'r_0_1_2_0', 'r_0_1_2_1', 'r_0_1_2_2', 'r_0_1_2_3', 'r_0_1_3_0', 'r_0_1_3_1',
                    'r_0_1_3_2', 'r_0_1_3_3', 'r_0_2_0_0', 'r_0_2_0_1', 'r_0_2_0_2', 'r_0_2_0_3', 'r_0_2_1_0',
                    'r_0_2_1_1', 'r_0_2_1_2', 'r_0_2_1_3', 'r_0_2_2_0', 'r_0_2_2_1', 'r_0_2_2_2', 'r_0_2_2_3',
                    'r_0_2_3_0', 'r_0_2_3_1', 'r_0_2_3_2', 'r_0_2_3_3', 'r_0_3_0_0', 'r_0_3_1_0', 'r_0_3_2_0',
                    'r_0_3_3_0', 'r_0_4_0_0', 'r_0_4_1_0', 'r_0_4_2_0', 'r_0_4_3_0', 'r_0_5_0_0', 'r_0_5_1_0',
                    'r_0_5_2_0', 'r_0_5_3_0', 'r_1_2_0_0', 'r_1_2_0_1', 'r_1_2_0_2', 'r_1_2_0_3', 'r_1_2_1_0',
                    'r_1_2_1_1', 'r_1_2_1_2', 'r_1_2_1_3', 'r_1_2_2_0', 'r_1_2_2_1', 'r_1_2_2_2', 'r_1_2_2_3',
                    'r_1_2_3_0', 'r_1_2_3_1', 'r_1_2_3_2', 'r_1_2_3_3', 'r_1_3_0_0', 'r_1_3_1_0', 'r_1_3_2_0',
                    'r_1_3_3_0', 'r_1_4_0_0', 'r_1_4_1_0', 'r_1_4_2_0', 'r_1_4_3_0', 'r_1_5_0_0', 'r_1_5_1_0',
                    'r_1_5_2_0', 'r_1_5_3_0', 'r_2_3_0_0', 'r_2_3_1_0', 'r_2_3_2_0', 'r_2_3_3_0', 'r_2_4_0_0',
                    'r_2_4_1_0', 'r_2_4_2_0', 'r_2_4_3_0', 'r_2_5_0_0', 'r_2_5_1_0', 'r_2_5_2_0', 'r_2_5_3_0',
                    'r_3_4_0_0', 'r_3_5_0_0', 'r_4_5_0_0', 'th_0_1_0_0', 'th_0_1_0_1', 'th_0_1_0_2', 'th_0_1_0_3',
                    'th_0_1_1_0', 'th_0_1_1_1', 'th_0_1_1_2', 'th_0_1_1_3', 'th_0_1_2_0', 'th_0_1_2_1', 'th_0_1_2_2',
                    'th_0_1_2_3', 'th_0_1_3_0', 'th_0_1_3_1', 'th_0_1_3_2', 'th_0_1_3_3', 'th_0_2_0_0', 'th_0_2_0_1',
                    'th_0_2_0_2', 'th_0_2_0_3', 'th_0_2_1_0', 'th_0_2_1_1', 'th_0_2_1_2', 'th_0_2_1_3', 'th_0_2_2_0',
                    'th_0_2_2_1', 'th_0_2_2_2', 'th_0_2_2_3', 'th_0_2_3_0', 'th_0_2_3_1', 'th_0_2_3_2', 'th_0_2_3_3',
                    'th_0_3_0_0', 'th_0_3_1_0', 'th_0_3_2_0', 'th_0_3_3_0', 'th_0_4_0_0', 'th_0_4_1_0', 'th_0_4_2_0',
                    'th_0_4_3_0', 'th_0_5_0_0', 'th_0_5_1_0', 'th_0_5_2_0', 'th_0_5_3_0', 'th_1_2_0_0', 'th_1_2_0_1',
                    'th_1_2_0_2', 'th_1_2_0_3', 'th_1_2_1_0', 'th_1_2_1_1', 'th_1_2_1_2', 'th_1_2_1_3', 'th_1_2_2_0',
                    'th_1_2_2_1', 'th_1_2_2_2', 'th_1_2_2_3', 'th_1_2_3_0', 'th_1_2_3_1', 'th_1_2_3_2', 'th_1_2_3_3',
                    'th_1_3_0_0', 'th_1_3_1_0', 'th_1_3_2_0', 'th_1_3_3_0', 'th_1_4_0_0', 'th_1_4_1_0', 'th_1_4_2_0',
                    'th_1_4_3_0', 'th_1_5_0_0', 'th_1_5_1_0', 'th_1_5_2_0', 'th_1_5_3_0', 'th_2_3_0_0', 'th_2_3_1_0',
                    'th_2_3_2_0', 'th_2_3_3_0', 'th_2_4_0_0', 'th_2_4_1_0', 'th_2_4_2_0', 'th_2_4_3_0', 'th_2_5_0_0',
                    'th_2_5_1_0', 'th_2_5_2_0', 'th_2_5_3_0', 'th_3_4_0_0', 'th_3_5_0_0', 'th_4_5_0_0']
        self.assertEqual(exp_out, actual)
        self.assertEqual(174, len(actual))
        print(actual[1])
        print(actual)
        print(type(actual))
        print(len(actual))
