import unittest

from tests.fast.config import GHZ_346
from theseus.theseus import stateDimensions, buildAllEdges, graphDimensions, findPerfectMatchings, stateCatalog


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
