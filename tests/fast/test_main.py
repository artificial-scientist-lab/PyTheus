import abc
import unittest
from filecmp import cmp
from typing import List

from theseus import main
from theseus.main import read_config, get_dimensions_and_target_state, build_starting_graph


class TestMainModule(unittest.TestCase):

    def test_read_config_from_example_dir_with_json_ending(self):
        config, filename = read_config(is_example=True, filename='ghz_346.json')

        self.assertEqual(
            config['target_state'], ["000", "111", "222", "333"]
        )

    def test_read_config_from_example_dir_without_json_ending(self):
        config, filename = read_config(is_example=True, filename='ghz_346')

        self.assertEqual(
            config['target_state'], ["000", "111", "222", "333"]
        )
        self.assertEqual('ghz_346.json', filename.name)

    def test_build_starting_graph(self):
        cnfg = read_config(is_example=True, filename='ghz_346')
        confi = {'bulk_thr': 0.01, 'edges_tried': 20, 'foldername': 'ghz_346', 'ftol': 1e-06, 'loss_func': 'cr',
         'num_anc': 3, 'num_pre': 1, 'optimizer': 'L-BFGS-B', 'imaginary': False, 'safe_hist': True,
         'samples': 1, 'target_state': ['000', '111', '222', '333'], 'thresholds': [0.25, 0.1],
         'tries_per_edge': 5, 'unicolor': False}
        dimension_key = [4, 4, 4, 1, 1, 1]
        expected_outcome = {(0, 1, 0, 0): True, (0, 1, 0, 1): True, (0, 1, 0, 2): True, (0, 1, 0, 3): True,
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
        actual = build_starting_graph(cnfg, dimension_key)
        self.assertEqual(87, len(actual))
        self.assertEqual(dimension_key, actual.dimensions)
        self.assertEqual(expected_outcome, actual.graph)
        self.assertEqual(list(expected_outcome.values()), actual.weights)

    def test_get_dimensions_and_target_state(self):
        cnfig = (read_config(is_example=True, filename='ghz_346'))
        cnfg = {'bulk_thr': 0.01, 'edges_tried': 20, 'foldername': 'ghz_346', 'ftol': 1e-06, 'loss_func': 'cr',
         'num_anc': 3, 'num_pre': 1, 'optimizer': 'L-BFGS-B', 'imaginary': False, 'safe_hist': True,
         'samples': 1, 'target_state': ['000', '111', '222', '333'], 'thresholds': [0.25, 0.1],
         'tries_per_edge': 5, 'unicolor': False}
        exp_check_out = {'|000000>': True, '|111000>': True, '|222000>': True, '|333000>': True}
        exp = ([4, 4, 4, 1, 1, 1], None, {((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)): True,
                ((0, 1), (1, 1), (2, 1), (3, 0), (4, 0), (5, 0)): True,
                ((0, 2), (1, 2), (2, 2), (3, 0), (4, 0), (5, 0)): True,
                ((0, 3), (1, 3), (2, 3), (3, 0), (4, 0), (5, 0)): True})
        actual = get_dimensions_and_target_state(cnfg)
        print(actual)
        print(type(actual))
        print(type(exp))
        print(type(exp_check_out))
        print("tup2[2]: ", exp[2])
        print("actual[2]", actual[2])
        print(len(exp))
        #self.assertEqual(exp[0], actual[0])
        #self.assertEqual(exp[1], actual[1])
        self.assertEqual(exp[2], actual[2])

        print(len(actual))
        #self.assertEqual(exp, actual.)
        #print(exp_check_out)
        #print(actual)
#        return self.test_get_dimensions_and_target_state()