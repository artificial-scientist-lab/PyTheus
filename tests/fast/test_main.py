import abc
import json
import sys
import traceback
import unittest
from filecmp import cmp
from typing import List
from pathlib import Path

import numpy as np
from numpy import array

from pytheus import main
from pytheus.help_functions import readableState
from pytheus.main import read_config, get_dimensions_and_target_state, build_starting_graph, setup_for_ent, \
    setup_for_target, setup_for_fockbasis, optimize_graph, run_main


class TestMainModule(unittest.TestCase):
    @unittest.skip # i think this is not necessary anymore
    def test_read_config_from_example_dir_with_json_ending(self):
        config, filename = read_config(is_example=True, filename='config_ghz_346.json')

        self.assertEqual(
            config['target_state'], ["000", "111", "222", "333"]
        )

    def test_read_config_from_example_dir_without_json_ending(self):
        config, filename = read_config(is_example=True, filename='ghz_346')

        self.assertEqual(
            config['target_state'], ["000", "111", "222", "333"]
        )
        self.assertEqual('config_ghz_346', Path(filename).stem)

    def test_build_starting_graph(self):
        cnfg, filename = read_config(is_example=True, filename='ghz_346')
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
        cnfg, filename = read_config(is_example=True, filename='ghz_346')
        exp = ([4, 4, 4, 1, 1, 1], None, {((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)): True,
                                          ((0, 1), (1, 1), (2, 1), (3, 0), (4, 0), (5, 0)): True,
                                          ((0, 2), (1, 2), (2, 2), (3, 0), (4, 0), (5, 0)): True,
                                          ((0, 3), (1, 3), (2, 3), (3, 0), (4, 0), (5, 0)): True})
        actual = get_dimensions_and_target_state(cnfg)
        self.assertEqual([4, 4, 4, 1, 1, 1], actual[0])
        self.assertIsNone(actual[1])
        self.assertEqual(list(exp[2].values()), actual[2].amplitudes)
        self.assertEqual(list(exp[2].keys()), actual[2].kets)

    @unittest.skip #does not exist anymore
    def test_setup_for_ent(self):
        cnfg, filename = read_config(is_example=True, filename='conc_4-3')
        exp = ([2, 2, 2, 2], {'dimensions': [2, 2, 2, 2], 'num_ancillas': 0, 'num_particles': 4,
                              'all_states': [((0, 0), (1, 0), (2, 0), (3, 0)), ((0, 0), (1, 0), (2, 0), (3, 1)),
                                             ((0, 0), (1, 0), (2, 1), (3, 0)), ((0, 0), (1, 0), (2, 1), (3, 1)),
                                             ((0, 0), (1, 1), (2, 0), (3, 0)), ((0, 0), (1, 1), (2, 0), (3, 1)),
                                             ((0, 0), (1, 1), (2, 1), (3, 0)), ((0, 0), (1, 1), (2, 1), (3, 1)),
                                             ((0, 1), (1, 0), (2, 0), (3, 0)), ((0, 1), (1, 0), (2, 0), (3, 1)),
                                             ((0, 1), (1, 0), (2, 1), (3, 0)), ((0, 1), (1, 0), (2, 1), (3, 1)),
                                             ((0, 1), (1, 1), (2, 0), (3, 0)), ((0, 1), (1, 1), (2, 0), (3, 1)),
                                             ((0, 1), (1, 1), (2, 1), (3, 0)), ((0, 1), (1, 1), (2, 1), (3, 1))],
                              'dim_total': 16, 'bipar_for_opti': [([0, 1], [2, 3]), ([1, 2], [0, 3]), ([0, 2], [1, 3])],
                              'imaginary': False}, {(0, 1, 0, 0): True, (0, 1, 0, 1): True, (0, 1, 1, 0): True,
                                                    (0, 1, 1, 1): True, (0, 2, 0, 0): True, (0, 2, 0, 1): True,
                                                    (0, 2, 1, 0): True, (0, 2, 1, 1): True, (0, 3, 0, 0): True,
                                                    (0, 3, 0, 1): True, (0, 3, 1, 0): True, (0, 3, 1, 1): True,
                                                    (1, 2, 0, 0): True, (1, 2, 0, 1): True, (1, 2, 1, 0): True,
                                                    (1, 2, 1, 1): True, (1, 3, 0, 0): True, (1, 3, 0, 1): True,
                                                    (1, 3, 1, 0): True, (1, 3, 1, 1): True, (2, 3, 0, 0): True,
                                                    (2, 3, 0, 1): True, (2, 3, 1, 0): True, (2, 3, 1, 1): True})
        actual = setup_for_ent(cnfg)
        self.assertEqual([2, 2, 2, 2], actual[0])
        self.assertEqual(exp[1], actual[1])
        self.assertEqual(list(exp[1].values()), list(actual[1].values()))
        self.assertEqual(exp[1].keys(), actual[1].keys())
        self.assertEqual(list(exp[2].values()), actual[2].weights)
        self.assertEqual(list(exp[2].keys()), actual[2].edges)

    @unittest.skip
    #added some features to the config. is it possible to check if config contains out_config, so we dont get failing
    #tests if we expand the config features more in the future
    def test_setup_for_target(self):
        cnfg, filename = read_config(is_example=True, filename='cnot_22.json')
        read_state = {'|000000>': True, '|010100>': True, '|101100>': True, '|111000>': True}
        kets = [((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)), ((0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 0)),
                ((0, 1), (1, 0), (2, 1), (3, 1), (4, 0), (5, 0)), ((0, 1), (1, 1), (2, 1), (3, 0), (4, 0), (5, 0))]
        out_config = {'description': 'Postselected CNOT between two qubits. Two ancillary particles from SPDC.',
                      'edges_tried': 30, 'foldername': 'cnot_22', 'ftol': 1e-06, 'loss_func': 'cr', 'num_anc': 2,
                      'optimizer': 'L-BFGS-B', 'imaginary': False, 'safe_hist': True, 'samples': 10,
                      'target_state': ['0000', '0101', '1011', '1110'], 'in_nodes': [0, 1], 'out_nodes': [2, 3],
                      'heralding_out': True, 'novac': True, 'thresholds': [0.3, 0.1], 'tries_per_edge': 5,
                      'topopt': True, 'single_emitters': [], 'removed_connections': [[0, 1]], 'unicolor': False,
                      'amplitudes': [], 'number_resolving': False, 'brutal_covers': False, 'bulk_thr': 0,
                      'save_hist': True, 'num_pre': 1, 'dimensions': [2, 2, 2, 2, 1, 1],
                      'verts': array([0, 1, 2, 3, 4, 5]), 'anc_detectors': [4, 5]}
        graph = {(0, 2, 0, 0): True, (0, 2, 0, 1): True, (0, 2, 1, 0): True, (0, 2, 1, 1): True, (0, 3, 0, 0): True,
                 (0, 3, 0, 1): True, (0, 3, 1, 0): True, (0, 3, 1, 1): True, (0, 4, 0, 0): True, (0, 4, 1, 0): True,
                 (0, 5, 0, 0): True, (0, 5, 1, 0): True, (1, 2, 0, 0): True, (1, 2, 0, 1): True, (1, 2, 1, 0): True,
                 (1, 2, 1, 1): True, (1, 3, 0, 0): True, (1, 3, 0, 1): True, (1, 3, 1, 0): True, (1, 3, 1, 1): True,
                 (1, 4, 0, 0): True, (1, 4, 1, 0): True, (1, 5, 0, 0): True, (1, 5, 1, 0): True, (2, 3, 0, 0): True,
                 (2, 3, 0, 1): True, (2, 3, 1, 0): True, (2, 3, 1, 1): True, (2, 4, 0, 0): True, (2, 4, 1, 0): True,
                 (2, 5, 0, 0): True, (2, 5, 1, 0): True, (3, 4, 0, 0): True, (3, 4, 1, 0): True, (3, 5, 0, 0): True,
                 (3, 5, 1, 0): True, (4, 5, 0, 0): True}
        actual = setup_for_target(cnfg)
        self.assertEqual(list(kets), list(actual[0].kets))
        self.assertTrue(all(actual[0].amplitudes))
        # self.assertEqual(read_state, readableState(actual[0]))
        self.assertEqual(list(graph.values()), actual[1].weights)
        self.assertEqual(list(graph.keys()), actual[1].edges)
        self.assertEqual(out_config.keys(), actual[2].keys())
        self.assertSetEqual(set(map(type, out_config.values())), set(map(type, actual[2].values())))
        self.assertEqual(all(out_config.values()), all(actual[2].values()))

    @unittest.skip #does not exist anymore
    def test_setup_for_fockbasis(self):
        cnfg, filename = read_config(is_example=True, filename='fock_tetrahedron_short.json')
        actual = setup_for_fockbasis(cnfg)
        self.assertEqual([((0, 0), (0, 0), (0, 0), (2, 0)), ((1, 0), (1, 0), (1, 0), (2, 0))], actual[0].kets)
        self.assertEqual([1, 1.4142135623730951], actual[0].amplitudes)
        self.assertEqual([1, 1, 1], actual[1])
        self.assertIsNone(actual[2])
        self.assertEqual([(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (2, 2, 0, 0)],
                         actual[3].edges)
        self.assertTrue(all(actual[3].weights))

    @unittest.skip
    #this fails because the results will generally vary with every run. we did implement a 'seed' option for the config files.
    #when seed is set the result of the first sample will always be the same
    def test_optimize_graph(self):
        cnfg, filename = read_config(is_example=True, filename='werner.json')
        dimension = [2, 2, 5, 1]
        exp_output = {(0, 1, 1, 1): -0.4115376348306805, (1, 2, 0, 1): 0.43097397912168994,
                      (1, 2, 1, 2): 0.4309739796322021, (0, 2, 1, 3): 0.44939847729263027,
                      (1, 3, 0, 0): -0.8812150910946458, (0, 3, 0, 0): -0.9185771000730444,
                      (1, 2, 1, 0): 0.9592641504673297, (2, 3, 4, 0): 0.962075654675881, (0, 2, 1, 0): 1.0}
        t_state = setup_for_target(cnfg)
        np.random.seed(0)
        actual = optimize_graph(cnfg, dimension, filename, build_starting_graph(cnfg, dimension), None, t_state[0])
        self.assertEqual([5,5,5,1], actual.dimensions)
        self.assertEqual(9, len(actual))
        self.assertEqual(exp_output, actual.graph)
