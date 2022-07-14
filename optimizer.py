# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 09:07:40 2022

@author: janpe
"""

from fancy_classes import Graph
import config as confi
from lossfunctions import loss_dic
import numpy as np
from scipy import optimize

cnfg = {item: getattr(confi, item) for item in dir(confi)
        if not item.startswith("__") and not item.endswith("__")}


class topological_opti:

    def __init__(self, start_graph: Graph, ent_dic=None, target_state=None):

        self.real = confi.real
        if confi.loss_func == 'ent':
            self.ent_dic = ent_dic
        else:
            self.target = target_state  # object of State class

        self.graph = self.pre_optimize_start_graph(start_graph)

    def check(self, result: object, lossfunctions: object):
        """
        check if all loss functions fulfills conditions for success.

        Parameters
        ----------
        result : object
             from scipy.mimnimizer class
        lossfunctions : object
            list of all loss functions

        Returns
        -------
        bool
            False if we keep Graph or True if we can delete edge

        """

        if confi.loss_func == 'ent':
            if abs(result.fun) - abs(self.loss_val) > confi.thresholds[0]:
                return False
        else:
            # uncomment to see where checks fail
            # print(result.fun, confi.thresholds[0])
            if result.fun > confi.thresholds[0]:
                return False
            # check if all loss functions are under the corresponding threshold
            for ii in range(1, len(lossfunctions)):
                if lossfunctions[ii](result.x) > confi.thresholds[ii]:
                    return False
        # when no check fails return True  = success
        return True

    def get_loss_functions(self, current_graph: Graph):
        """
        get a list of all loss functions mentioned in config

        """
        # get loss function acc. to config
        lossfunctions = loss_dic[confi.loss_func]
        if confi.loss_func == 'ent':  # we optimize for entanglement
            return [func(current_graph, self.ent_dic) for func in lossfunctions]
        else:
            return [func(self.target.kets,
                         current_graph.edges,
                         coefficients=self.target.amplitudes,
                         real=self.real) for func in lossfunctions]

    def pre_optimize_start_graph(self, graph) -> Graph:
        losses = self.get_loss_functions(graph)
        valid = False
        while not valid:
            # find one preoptimization that is valid
            initial_values, bounds = self.prepOptimizer(len(graph))
            best_result = optimize.minimize(losses[0], x0=initial_values,
                                            bounds=bounds,
                                            method=confi.optimizer,
                                            options={'ftol': confi.ftol})
            self.loss_val = best_result.fun
            valid = self.check(best_result, losses)

        for __ in range(confi.num_pre - 1):
            # if stated in config file, do more preoptimizations (esp. useful for concurrence)
            initial_values, bounds = self.prepOptimizer(len(graph))
            result = optimize.minimize(losses[0], x0=initial_values,
                                       bounds=bounds,
                                       method=confi.optimizer,
                                       options={'ftol': confi.ftol})

            if result.fun < best_result.fun:
                best_result = result
        self.loss_val = best_result.fun
        print(f'best result from pre-opt: {abs(best_result.fun)}')

        preopt_graph = Graph(graph.edges, weights=best_result.x)

        try:
            bulk_thr = confi.bulk_thr
        except:
            bulk_thr = 0
        if bulk_thr > 0:
            # cut all edges smaller than bulk_thr and optimize again
            # this can save a lot of time
            cont = True
            num_deleted = 0
            while cont:
                # delete smallest edges one by one
                idx_of_edge = preopt_graph.minimum()
                amplitude = preopt_graph[idx_of_edge]
                if abs(amplitude) < bulk_thr:
                    preopt_graph.remove(idx_of_edge)
                    num_deleted += 1
                else:
                    cont = False
            print(f'{num_deleted} edges deleted')
            valid = False
            while not valid:
                # it is necessary that the truncated graph passes the checks
                initial_values, bounds = self.prepOptimizer(len(preopt_graph))
                losses = self.get_loss_functions(preopt_graph)
                trunc_result = optimize.minimize(losses[0], x0=initial_values,
                                                 bounds=bounds,
                                                 method=confi.optimizer,
                                                 options={'ftol': confi.ftol})
                self.loss_val = trunc_result.fun
                print(f'result after truncation: {abs(trunc_result.fun)}')
                valid = self.check(trunc_result, losses)
            preopt_graph = Graph(preopt_graph.edges, weights=trunc_result.x)

        return preopt_graph

    def prepOptimizer(self, numweights, x=[]):
        '''
        returns initial values and bounds for use in optimization.

        '''

        if self.real:
            bounds = numweights * [(-1, 1)]
            if len(x) == 0:
                initial_values = 2 * np.random.random(numweights) - 1
            else:
                initial_values = x
        else:
            bounds = numweights * [(-1, 1)] + numweights * [(-np.pi, np.pi)]
            if len(x) == 0:
                rands_r = 2 * np.random.random(numweights) - 1
                rands_th = 2 * np.pi * np.random.random(numweights) - np.pi
                initial_values = np.concatenate([rands_r, rands_th])
            else:
                initial_values = x

        return initial_values, bounds

    def termination_condition(self, reps) -> bool:
        """
        conditions that stop optimization

        """
        if confi.loss_func=='ent':
            return len(self.graph) > confi.min_edge and reps < 1
        else:
            return reps < min(len(self.graph), confi.minimal_cycles)

    def optimize_one_edge(self, num_edge: int,
                          num_tries_one_edge: int) -> (Graph, bool):
        """
        delete the num_edge-th smallest edge and optimize num_tries_one_edge times
        and check if corresponding loss function fulfills checks

        """
        # copy current Graph and delete num_edge´s smallest weight
        red_graph = self.graph
        idx_of_edge = red_graph.minimum(num_edge)
        amplitude = red_graph[idx_of_edge]
        red_graph.remove(idx_of_edge)
        for ii in range(num_tries_one_edge):
            if ii == 0:
                try:
                    # redefine loss functions for reduced graph
                    losses = self.get_loss_functions(red_graph)
                    # use initial values x0 from previous Graph
                    x0 = red_graph.weights
                    initial_values, bounds = self.prepOptimizer(len(red_graph),
                                                                x=x0)
                    result = optimize.minimize(losses[0], x0=initial_values,
                                               bounds=bounds,
                                               method=confi.optimizer,
                                               options={'ftol': confi.ftol})
                except KeyError:
                    red_graph[idx_of_edge] = amplitude
                    return red_graph, False  # no success keep current Graph
            else:
                initial_values, bounds = self.prepOptimizer(len(red_graph))
                result = optimize.minimize(losses[0], x0=initial_values,
                                           bounds=bounds,
                                           method=confi.optimizer,
                                           options={'ftol': confi.ftol})
            valid = self.check(result, losses)

            if valid:  # if criterion is reached then save reduced graph as graph, else leave graph as is
                self.loss_val = result.fun
                return Graph(red_graph.edges, weights=result.x), True
        # all tries failed keep current Graph
        red_graph[idx_of_edge] = amplitude
        return red_graph, False

    def topologicalOptimization(self) -> Graph:
        """
        does the topological main loop and returns optimized Graph
        """
        num_edge = 0
        reps = 0
        graph_history = []
        while self.termination_condition(reps):
            self.graph, success = self.optimize_one_edge(num_edge, 5)
            num_edge += 1
            print(f'deleting edge {num_edge}')
            if success:
                print(f"deleted: {len(self.graph)}  edges left with loss {self.loss_val:.3f}")
                num_edge = 0
                graph_history.append(self.graph)
            if num_edge == len(self.graph):
                reps += 1
                num_edge = 0
        return self.graph
