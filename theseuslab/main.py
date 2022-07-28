# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:24:53 2022

@author: janpe
"""

import sys
import json

from theseuslab.optimizer import topological_opti
from theseuslab.fancy_classes import Graph, State
from theseuslab.state import state1 as sst
import theseuslab.theseus as th
import theseuslab.help_functions as hf
import theseuslab.saver as saver


def run_main():
    try:
        config = sys.argv[1]
    except IndexError:
        print("No input file given.")
        sys.exit()

    if not config.endswith(".json"):
        config += ".json"
    with open(config) as input_file:
        cnfg = json.load(input_file)

    sys.setrecursionlimit(1000000000)

    if cnfg['loss_func'] == 'ent':
        # concurrence optimization
        # define local dimensions
        dimensions = [int(ii) for ii in str(cnfg['dim'])]
        if len(dimensions) % 2 != 0:
            dimensions.append(1)
        target_state = None
        sys_dict = hf.get_sysdict(dimensions, bipar_for_opti=cnfg['K'],
                                  imaginary=cnfg['imaginary'])
    else:
        # target state optimization
        sys_dict = None
        # add ancillas
        term_list = [term + cnfg['num_anc'] * '0' for term in cnfg['target_state']]
        if 'amplitudes' in cnfg:
            target_state = State(term_list, amplitudes=cnfg['amplitudes'], imaginary=cnfg['imaginary'])
        else:
            target_state = State(term_list, imaginary=cnfg['imaginary'])
        print(hf.readableState(target_state))
        target_kets = target_state.kets
        # define local dimensions
        dimensions = th.stateDimensions(target_kets)

    # build starting graph
    edge_list = th.buildAllEdges(dimensions, imaginary=cnfg['imaginary'])
    edge_list = hf.prepEdgeList(edge_list, cnfg)

    print(f'start graph has {len(edge_list)} edges.')
    start_graph = Graph(edge_list, imaginary=cnfg['imaginary'])

    # topological optimization
    sv = saver.saver(config=cnfg, name_config_file=config, dim=dimensions)
    for i in range(cnfg['samples']):
        optimizer = topological_opti(start_graph, sv, ent_dic=sys_dict, target_state=target_state, config=cnfg)
        graph_res = optimizer.topologicalOptimization()
        sv.save_graph(optimizer)

    graph_res.getState()
    print(f'finished with graph with {len(graph_res.edges)} edges.')
    print(graph_res.state.state)

    ancillas = dimensions.count(1)
    if ancillas != 0:
        end_res = dict()
        for kets, ampl in graph_res.state.state.items():
            end_res[kets[:-ancillas]] = ampl
    else:
        end_res = graph_res.state.state
    result = sst(end_res)
    result.info()


if __name__ == '__main__':
    run_main()
