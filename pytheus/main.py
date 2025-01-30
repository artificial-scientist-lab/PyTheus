# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:24:53 2022

@author: janpe
"""

import json
import os.path
import sys
from pathlib import Path

import pkg_resources
import logging

log = logging.getLogger(__name__)

import pytheus
import pytheus.help_functions as hf
import pytheus.saver as saver
import pytheus.theseus as th
from pytheus.fancy_classes import Graph, State
from pytheus.optimizer import topological_opti
import itertools
import numpy as np
import random


def run_main(filename, example, run_opt=True, state_cat=True):
    """Run the Theseus algorithm on a given input file.

    Parameters
    ----------
    filename: str
        case name or input file path
    example: bool
        flag indicating whether to run included example case or external file.


    Raises
    ------
    IOError
        if filename is not valid.
    """
    # step 1: read in config file
    cnfg, filename = read_config(example, filename)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info(filename)
    if 'description' in cnfg.keys():
        logging.info(cnfg['description'])

    sys.setrecursionlimit(1000000000)

    # step 2: build up target and starting graph
    if cnfg['loss_func'] == 'ent':  # optimization for entanglement requires specific setup
        dimensions, sys_dict, start_graph = setup_for_ent(cnfg)
        target_state = None
    elif cnfg['loss_func'] == 'lff':  # optimization of a custom loss function
        edge_list = th.buildAllEdges(cnfg["dimensions"], imaginary=cnfg['imaginary'])
        print(f'start graph has {len(edge_list)} edges.')
        start_graph = Graph(edge_list, imaginary=cnfg['imaginary'])
        dimensions = cnfg["dimensions"]
        target_state = None
        sys_dict = None
    elif cnfg['loss_func'] in ['fockcr', 'fockfid']:
        # ADD SETUP FOR FOCK OPTIMIZATION HERE
        # start_graph, target_state, dimensions = setup_for_fockbasis()
        sys_dict = None
        target_state, dimensions, sys_dict, start_graph = setup_for_fockbasis(cnfg)

    else:  # optimization for target given in config
        # read out target and starting graph from cnfg
        # modifies cnfg to incorporate topological constraints
        target_state, start_graph, cnfg = setup_for_target(cnfg, state_cat=state_cat)
        # target_state is state object
        # start_graph is graph object
        dimensions = cnfg["dimensions"]
        sys_dict = None

    # step 3: start optimization
    if run_opt:
        optimize_graph(cnfg, dimensions, filename, start_graph, sys_dict, target_state)
    else:
        return cnfg, target_state


def optimize_graph(cnfg, dimensions, filename, start_graph, sys_dict, target_state):
    '''
    main optimization routine

    Parameters
    ----------
    cnfg
    dimensions
    filename
    start_graph
    sys_dict
    target_state

    Returns
    -------
    result graph
    '''
    # initialize saver object, keeps track of loss history, writes solutions to files, writes best/summary file
    sv = saver.saver(config=cnfg, name_config_file=filename, dim=dimensions)
    # iterate over number samples
    for i in range(cnfg['samples']):
        if i == 0:
            seed = cnfg['seed']
        else:
            random.seed()
            seed = random.randrange(1, 2 ** 32 - 1)

        random.seed(seed)
        np.random.seed(seed=seed)
        cnfg['seed'] = seed

        # initialize optimizer object, do preoptimization on complete graph, truncate graph according to bulk_thr
        optimizer = topological_opti(start_graph, sv, ent_dic=sys_dict, target_state=target_state, config=cnfg)
        if cnfg['topopt']:
            # topological optimization (deleting edges one by one)
            graph_res = optimizer.topologicalOptimization()
        else:
            # if topological optimization not wanted, return graph after optimization on complete graph
            graph_res = optimizer.graph
        # write solution to file
        sv.save_graph(optimizer)
        # compute state of result graph
        graph_res.getState()
        print(f'finished with graph with {len(graph_res.edges)} edges.')
        print(graph_res.state.state)
    return graph_res


def setup_for_fockbasis(cnfg):
    try:
        if cnfg["amplitudes"]:
            print('amplitudes = ', cnfg["amplitudes"])
        else:
            print('amplitudes left empty, assuming constant values of one')
    except KeyError:
        print('amplitudes not given, assuming constant values of one')
        cnfg["amplitudes"] = []

    try:
        if cnfg["imaginary"]:
            print('imaginary given: ', cnfg["imaginary"])
        else:
            print('real numbers used')
    except KeyError:
        print('imaginary not given, assuming real numbers.')
        cnfg["imaginary"] = False

    sys_dict = None

    # term_list = [term + cnfg['num_anc'] * '1' for term in cnfg["target_state"]]
    term_list = []
    for term in cnfg["target_state"]:
        ket = []
        for ii, tt in enumerate(term):
            ket += [(ii, 0)] * tt
        # ket = [ for ii, tt in enumerate(term)]
        for ii in range(cnfg['num_anc']):
            ket.append((len(term) + ii, 0))
        term_list.append(tuple(ket))
    # print(np.shape(term_list))
    num_out = len(cnfg["target_state"][0])
    cnfg["out_nodes"] = list(range(num_out))
    cnfg["in_nodes"] = []
    cnfg["single_emitters"] = []
    cnfg["verts"] = list(range(num_out + cnfg["num_anc"]))

    # not the corrected target_state but has been modified in the loss function
    # this can be changed afterwards
    target_state = State(term_list, amplitudes=cnfg['amplitudes'], imaginary=cnfg['imaginary'])

    # print(hf.readableState(target_state))
    num_mode_particle = len(cnfg["target_state"][0])
    dimensions = [1] * (num_mode_particle + cnfg['num_anc'])  # only one dimension at the moment

    edge_list = th.buildAllEdges(dimensions, imaginary=cnfg["imaginary"], loops=cnfg["loops"])
    edge_list = hf.prepEdgeList(edge_list, cnfg)
    print(f'start graph has {len(edge_list)} edges.')
    start_graph = Graph(edge_list, imaginary=cnfg['imaginary'])
    print(dimensions)
    return target_state, dimensions, sys_dict, start_graph


def setup_for_ent(cnfg):
    # concurrence optimization
    # define local dimensions
    dimensions = [int(ii) for ii in str(cnfg['dim'])]
    cnfg['dimensions'] = dimensions
    if len(dimensions) % 2 != 0:
        dimensions.append(1)
    target_state = None
    # compute sys_dict
    sys_dict = hf.get_sysdict(dimensions, bipar_for_opti=cnfg['K'], imaginary=cnfg['imaginary'])
    # build starting graph
    edge_list = th.buildAllEdges(dimensions, imaginary=cnfg['imaginary'])
    edge_list = hf.prepEdgeList(edge_list, cnfg)
    print(f'start graph has {len(edge_list)} edges.')
    start_graph = Graph(edge_list, imaginary=cnfg['imaginary'])
    return dimensions, sys_dict, start_graph


def setup_for_target(cnfg, state_cat=True):
    # default values
    try:
        cnfg["in_nodes"]
    except KeyError:
        cnfg["in_nodes"] = []

    try:
        cnfg["out_nodes"]
    except KeyError:
        cnfg["out_nodes"] = []

    if not cnfg["in_nodes"]:
        if not cnfg["out_nodes"]:
            print('no in/out nodes given. assuming that target terms correspond to out_nodes. state creation mode')
            cnfg["out_nodes"] = list(range(len(cnfg["target_state"][0])))
        else:
            print('no in_nodes given. state creation mode.')
    else:
        if not cnfg["out_nodes"]:
            print('no out_nodes given. assuming that target terms correspond to in_nodes. measurement mode')
        else:
            print('in_nodes and out_nodes given. target terms are read as a logic table for a quantum gate')

    if len(cnfg["out_nodes"]) + len(cnfg["in_nodes"]) != len(cnfg["target_state"][0]):
        print('TARGET DOES NOT MATCH IN_NODES AND cnfg.out_nodes')

    # num_anc gives the number of photons that go into detectors that are not cnfg.out_nodes (including those coming from single photon sources)
    try:
        print(f'number of ancillary photons = {cnfg["num_anc"]}')
    except KeyError:
        print('num_anc not given, assuming that number of ancillary photons = 0')
        cnfg["num_anc"] = 0

    try:
        if cnfg["single_emitters"]:
            print('single_emitters given. nodes corresponding to single photon sources: ', cnfg["single_emitters"])
        else:
            print('single_emitters not given. no single photon sources in setup.')
    except KeyError:
        print('no single photon emitters used')
        cnfg["single_emitters"] = []

    # add num_anc+len(single_emitters) vertices to graph (every ancillary detector and every single emitter needs a node)
    if cnfg["num_anc"] + len(cnfg["out_nodes"]) < len(cnfg["in_nodes"]) + len(cnfg["single_emitters"]):
        print('not enough ancillas given')
    additional_nodes = cnfg["num_anc"] + len(cnfg["single_emitters"])
    if not cnfg["out_nodes"]:
        additional_nodes += len(cnfg["in_nodes"])

    try:
        if cnfg["removed_connections"]:
            print('removed_connections given. additional constraints on the graph.')
            print('removed_connections = ', cnfg["removed_connections"])
        else:
            print('removed_connections not given. no additional constraints on the graph')
    except KeyError:
        print('removed_connections not given. assuming no additional constraints')
        cnfg["removed_connections"] = []

    try:
        if cnfg["unicolor"]:
            print("unicolor simplification used.")
    except KeyError:
        cnfg["unicolor"] = False

    try:
        if cnfg["amplitudes"]:
            print('amplitudes = ', cnfg["amplitudes"])
        else:
            print('amplitudes left empty, assuming constant values of one')
    except KeyError:
        print('amplitudes not given, assuming constant values of one')
        cnfg["amplitudes"] = []

    try:
        if cnfg["imaginary"]:
            print('imaginary given: ', cnfg["imaginary"])
        else:
            print('real numbers used')
    except KeyError:
        print('imaginary not given, assuming real numbers.')
        cnfg["imaginary"] = False

    try:
        if cnfg["heralding_out"]:
            print("heralding_out = True. out_nodes are not detected. ancillary detectors herald the outgoing state")
            try:
                if cnfg["novac"] == True:
                    print("novac = True. we assume the correct number of photons is leaving the setup")
                else:
                    print("novac = False. we assume that the number of photons leaving the setup is not fixed")
            except KeyError:
                print("novac not given, that the number of photons leaving the setup is not fixed")
                cnfg["novac"] = False
        else:
            print("heralding_out = False. out_nodes are detected. outgoing state is post-selected.")
    except KeyError:
        print('heralding_out not given, assuming post-selection')
        cnfg["heralding_out"] = False

    try:
        if cnfg["number_resolving"]:
            print('number resolving detectors used.')
        else:
            print('no number resolving detectors used')
    except KeyError:
        print('no information about photon-number resolving detectors given, assuming none are used')
        cnfg["number_resolving"] = False

    try:
        cnfg["brutal_covers"]
    except KeyError:
        cnfg["brutal_covers"] = False

    try:
        cnfg["bipartite"]
    except KeyError:
        cnfg["bipartite"] = False

    try:
        cnfg["bulk_thr"]
    except KeyError:
        cnfg["bulk_thr"] = 0

    try:
        cnfg["save_hist"]
    except KeyError:
        cnfg["save_hist"] = True

    try:
        cnfg["num_pre"]
    except KeyError:
        cnfg["num_pre"] = 1

    try:
        cnfg["target_state"]
    except KeyError:
        print('no target state given')

    try:
        # define target
        target = [term + additional_nodes * '0' for term in cnfg["target_state"]]
        target_state = State(target, amplitudes=cnfg["amplitudes"], imaginary=cnfg["imaginary"])
        # print readable expression of the target state
        print(hf.readableState(target_state))
    except:
        raise ValueError('Target state had invalid format and could not be created.')


    # build starting graph
    # local dimensions necessary for each node to produce target
    cnfg["dimensions"] = th.stateDimensions(target_state.kets)
    # get complete starting graph according to local dimensions
    edge_list = th.buildAllEdges(cnfg["dimensions"], imaginary=cnfg["imaginary"])
    cnfg["verts"] = np.unique(list(itertools.chain(*th.edgeBleach(edge_list).keys())))
    cnfg["anc_detectors"] = [ii for ii in cnfg["verts"] if
                            ii not in cnfg["out_nodes"] + cnfg["single_emitters"] + cnfg["in_nodes"]]
    # introduce topological constraints
    # start with explicitly removed connections
    removed_connections = cnfg["removed_connections"]
    # add other restrictions imposed by specific kinds of nodes
    disjoint_nodes = cnfg["single_emitters"] + cnfg["in_nodes"]
    removed_connections += [sorted(con) for con in list(itertools.combinations(disjoint_nodes, 2))]
    if cnfg['bipartite']:
        disjoint_nodes = cnfg["out_nodes"] + cnfg['anc_detectors']
        removed_connections += [sorted(con) for con in list(itertools.combinations(disjoint_nodes, 2))]
    edge_list = hf.removeConnections(edge_list, removed_connections)
    # apply unicolor simplification
    if cnfg['unicolor']:
        num_data_nodes = len(cnfg['target_state'][0])
        edge_list = hf.makeUnicolor(edge_list, num_data_nodes)
    print(f'start graph has {len(edge_list)} edges.')

    # turn edge list into graph
    graph = Graph(edge_list, imaginary=cnfg["imaginary"])  # , state_cat=state_cat)
    return target_state, graph, cnfg


def build_starting_graph(cnfg, dimensions):
    # build starting graph
    edge_list = th.buildAllEdges(dimensions, imaginary=cnfg['imaginary'])
    edge_list = hf.prepEdgeList(edge_list, cnfg)
    print(f'start graph has {len(edge_list)} edges.')
    start_graph = Graph(edge_list, imaginary=cnfg['imaginary'])
    return start_graph


def get_dimensions_and_target_state(cnfg):
    # check if we are optimizing for entanglement
    # there is not a concrete target state and you have to define sys_dict
    if cnfg['loss_func'] == 'ent':
        # concurrence optimization
        # define local dimensions
        dimensions = [int(ii) for ii in str(cnfg['dim'])]
        if len(dimensions) % 2 != 0:
            dimensions.append(1)
        target_state = None
        # compute sys_dict
        sys_dict = hf.get_sysdict(dimensions, bipar_for_opti=cnfg['K'],
                                  imaginary=cnfg['imaginary'])
    else:
        # target state optimization
        sys_dict = None
        # add ancillas
        term_list = [term + cnfg['num_anc'] * '0' for term in cnfg['target_state']]
        # include amplitudes in target state if given
        if 'amplitudes' in cnfg:
            target_state = State(term_list, amplitudes=cnfg['amplitudes'], imaginary=cnfg['imaginary'])
        else:
            target_state = State(term_list, imaginary=cnfg['imaginary'])
        # print readable expression of the target state
        print(hf.readableState(target_state))
        target_kets = target_state.kets
        # define local dimensions
        dimensions = th.stateDimensions(target_kets)
    return dimensions, sys_dict, target_state


def read_config(is_example, filename):
    ''''
    read config json and output cnfg dict
    '''

    # option for running files from example folder
    if is_example:
        configs_dir = pkg_resources.resource_filename(pytheus.__name__, "graphs")
        walk = os.walk(configs_dir)
        for root, dirs, files in walk:
            if os.path.basename(root) == filename:
                for file in files:
                    if file.startswith('config'):
                        filename = os.path.join(root, file)
                        break

    # check if filename ends in json, add extension if needed
    if not filename.endswith('.json'):
        filename += '.json'

    # error if file does not exist
    if not os.path.exists(filename) or os.path.isdir(filename):
        raise IOError(f'File does not exist: {filename}')
    # load json into dict
    with open(filename) as input_file:
        cnfg = json.load(input_file)
    # set some default value for some keys of dict
    if 'topopt' not in cnfg:
        cnfg['topopt'] = True
    if 'seed' not in cnfg:
        cnfg['seed'] = random.randrange(1, 2 ** 32 - 1)
    if not cnfg['topopt']:
        cnfg['bulk_thr'] = 0
    return cnfg, filename
