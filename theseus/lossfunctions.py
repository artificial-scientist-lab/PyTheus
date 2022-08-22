# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:54:12 2022

@author: janpe
"""

import theseus.theseus as th
from collections import Counter
import itertools
import numpy as np
from itertools import combinations
from itertools import combinations_with_replacement
from theseus.custom_loss.assembly_index import assembly_index, top_n_assembly
import logging

log = logging.getLogger(__name__)


def brutal_covers(cnfg, graph):
    non_output_verts = [vert for vert in cnfg["verts"] if vert not in cnfg["out_nodes"]]
    # minimum number of edges that can cover non output vertices
    min_edges = len(non_output_verts) // 2
    # maximum number of edges, perfect matching order of whole graph
    max_edges = len(cnfg["verts"]) // 2
    orders = list(range(min_edges, max_edges + 1))

    avail_colors = th.edgeBleach(graph.edges)
    tmp_raw_covers = []
    for num_edges in orders:
        tmp_raw_covers += combinations_with_replacement(list(avail_colors.keys()), num_edges)

    raw_covers = []
    ii = 0
    for cover in tmp_raw_covers:
        ii += 1
        keep = True
        counter = Counter(list(itertools.chain(*cover)))

        # check inputs and single emitters
        for vert in cnfg["in_nodes"] + cnfg["single_emitters"]:
            # if any are covered more than once, don't keep edge cover
            if counter[vert] != 1:
                keep = False

        if cnfg['number_resolving']:
            for vert in cnfg["anc_detectors"]:
                if counter[vert] != 1:
                    keep = False
        else:
            for vert in cnfg["anc_detectors"]:
                if counter[vert] == 0:
                    keep = False

        # if in maximum order, we select for events where the correct total number of photons go into output paths
        if len(cover) == max_edges:
            sum = 0
            for vert in cnfg["out_nodes"]:
                sum += counter[vert]
            if sum != len(cnfg["out_nodes"]):
                keep = False

        # if all conditions are met, use edgecover for norm
        if keep:
            raw_covers.append(cover)
    painted_covers = []
    for cover in raw_covers:
        for coloring in itertools.product(*[avail_colors[edge] for edge in cover]):
            color_cover = [edge + color for edge, color in zip(cover, coloring)]
            painted_covers.append(sorted(color_cover))
    return [[tuple(ed) for ed in graph] for graph in np.unique(painted_covers, axis=0)]


def heralded_covers(cnfg, graph):
    non_output_verts = [vert for vert in cnfg["verts"] if vert not in cnfg["out_nodes"]]
    # minimum number of edges that can cover non output vertices
    min_edges = len(non_output_verts) // 2
    # maximum number of edges, perfect matching order of whole graph
    max_edges = len(cnfg["verts"]) // 2
    orders = list(range(min_edges, max_edges + 1))
    # find edge suitable edge covers for all possible numbers of edges
    tmp_edgecovers = []
    for num_edges in orders:
        tmp_edgecovers += th.findEdgeCovers(graph.edges, nodes_left=non_output_verts, edges_left=num_edges)
    # select for edgecovers that fulfill conditions
    edgecovers = []
    for cover in tmp_edgecovers:
        keep = True
        counter = Counter(list(itertools.chain(*th.edgeBleach(cover).keys())))

        # check inputs and single emitters
        for vert in cnfg["in_nodes"] + cnfg["single_emitters"]:
            # if any are covered more than once, don't keep edge cover
            if counter[vert] != 1:
                keep = False

        if cnfg['number_resolving']:
            for vert in cnfg["anc_detectors"]:
                if counter[vert] != 1:
                    keep = False
        else:
            for vert in cnfg["anc_detectors"]:
                if counter[vert] == 0:
                    keep = False

        # if in maximum order, we select for events where the correct total number of photons go into output paths
        if len(cover) == max_edges:
            sum = 0
            for vert in cnfg["out_nodes"]:
                sum += counter[vert]
            if sum != len(cnfg["out_nodes"]):
                keep = False

        # if all conditions are met, use edgecover for norm
        if keep:
            edgecovers.append(cover)
    return edgecovers


def count_rate(graph, target_state, cnfg):
    # set up target equation
    target = target_state.targetEquation(state_catalog=graph.state_catalog, imaginary=cnfg["imaginary"])
    # get variable names
    variables = th.stringEdges(graph.edges, imaginary=cnfg["imaginary"])

    # non-heralded, post-selection case
    if not cnfg["heralding_out"]:
        # only looking at perfect matchings
        graph.getNorm()
        norm = graph.norm
    # heralded case, more complicated selection rules
    else:
        if not cnfg["brutal_covers"]:
            edgecovers = heralded_covers(cnfg, graph)
        else:
            edgecovers = brutal_covers(cnfg, graph)
        cat = th.stateCatalog(edgecovers)
        norm = th.writeNorm(cat, imaginary=cnfg["imaginary"])
    lambdaloss = "".join(["1-", target, "/(1+", norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func


def fidelity(graph, target_state, cnfg):
    # set up target equation
    target = target_state.targetEquation(state_catalog=graph.state_catalog, imaginary=cnfg["imaginary"])
    # get variable names
    variables = th.stringEdges(graph.edges, imaginary=cnfg["imaginary"])

    # non-heralded, post-selection case
    if not cnfg["heralding_out"]:
        # only looking at perfect matchings
        graph.getNorm()
        norm = graph.norm
    # heralded case, more complicated selection rules
    else:
        if not cnfg["brutal_covers"]:
            edgecovers = heralded_covers(cnfg, graph)
        else:
            edgecovers = brutal_covers(cnfg, graph)
        cat = th.stateCatalog(edgecovers)
        norm = th.writeNorm(cat, imaginary=cnfg["imaginary"])
    lambdaloss = "".join(["1-", target, "/(0+", norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func


def state_countrate(graph, target_state, imaginary=False):
    '''

        returns simplified count rate post-selected for clicks in every outgoing detector. can also deal with
        post-selected gates

        only perfect matchings are considered for the computation of the norm.

        Parameters
        ----------
        graph
        target_state
        imaginary

        Returns
        -------
        callable loss function
        '''
    target = target_state.targetEquation(state_catalog=graph.state_catalog, imaginary=imaginary)
    variables = th.stringEdges(graph.edges, imaginary=imaginary)
    graph.getNorm()
    lambdaloss = "".join(["1-", target, "/(1+", graph.norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func


def state_fidelity(graph, target_state, imaginary=False):
    '''

    returns fidelity post-selected for clicks in every outgoing detector. can also deal with post-selected gates

    only perfect matchings are considered for the computation of the norm.

    Parameters
    ----------
    graph
    target_state
    imaginary

    Returns
    -------
    callable loss function
    '''
    target = target_state.targetEquation(state_catalog=graph.state_catalog, imaginary=imaginary)
    variables = th.stringEdges(graph.edges, imaginary=imaginary)
    graph.getNorm()
    lambdaloss = "".join(["1-", target, "/(0+", graph.norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func

    # TODO fix up lower order terms
    # TODO this could be combined with heralded loss functions


def gate_countrate(graph, target_state, imaginary=False, out_nodes=None):
    '''
    returns simplified count rate for a gate built from SPDC crystals. selects for events where the correct number of
    photons  enter the out_nodes and all detectors for the ancillary photons click (heralding)

    select for edge covers that cover all heralding detectors, including lower order terms. for contributions in the
    perfect matching order, only events are selected for where len(out_nodes) go into out_nodes.

    Parameters
    ----------
    graph
    target_state
    imaginary
    out_nodes

    Returns
    -------
    callable loss function

    '''
    target = target_state.targetEquation(state_catalog=graph.state_catalog, imaginary=imaginary)
    variables = th.stringEdges(graph.edges, imaginary=imaginary)

    verts = np.unique(list(itertools.chain(*th.edgeBleach(graph.edges).keys())))
    nonoutput_verts = [ii for ii in verts if ii not in out_nodes]

    tmp_edgecovers = th.findEdgeCovers(graph.edges, nodes_left=nonoutput_verts, edges_left=len(verts) / 2)
    edgecovers = []
    for ec in tmp_edgecovers:
        counter = Counter(list(itertools.chain(*th.edgeBleach(ec).keys())))
        sum = 0
        for vert in out_nodes:
            sum += counter[vert]
        if sum == len(out_nodes):
            edgecovers.append(ec)

    cat = th.stateCatalog(edgecovers)
    norm = th.writeNorm(cat, imaginary=imaginary)
    lambdaloss = "".join(["1-", target, "/(1+", norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func

    # TODO fix up lower order terms
    # TODO this could be combined with heralded loss functions


def gate_fidelity(graph, target_state, imaginary=False, out_nodes=None):
    '''
    returns fidelity for a gate built from SPDC crystals. selects for events where the correct number of
    photons  enter the out_nodes and all detectors for the ancillary photons click (heralding)

    select for edge covers that cover all heralding detectors, including lower order terms. for contributions in the
    perfect matching order, only events are selected for where len(out_nodes) go into out_nodes.


    Parameters
    ----------
    graph
    target_state
    imaginary
    out_nodes

    Returns
    -------
    callable loss function

    '''
    target = target_state.targetEquation(state_catalog=graph.state_catalog, imaginary=imaginary)
    variables = th.stringEdges(graph.edges, imaginary=imaginary)

    verts = np.unique(list(itertools.chain(*th.edgeBleach(graph.edges).keys())))
    nonoutput_verts = [ii for ii in verts if ii not in out_nodes]

    tmp_edgecovers = th.findEdgeCovers(graph.edges, nodes_left=nonoutput_verts, edges_left=len(verts) / 2)
    edgecovers = []
    for ec in tmp_edgecovers:
        counter = Counter(list(itertools.chain(*th.edgeBleach(ec).keys())))
        sum = 0
        for vert in out_nodes:
            sum += counter[vert]
        if sum == len(out_nodes):
            edgecovers.append(ec)

    cat = th.stateCatalog(edgecovers)
    norm = th.writeNorm(cat, imaginary=imaginary)
    lambdaloss = "".join(["1-", target, "/(0+", norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func


def gate_countrate_se(graph, target_state, imaginary=False, out_nodes=None, in_nodes=None, single_emitters=None):
    '''
    returns simplified count rate for a gate built from single_emitters and SPDC crystals.
    selects for events where the correct number of  photons  enter the out_nodes and all detectors for the
    ancillary photons click (heralding)

    Parameters
    ----------
    graph
    target_state
    imaginary
    out_nodes
    in_nodes
    single_emitters

    Returns
    -------

    '''
    target = target_state.targetEquation(state_catalog=graph.state_catalog, imaginary=imaginary)
    variables = th.stringEdges(graph.edges, imaginary=imaginary)

    verts = np.unique(list(itertools.chain(*th.edgeBleach(graph.edges).keys())))
    nonoutput_verts = [ii for ii in verts if ii not in out_nodes]

    tmp_edgecovers = th.findEdgeCovers(graph.edges, nodes_left=nonoutput_verts, edges_left=len(verts) / 2)
    edgecovers = []
    for ec in tmp_edgecovers:
        keep = True
        counter = Counter(list(itertools.chain(*th.edgeBleach(ec).keys())))
        sum = 0
        for vert in out_nodes:
            sum += counter[vert]
        if sum != len(out_nodes):
            keep = False
        for vert in in_nodes + single_emitters:
            if counter[vert] != 1:
                keep = False
        if keep:
            edgecovers.append(ec)

    cat = th.stateCatalog(edgecovers)
    norm = th.writeNorm(cat, imaginary=imaginary)
    lambdaloss = "".join(["1-", target, "/(1+", norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func


def gate_fidelity_se(graph, target_state, imaginary=False, out_nodes=None, in_nodes=None, single_emitters=None):
    '''
    returns fidelity for a gate built from single_emitters and SPDC crystals.
    selects for events where the correct number of  photons  enter the out_nodes and all detectors for the
    ancillary photons click (heralding)

    Parameters
    ----------
    graph
    target_state
    imaginary
    out_nodes
    in_nodes
    single_emitters

    Returns
    -------

    '''
    target = target_state.targetEquation(state_catalog=graph.state_catalog, imaginary=imaginary)
    variables = th.stringEdges(graph.edges, imaginary=imaginary)

    verts = np.unique(list(itertools.chain(*th.edgeBleach(graph.edges).keys())))
    nonoutput_verts = [ii for ii in verts if ii not in out_nodes]

    tmp_edgecovers = th.findEdgeCovers(graph.edges, nodes_left=nonoutput_verts, edges_left=len(verts) / 2)
    edgecovers = []
    for ec in tmp_edgecovers:
        keep = True
        counter = Counter(list(itertools.chain(*th.edgeBleach(ec).keys())))
        sum = 0
        for vert in out_nodes:
            sum += counter[vert]
        if sum != len(out_nodes):
            keep = False
        for vert in in_nodes + single_emitters:
            if counter[vert] != 1:
                keep = False
        if keep:
            edgecovers.append(ec)

    cat = th.stateCatalog(edgecovers)
    norm = th.writeNorm(cat, imaginary=imaginary)
    lambdaloss = "".join(["1-", target, "/(0+", norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func


def heralded_countrate(graph, target_state, imaginary=False, out_nodes=None):
    '''
    returns simplified count rate of a target state heralded in out_nodes when the detectors belonging to the
    rest of the nodes click. only works with SPDC.

    Parameters
    ----------
    graph
    target_state
    imaginary
    out_nodes

    Returns
    -------
    callable loss function
    '''
    target = target_state.targetEquation(state_catalog=graph.state_catalog, imaginary=imaginary)
    variables = th.stringEdges(graph.edges, imaginary=imaginary)

    verts = np.unique(list(itertools.chain(*th.edgeBleach(graph.edges).keys())))
    nonoutput_verts = [ii for ii in verts if ii not in out_nodes]
    all_edgecovers = []
    orders = (len(verts) - len(nonoutput_verts)) // 2 + 1
    for ii in range(orders):
        log.info('starting EC')
        all_edgecovers += th.findEdgeCovers(graph.edges, nodes_left=nonoutput_verts, edges_left=len(verts) / 2 - ii)
        log.info(len(all_edgecovers))
    cat = th.stateCatalog(all_edgecovers)
    norm = th.writeNorm(cat, imaginary=imaginary)
    lambdaloss = "".join(["1-", target, "/(1+", norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func


def heralded_fidelity(graph, target_state, imaginary=False, out_nodes=None):
    '''
    returns simplified count rate of a target state heralded in out_nodes when the detectors belonging to the
    rest of the nodes click. only works with SPDC.

    Parameters
    ----------
    graph
    target_state
    imaginary
    out_nodes

    Returns
    -------
    callable loss function
    '''
    target = target_state.targetEquation(state_catalog=graph.state_catalog, imaginary=imaginary)
    variables = th.stringEdges(graph.edges, imaginary=imaginary)

    verts = np.unique(list(itertools.chain(*th.edgeBleach(graph.edges).keys())))
    nonoutput_verts = [ii for ii in verts if ii not in out_nodes]
    all_edgecovers = []
    orders = (len(verts) - len(nonoutput_verts)) // 2 + 1
    for ii in range(orders):
        log.info('starting EC')
        all_edgecovers += th.findEdgeCovers(graph.edges, nodes_left=nonoutput_verts, edges_left=len(verts) / 2 - ii)
        log.info(len(all_edgecovers))
    cat = th.stateCatalog(all_edgecovers)
    norm = th.writeNorm(cat, imaginary=imaginary)
    lambdaloss = "".join(["1-", target, "/(0+", norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func


def heralded_countrate_se(graph, target_state, imaginary=False, out_nodes=None, single_emitters=None):
    '''
    returns simplified count rate for a target state created from single photon emitters, heralded by clicks
    in heralding detectors.

    Parameters
    ----------
    graph
    target_state
    imaginary
    out_nodes
    single_emitters

    Returns
    -------
    callable loss
    '''
    # for now no SPDC allowed, only single particle emitters
    target = target_state.targetEquation(state_catalog=graph.state_catalog, imaginary=imaginary)
    variables = th.stringEdges(graph.edges, imaginary=imaginary)

    verts = np.unique(list(itertools.chain(*th.edgeBleach(graph.edges).keys())))
    nonoutput_verts = [ii for ii in verts if ii not in out_nodes]
    # all_edgecovers = th.findEdgeCovers(graph.edges, nodes_left=nonoutput_verts, edges_left=len(single_emitters))

    # hacky way of circumventing slowness of findedgecovers here
    tmp_edgecovers = list(itertools.combinations(graph.edges, len(single_emitters)))
    log.info(len(tmp_edgecovers))

    # number resolving detectors
    all_edgecovers = []
    for ec in tmp_edgecovers:
        keep = True
        counter = Counter(list(itertools.chain(*th.edgeBleach(ec).keys())))
        sum = 0
        for vert in out_nodes:
            sum += counter[vert]
        if sum != len(out_nodes):
            keep = False
        for vert in nonoutput_verts:
            if counter[vert] != 1:
                keep = False
        if keep:
            all_edgecovers.append(ec)

    log.info(len(all_edgecovers))
    cat = th.stateCatalog(all_edgecovers)
    norm = th.writeNorm(cat, imaginary=imaginary)
    lambdaloss = "".join(["1-", target, "/(1+", norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func


def heralded_fidelity_se(graph, target_state, imaginary=False, out_nodes=None, single_emitters=None):
    '''
    returns fidelity for a target state created from single photon emitters, heralded by clicks
    in heralding detectors.

    Parameters
    ----------
    graph
    target_state
    imaginary
    out_nodes
    single_emitters

    Returns
    -------
    callable loss
    '''
    target = target_state.targetEquation(state_catalog=graph.state_catalog, imaginary=imaginary)
    variables = th.stringEdges(graph.edges, imaginary=imaginary)

    verts = np.unique(list(itertools.chain(*th.edgeBleach(graph.edges).keys())))
    nonoutput_verts = [ii for ii in verts if ii not in out_nodes]
    # all_edgecovers = th.findEdgeCovers(graph.edges, nodes_left=nonoutput_verts, edges_left=len(single_emitters))

    # hacky way of circumventing slowness of findedgecovers here
    tmp_edgecovers = list(itertools.combinations(graph.edges, len(single_emitters)))
    log.info(len(tmp_edgecovers))
    # number resolving detectors
    all_edgecovers = []
    for ec in tmp_edgecovers:
        keep = True
        counter = Counter(list(itertools.chain(*th.edgeBleach(ec).keys())))
        sum = 0
        for vert in out_nodes:
            sum += counter[vert]
        if sum != len(out_nodes):
            keep = False
        for vert in nonoutput_verts:
            if counter[vert] != 1:
                keep = False
        if keep:
            all_edgecovers.append(ec)

    log.info(len(all_edgecovers))
    cat = th.stateCatalog(all_edgecovers)
    norm = th.writeNorm(cat, imaginary=imaginary)
    lambdaloss = "".join(["1-", target, "/(0+", norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func


def make_lossString_entanglement(graph, sys_dict: dict, imaginary=False,
                                 var_factor=0):
    """
    get the loss functions of a graph for the concurrence:
        C( |Psi> ) = √( 2 * ( 1 - TR_M( <Psi|Psi> ) ) ) 
        where TR_M is partial trace (in subsystem M)
        and return is sum over all possible bi-partition

    Parameters
    ----------
    edge_list : list
        list of all edges 
    sys_dict : dict
        that stores essential information about the quantuum system (see hf.get_sysdict)

    Returns
    -------
    func : function as object
        loss function in concurrence as lambda object of current graph.
    lossstring : String
        loss function as string

    """

    cat = graph.state_catalog
    target = th.entanglement_fast(cat, sys_dict, var_factor)
    # norm = th.Norm.fromDictionary(cat, real=sys_dict['real'])
    variables = th.stringEdges(graph.edges, imaginary=imaginary)

    lambdaloss = "".join(["", target])
    func, lossstring = th.buildLossString(lambdaloss, variables)

    return func


def sum_of_weights(inputgraph, cnfg):
    # test function for an arbitrary loss function (cnfg["loss_func"] = "lff", cnfg["lff_name"] = "sum_of_weights")
    return sum(np.abs(inputgraph.weights))


def loss_from_function(graph, cnfg=[]):
    # takes a graph and returns a function of a parameter vector
    def func(vector):
        inputgraph = graph
        for ii, edge in enumerate(graph.edges):
            inputgraph[edge] = vector[ii]
        # get function (defined in this module)
        function = globals()[cnfg["lff_name"]]
        return function(inputgraph, cnfg)

    return func


loss_dic = {'ent': [make_lossString_entanglement],
            'fid': [fidelity, count_rate],
            'cr': [count_rate, fidelity],
            'fidold': [state_fidelity, state_countrate],
            'crold': [state_countrate, state_fidelity],
            'gcr': [gate_countrate, gate_fidelity],
            'gfid': [gate_fidelity, gate_countrate],
            'gcrse': [gate_countrate_se, gate_fidelity_se],
            'gfidse': [gate_fidelity_se, gate_countrate_se],
            'hcr': [heralded_countrate, heralded_fidelity],
            'hfid': [heralded_fidelity, heralded_countrate],
            'hcrse': [heralded_countrate_se, heralded_fidelity_se],
            'hfidse': [heralded_fidelity_se, heralded_countrate_se],
            'lff': [loss_from_function]}
