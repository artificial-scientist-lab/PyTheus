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

import logging
log = logging.getLogger(__name__)

def state_countrate(graph, target_state, imaginary=False):
    target = target_state.targetEquation(state_catalog=graph.state_catalog, imaginary=imaginary)
    variables = th.stringEdges(graph.edges, imaginary=imaginary)
    graph.getNorm()
    lambdaloss = "".join(["1-", target, "/(1+", graph.norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func


def state_fidelity(graph, target_state, imaginary=False):
    target = target_state.targetEquation(state_catalog=graph.state_catalog, imaginary=imaginary)
    variables = th.stringEdges(graph.edges, imaginary=imaginary)
    graph.getNorm()
    lambdaloss = "".join(["1-", target, "/(0+", graph.norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func


def gate_countrate(graph, target_state, imaginary=False, out_nodes=None):
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


def gate_fidelity(graph, target_state, imaginary=False, out_nodes=None):
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
    target = target_state.targetEquation(state_catalog=graph.state_catalog, imaginary=imaginary)
    variables = th.stringEdges(graph.edges, imaginary=imaginary)

    verts = np.unique(list(itertools.chain(*th.edgeBleach(graph.edges).keys())))
    nonoutput_verts = [ii for ii in verts if ii not in out_nodes]
    all_edgecovers = []
    orders = (len(verts) - len(nonoutput_verts)) // 2 + 1
    for ii in range(orders):
        log.info('starting EC')
        all_edgecovers  += th.findEdgeCovers(graph.edges, nodes_left=nonoutput_verts, edges_left=len(verts) / 2 - ii)
        log.info(len(all_edgecovers))
    cat = th.stateCatalog(all_edgecovers)
    norm = th.writeNorm(cat, imaginary=imaginary)
    lambdaloss = "".join(["1-", target, "/(1+", norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func


def heralded_fidelity(graph, target_state, imaginary=False, out_nodes=None):
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


def make_lossString_entanglement(graph, sys_dict: dict, imaginary=False,
                                 var_factor = 0):
    """
    get the loss funcitons of a graph for the concuurence:
        C( |Psi> ) = √( 2 * ( 1 - TR_M( <Psi|Psi> ) ) ) 
        where TR_M is partial trace (in subsystem M)
        and return is sum over all possible bipartion

    Parameters
    ----------
    edge_list : list
        list of all edges 
    sys_dict : dict
        that stores essential information about the quantuum system (see hf.get_sysdict)

    Returns
    -------
    func : funciton as object
        loss function in conncurrence as lambda object of current graph.
    lossstring : String
        loss funciton as string

    """

    cat = graph.state_catalog
    target = th.entanglement_fast(cat, sys_dict,var_factor)
    # norm = th.Norm.fromDictionary(cat, real=sys_dict['real'])
    variables = th.stringEdges(graph.edges, imaginary=imaginary)

    lambdaloss = "".join(["", target])
    func, lossstring = th.buildLossString(lambdaloss, variables)

    return func


loss_dic = {'ent': [make_lossString_entanglement],
            'fid': [state_fidelity, state_countrate],
            'cr': [state_countrate, state_fidelity],
            'gcr': [gate_countrate, gate_fidelity],
            'gfid': [gate_fidelity, gate_countrate],
            'gcrse': [gate_countrate_se, gate_fidelity_se],
            'gfidse': [gate_fidelity_se, gate_countrate_se],
            'hcr': [heralded_countrate, heralded_fidelity],
            'hfid': [heralded_fidelity, heralded_countrate]}
