# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:54:12 2022

@author: janpe
"""

import pytheus.theseus as th
from collections import Counter
import itertools
import numpy as np
from itertools import combinations, product
from itertools import combinations_with_replacement
from pytheus.custom_loss.assembly_index import assembly_index, top_n_assembly
import logging

log = logging.getLogger(__name__)


def brutal_covers_heralding_sps(cnfg, graph):
    # this function finds heralding edgecovers for a bipartite graph
    in_verts = cnfg["single_emitters"]
    out_verts = cnfg["out_nodes"] + cnfg["anc_detectors"]

    avail_colors = th.edgeBleach(graph.edges)

    sep_edges = []
    for node in in_verts:
        group = [edge for edge in list(avail_colors.keys()) if node in edge[:2]]
        sep_edges.append(group)
    tmp_raw_covers = list(product(*sep_edges))

    raw_covers = []
    for cover in tmp_raw_covers:
        counter = Counter(list(itertools.chain(*cover)))
        keep = True

        if cnfg['number_resolving']:
            for vert in cnfg["anc_detectors"]:
                if counter[vert] != 1:
                    keep = False
        else:
            for vert in cnfg["anc_detectors"]:
                if counter[vert] == 0:
                    keep = False
        # if all conditions are met, use edgecover for norm
        if keep:
            raw_covers.append(cover)

    painted_covers = []
    for cover in raw_covers:
        for coloring in itertools.product(*[avail_colors[edge] for edge in cover]):
            color_cover = [edge + color for edge, color in zip(cover, coloring)]
            painted_covers.append(sorted(color_cover))
    return [[tuple(ed) for ed in graph] for graph in painted_covers]


def brutal_covers(cnfg, graph):
    # this function is sometimes used instead of findEdgecovers, should be much slower in general, but faster at
    # applying topological constraints atm

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

        # # if in maximum order, we select for events where the correct total number of photons go into output paths
        if cnfg['novac']:
            if len(cover) == max_edges:
                sum = 0
                for vert in cnfg["out_nodes"]:
                    sum += counter[vert]
                if sum != len(cnfg["out_nodes"]):
                    keep = False
            else:
                keep = False

        # if all conditions are met, use edgecover for norm
        if keep:
            raw_covers.append(cover)
    painted_covers = []
    for cover in raw_covers:
        for coloring in itertools.product(*[avail_colors[edge] for edge in cover]):
            color_cover = [edge + color for edge, color in zip(cover, coloring)]
            painted_covers.append(sorted(color_cover))
    final_list = []
    for order in orders:
        painted_covers_order = [graph for graph in painted_covers if len(graph) == order]
        final_list += [tuple([tuple(ed) for ed in gg]) for gg in np.unique(painted_covers_order, axis=0)]
    return final_list


def heralded_covers(cnfg, graph):
    # calls findEdgecovers and applies topological constraints

    non_output_verts = [vert for vert in cnfg["verts"] if vert not in cnfg["out_nodes"]]
    # minimum number of edges that can cover non output vertices
    min_edges = len(non_output_verts) // 2
    # maximum number of edges, perfect matching order of whole graph
    max_edges = len(cnfg["verts"]) // 2
    orders = list(range(min_edges, max_edges + 1))
    try:
        if cnfg["novac"]:
            orders = [max_edges]
    except KeyError:
        pass
    # find edge suitable edge covers for all possible numbers of edges
    tmp_edgecovers = []
    for num_edges in orders:
        tmp_edgecovers += th.findEdgeCovers(graph.edges, nodes_left=non_output_verts, edges_left=num_edges)
    # select for edgecovers that fulfill conditions
    edgecovers = []
    for cover in tmp_edgecovers:
        keep = True
        bleached_cover = [e[:2] for e in cover]
        concat_cover = list(itertools.chain(*bleached_cover))
        counter = Counter(concat_cover)
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
        if cnfg['novac']:
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
    # can be used for state-creation/measurements/gates, post-selected/heralded

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
        elif cnfg["bipartite"]:
            edgecovers = brutal_covers_heralding_sps(cnfg, graph)
        else:
            edgecovers = brutal_covers(cnfg, graph)
        cat = th.stateCatalog(edgecovers)
        norm = th.writeNorm(cat, imaginary=cnfg["imaginary"])
    lambdaloss = "".join(["1-", target, "/(1+", norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    print('count rate done', flush=True)
    return func


def fidelity(graph, target_state, cnfg):
    # can be used for state-creation/measurements/gates, post-selected/heralded

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
        elif cnfg["bipartite"]:
            edgecovers = brutal_covers_heralding_sps(cnfg, graph)
        else:
            edgecovers = brutal_covers(cnfg, graph)
        cat = th.stateCatalog(edgecovers)
        norm = th.writeNorm(cat, imaginary=cnfg["imaginary"])
    lambdaloss = "".join(["1-", target, "/(0+", norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    print('fidelity done', flush=True)
    return func


"""
Number states in Fock basis
"""


def fock_fidelity(graph, target_state, num_anc, amplitudes, imaginary=False):  # num_anc_pre

    # original ket is in the form: [((0,m),(1,n),(2,k1),(3,k2)...]; 
    # here the m, n, k1, k2 are the number of particles instead of dimensions

    # make the ket in the correct form 
    # [((0,0),(0,0)...,(1,0),(1,0),... (2,0),(2,0),...,(3,0),(3,0),...]
    # m times (0,0); n times (1,0); k1 times (2,0); k2 times (3,0), etc.
    # target_kets_temp = target_state.kets 
    # target_kets=[]
    # accum = [sum([[(i,0)]*max(0,n) for i,n in k ],[]) for k in target_kets_temp]
    # for ii in accum:
    #     target_kets.append(tuple(ii))   

    # all_verts= range(graph.num_nodes)#np.unique(list(itertools.chain(*th.edgeBleach(graph.edges).keys())))
    # anc_each_num=[int(i) for i in num_anc_pre]
    # state_each_num=[int(i) for i in target_state_str[0]] # all terms have the same photon number

    total_particle_num = len(target_state.kets[0])
    # total_particle_num=sum(state_each_num)+num_anc #sum(anc_each_num)

    # anc_position=all_verts[len(all_verts)-num_anc:]
    # anc_kets=[]
    # for jj in anc_position:
    #     anc_kets.append((jj,0))

    anc_nodes = list(range(graph.num_nodes - num_anc, graph.num_nodes))

    edgecover_target = list(itertools.combinations_with_replacement(graph.edges, int(total_particle_num / 2)))
    cat = th.stateCatalog(edgecover_target)

    if len(anc_nodes) > 0:
        for ket in list(cat.keys()):
            shopping = Counter(ket)
            if all(shopping[(ii, 0)] == 1 for ii in anc_nodes):
                pass
            else:
                del cat[ket]

    # this cause a problem in the optimizer as it actually does not use len(useful_deges)
    # useful_edges = sorted(set(sum(sum(cat.values(),[]), ())))
    # variables = th.stringEdges(useful_edges)

    variables = th.stringEdges(graph.edges, imaginary=imaginary)

    target = th.targetEquation(target_state.kets, amplitudes=amplitudes, state_catalog=cat)
    norm = th.writeNorm(cat)

    # epsilon = 1e-10
    epsilon = 0
    lambdaloss = f'1-({target})/({epsilon} + {norm})'
    #  lambdaloss="".join(["1-", target, "/(0+", norm, ")"])

    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func


def fock_countrate(graph, target_state, num_anc, amplitudes, imaginary=False):  # num_anc_pre

    total_particle_num = len(target_state.kets[0])

    anc_nodes = list(range(graph.num_nodes - num_anc, graph.num_nodes))

    edgecover_target = list(itertools.combinations_with_replacement(graph.edges, int(total_particle_num / 2)))
    cat = th.stateCatalog(edgecover_target)

    if len(anc_nodes) > 0:
        for ket in list(cat.keys()):
            shopping = Counter(ket)
            if all(shopping[(ii, 0)] == 1 for ii in anc_nodes):
                pass
            else:
                del cat[ket]

    # this cause a problem in the optimizer as it actually does not use len(useful_deges)
    # useful_edges = sorted(set(sum(sum(cat.values(),[]), ())))
    # variables = th.stringEdges(useful_edges)

    variables = th.stringEdges(graph.edges, imaginary=imaginary)

    target = th.targetEquation(target_state.kets, amplitudes=amplitudes, state_catalog=cat)
    norm = th.writeNorm(cat)

    # epsilon = 1e-10
    epsilon = 1
    lambdaloss = f'1-({target})/({epsilon} + {norm})'
    #  lambdaloss="".join(["1-", target, "/(1+", norm, ")"])

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


# optimizer optimizes first function in list
# at each optimization step, all functions in list are checked if they satisfy the thresholds
loss_dic = {'ent': [make_lossString_entanglement],
            'fid': [fidelity, count_rate],
            'cr': [count_rate, fidelity],
            'lff': [loss_from_function],
            'fockfid': [fock_fidelity, fock_countrate],
            'fockcr': [fock_countrate, fock_fidelity]
            }
