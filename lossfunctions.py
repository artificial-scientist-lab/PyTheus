# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:54:12 2022

@author: janpe
"""

import theseus as th
import numpy as np


def state_countrate(state, graph, real=True, coefficients=None):
    if len(coefficients) == 0:
        coefficients = [1] * len(state)
    target = th.targetEquation(
        state, avail_states=graph.state_catalog, coefficients=coefficients, real=real)
    if real:
        variables = ["w_{}_{}_{}_{}".format(*edge) for edge in graph.edges]
    else:
        variables = ["r_{}_{}_{}_{}".format(
            *edge) for edge in graph.edges] + ["th_{}_{}_{}_{}".format(*edge) for edge in graph.edges]
    graph.getNorm()
    lambdaloss = "".join(["1-", target, "/(1+", graph.norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func


def state_fidelity(state, graph, real=True, coefficients=None):
    if len(coefficients) == 0:
        coefficients = [1]*len(state)
    target = th.targetEquation(
        state, avail_states=graph.state_catalog, coefficients=coefficients, real=real)
    if real:
        variables = ["w_{}_{}_{}_{}".format(*edge) for edge in graph.edges]
    else:
        variables = ["r_{}_{}_{}_{}".format(
            *edge) for edge in graph.edges]+["th_{}_{}_{}_{}".format(*edge) for edge in graph.edges]
    graph.getNorm()
    lambdaloss = "".join(["1-", target, "/(0+", graph.norm, ")"])
    func, lossstring = th.buildLossString(lambdaloss, variables)
    return func





def make_lossString_entanglement(graph, sys_dict: dict, real = True):
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
    target = th.entanglement_fast(cat, sys_dict)
    #norm = th.Norm.fromDictionary(cat, real=sys_dict['real'])
    if real:
        variables = ["w_{}_{}_{}_{}".format(*edge) for edge in graph.edges]
    else:
        variables = ["r_{}_{}_{}_{}".format(
            *edge) for edge in graph.edges]+["th_{}_{}_{}_{}".format(*edge) for edge in graph.edges]
        
    
    lambdaloss = "".join(["", target])

    func, lossstring = th.buildLossString(lambdaloss, variables)

    return func



loss_dic = {'ent': [make_lossString_entanglement],
            'fid': [state_fidelity,state_countrate],
            'cr': [state_countrate,state_fidelity]}










