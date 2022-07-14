# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:54:12 2022

@author: janpe
"""

import theseus as th
import numpy as np
import config as confi


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




def compute_entanglement(qstate: np.array, sys_dict: dict)-> float:
    """
    calculate for a set of bipartions given in config the mean of
    trace[ rho_A ], where rho_A is reduced density matrix of given state
    for the given bipartitions

    Parameters
    ----------
    qstate : np.array
        basis vector of corrosponding state as np.array
    sys_dict : dict
        that stores essential_infos (see help_functions)

    Returns
    -------
    float
        return sum_bipar[ trace( rho_bi ** 2)  ]

    """
    dimi = np.array(sys_dict['dimensions'])
    try: # for checking if norm is not zero -> if so return 2 cause no ket
        qstate *= 1/(np.linalg.norm(qstate))
    except TypeError:
        return 2
        
    def calc_con(mat, par):
        red = th.ptrace(mat, par, dimi, False)
        return np.einsum('ij,ji', red, red) # is equivalent to trace( red**2 ) but faster

    loss_vec = [ calc_con(qstate, par[0]) for par in sys_dict['bipar_for_opti'] ]
    lenght = len(loss_vec)
    mean = sum( loss_vec )/lenght
    if confi.var_factor == 0: # no need to compute variance if factor = 0
        return mean
    else:
        var = sum([ (x-mean)**2 for x in loss_vec])/(lenght)
        return mean + confi.var_factor  * var  

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










