# +
import theseus as th

import time
import itertools
from math import factorial, ceil
from collections import Counter
import random

import numpy as np
from scipy.optimize import fsolve
import scipy.optimize as optimize
from sympy import symbols, sqrt, simplify
from sympy.solvers import solve
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
import json


# +
def NEWedgeWeight(edge): 
    return 'w_{}_{}_{}_{}'.format(*edge)

def NEWweightProduct(graph):
    return '*'.join([NEWedgeWeight(edge) for edge in graph])

def NORMfromDictionary(states_dict):
    '''
    Build a normalization constant with all the states of a dictionary.
    '''
    norm_sum = []
    for key, values in states_dict.items():
        term_sum = [f'{NEWweightProduct(graph)}/{th.factProduct(graph)}'
                    for graph in values]
        term_sum = '+'.join(term_sum)
        norm_sum.append(f'{th.factProduct(key)}*(({term_sum})**2)') 
    return '+'.join(norm_sum)

def NEWtargetEquation(coefficients, states, avail_states=None):
    '''
    Introducing the coefficients for each ket of the states list, it builds a 
    non-normalized fidelity function with all the ways the state can be build 
    according to the dictionary avail_states. If no dictionary is introduced 
    it builds all possible graphs that generate the desired states.
    '''
    if len(coefficients)!=len(states):
        raise ValueError('The number of coefficients and states should be the same')
    norm2 = abs(np.conjugate(coefficients)@coefficients)
    if norm2 != 1: norm2 = str(norm2)
    if avail_states == None: 
        avail_states = {tuple(st):th.allColorGraphs(st) for st in states}
    equation_sum = []
    for coef, st in zip(coefficients,states):
        term_sum = [NEWweightProduct(graph) for graph in avail_states[tuple(st)]]
        term_sum = '+'.join(term_sum)
        equation_sum.append(f'({coef})*({term_sum})')
    equation_sum = '+'.join(equation_sum)
    return f'(({equation_sum})**2)/{norm2}'

def NEWbuildAllEdges(dimensions, loops=False):
    '''
    Given a collection of nodes, each with several possible colors/dimensions, 
    returns all possible edges of the graph.
    
    Parameters
    ----------
    dimensions : array_like
        Accesible dimensions (colors) for each of the nodes of the graph.
    loops : boolean, optional
        Allow edges to connect twice the same node.
    symbolic : boolean, optional
        If True, it returns a list of sympy symbols instead of tuples. 
    padding : boolean, optional
        If symbolic is True, padding determines whether the nodes numbers are
        padded with one zero or not: 0X if True, X if false. 
        
    Returns
    -------
    all_edges : list
        List of all possible edges given the dimensions of the nodes.
        If symbolic, it returns a list of sympy symbols.
        Else, it returns a list of tuples: (node1, node2, color1, color2).
    '''
    num_vertices = len(dimensions)
    all_edges = []
    if loops:
        combo_function = itertools.combinations_with_replacement
        for pair in combo_function(range(num_vertices),2):
            if pair[0]==pair[1]: # (node1, node1, 1, 0) is not stored
                for dims in combo_function(range(dimensions[pair[0]]),2):
                    all_edges.append((pair[0],pair[1],dims[0],dims[1]))
            else:
                for dims in itertools.product(*[range(dimensions[ii]) for ii in pair]):
                    all_edges.append((pair[0],pair[1],dims[0],dims[1]))
    else:
        for pair in itertools.combinations(range(num_vertices),2):
            for dims in itertools.product(*[range(dimensions[ii]) for ii in pair]):
                all_edges.append((pair[0],pair[1],dims[0],dims[1]))
    # returns edges whether as tuples or as sympy symbols
    return [NEWedgeWeight(edge) for edge in all_edges]


# -

pdv424 = [2,2,2,2] # Dimensions
graphs424 = th.allEdgeCovers(pdv424, order=0, loops=False) # Dictionary with states and the corresponding contributions
states424 = [((0,0),(1,0),(2,0),(3,0)),  # Ket basis of the state we want to obtain 
             ((0,1),(1,1),(2,1),(3,1))]  
target424 = NEWtargetEquation([1,-1],states424,avail_states=graphs424) # String expression with the non-normalized fidelity
norm424till0 = NORMfromDictionary(graphs424).replace('/1+','+') # Norm string (the replace is not needed)
loss = f' - ({target424}) / (1+{norm424till0})'

all_edges = NEWbuildAllEdges(pdv424, loops=False) # List with all the variables, the weights
lambda_loss = 'lambda ' + ', '.join(all_edges) + f': {loss}' # Filthy trick to define function with arbitrary variables
lambda_loss = f'funMario = lambda inputs: ({lambda_loss})(*inputs) ' # Roundabout needed for scipy optimizer
exec(lambda_loss) 

initial_values = 2*np.random.random(len(all_edges)) - 1
bounds_start = len(initial_values)*[(-1,1)]
result = optimize.minimize(funMario,x0 = initial_values, bounds = bounds_start,method ='L-BFGS-B') 

solution424 = {key:round(value,4) for key, value in zip(all_edges,result.x)} # readable solutions 
