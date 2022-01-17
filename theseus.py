#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 07:02:11 2021
@author: alejomonbar, cruizgo and soerenarlt
"""
import itertools
from math import factorial
from collections import Counter
import random
import numpy as np
import scipy.optimize as optimize
from sympy import symbols, sqrt
from sympy.solvers import solve
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
import json

        
###############################
###############################
###                         ###
###   AUXILIARY FUNCTIONS   ###
###                         ###
###############################
###############################


def allPairSplits(lst):
    """ 
    Generate all sets of unique pairs from a list `lst`.
    This is equivalent to all _partitions_ of `lst` (considered as an indexed 
    set) which have 2 elements in each partition.
    Recall how we compute the total number of such partitions. Starting with 
    a list [1, 2, 3, 4, 5, 6]
    
    one takes off the first element, and chooses its pair [from any of the 
    remaining 5]. For example, we might choose our first pair to be (1, 4). 
    Then, we take off the next element, 2, and choose which element it is 
    paired to (say, 3). So, there are 5 * 3 * 1 = 15 such partitions.
    That sounds like a lot of nested loops (i.e. recursion), because 1 could 
    pick 2, in which case our next element is 3. But, if one abstracts "what 
    the next element is", and instead just thinks of what index it is in the 
    remaining list, our choices are static and can be aided by the 
    itertools.product() function.
    
    From selfgatoatigrado: https://stackoverflow.com/a/13020502
    """
    N = len(lst)
    choice_indices = itertools.product(*[range(k) for k in reversed(range(1,N,2))])

    for choice in choice_indices:
        # calculate the list corresponding to the choices
        tmp = lst[:]
        result = []
        for index in choice:
            result.append( (tmp.pop(0), tmp.pop(index)) )
        yield result # use yield and then turn it into a list is faster than append

        
def targetEdges(nodes, graph):
    '''
    Returns all graph's edges which connect to the input nodes.
    '''    
    return [ed for ed in graph if (ed[0] in nodes)or(ed[1] in nodes)]


def removeNodes(nodes, graph):
    '''
    Removes all graph's edges which connect to the input nodes.
    '''
    return [ed for ed in graph if not ((ed[0] in nodes)or(ed[1] in nodes))]


def deadEndEdges(graph):
    '''
    Returns all edges connecting nodes with degree one.
    '''
    unq, counts = np.unique(np.array(graph)[:,:2], return_counts=True)
    return targetEdges(unq[counts==1],graph)
# Careful when applying this in subgraphs, the dead ends may be covered already


def edgeBleach(color_edges): # this may end up being useless
    '''
    Takes list of color edges and return dictionary with uncolored ones
    and all possible colors for each.
    '''
    raw_edges = np.unique(np.array(color_edges)[:,:2],axis=0)
    bleached_edges = {tuple(edge):[] for edge in raw_edges}
    for edge in color_edges:
        bleached_edges[edge[:2]].append(edge[2:])
    return bleached_edges


def nodeDegrees(edge_list, nodes_list=[], rising=True):
    '''
    Compute the degree of each node of a graph. Returning a list of tuples
    such that: [(node1, degree1), (node2, degree2) ...]
    By default, it sorts the nodes by their degree in increasing order.
    
    Parameters
    ----------
    edge_list : list
        List of all available edges.
    nodes_list : list, optional
        List of all nodes. By default, the list is obtained from the nodes 
        that appear in edge_list.
    rising : boolean, optional
        If True, the nodes are ordered by degree.
    Returns
    -------
    links : list
        List of nodes and their degrees: [(node1, degree1), ...]
    '''
    if len(nodes_list)==0: nodes_list = np.unique(np.array(edge_list)[:,:2])
    links = {ii:0 for ii in nodes_list}
    for edge in edge_list:
        links[edge[0]] += 1
        links[edge[1]] += 1
    if rising: return sorted(links.items(), key=lambda item: item[1])
    else: return [(k,v) for k,v in zip(links.keys(),links.values())]


def stateCatalog(graph_list):
    '''
    Given a list of colored graphs, returns a dictionary with graphs grouped 
    by the state they generate. Each of these states is a dictionary key.
    
    Parameters
    ----------
    graph_list : list
        List of graphs.
    Returns
    -------
    state_dict : dictionary
        Dictionary with all the states created by each graph as keys. If more 
        than one graph leads to the same state their are listed together in 
        the corresponding entrance of the dictionary.
    '''
    state_dict = {}
    for graph in graph_list:
        coloring = []
        for ed in graph: coloring += [(ed[0], ed[2]), (ed[1], ed[3])]
        coloring = tuple(sorted(coloring))
        try: state_dict[coloring] += [graph]
        except KeyError: state_dict[coloring] = [graph]
    return state_dict

        
############################
############################
###                      ###
###   GRAPH GENERATORS   ###
###                      ###
############################
############################


def buildAllEdges(dimensions, loops=False, symbolic=False, padding=False):
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
    if loops: combination_function = itertools.combinations_with_replacement
    else: combination_function = itertools.combinations
    for pair in combination_function(range(num_vertices),2):
        for dims in itertools.product(*[range(dimensions[ii]) for ii in pair]):
            all_edges.append((pair[0],pair[1],dims[0],dims[1]))
    # returns edges whether as tuples or as sympy symbols
    if symbolic: return [edgeWeight(edge,padding) for edge in all_edges]
    else: return all_edges


def buildRandomGraph(dimensions, num_edges, loops=False, cover_all=True):
    '''
    Given a set of nodes with different dimensions it build a random graph
    with a given number of edges that connects all nodes.
    '''
    all_edges = buildAllEdges(dimensions,loops=loops)
    # even when only one dimension is available we put it on the symbols
    # if sorted(dimensions)[-1]==1:
    #     all_edges = [edge[:2] for edge in all_edges]
    if cover_all:
        num_nodes = len(dimensions)
        if 2*num_edges > num_nodes:
            all_covered = False
        else: raise ValueError('num_edges is too low to cover all nodes')
        while not all_covered:
            graph = random.sample(all_edges,num_edges)
            all_covered = (len(np.unique(np.array(graph)[:,:2])) == num_nodes)
        return sorted(graph)
    else: return sorted(random.sample(all_edges,num_edges))


def allColorGraphs(color_nodes,loops=False):
    '''
    Given a list of colored nodes, i.e., an state. It uses AllPairSplits to 
    generate all graphs that leads to such state.
    This function is a building block of the function 'allEdgeCovers'.
    
    Parameters
    ----------
    color_nodes : list
        List of all colored nodes: [(node,color)...] Some may be repeated.
    loops : boolean, optional
        Allow edges to connect twice the same node, not necessarily with a
        single color: (node1, node1, color1, color2).
        
    Returns
    -------
    graph_list : list of tuples
        Nested list with all graphs that produce a given state.
        Example: given the nodes [(0,0),(0,1),(1,1),(1,1)] it produces:
            [[(0, 0, 0, 1), (1, 1, 1, 1)],    - only if loops are allowed -
             [(0, 1, 0, 1), (0, 1, 1, 1)]]
    '''
    color_graph = list(allPairSplits(sorted(list(color_nodes))))
    for graph in color_graph: graph.sort()
    # The following lines builds the edge (node1, node2, color1, color2).
    # After filtering loops we filter the corresponding graph configurations.
    # Loops represent colinear effects so we may recover them eventually.
    if loops:
        color_graph = [[[nd[0][0],nd[1][0],nd[0][1],nd[1][1]] for nd in graph] 
                         for graph in color_graph]
    else: 
        color_graph = [[[nd[0][0],nd[1][0],nd[0][1],nd[1][1]] for nd in graph 
                    if nd[0][0] != nd[1][0]] for graph in color_graph]
        color_graph = [graph for graph in color_graph
                       if len(graph)==len(color_nodes)/2]
    return [[tuple(ed) for ed in graph] for graph in np.unique(color_graph,axis=0)]


def allEdgeCovers(dimensions, order=1, loops=False):
    '''
    Given a collection of nodes with different dimensions/colors available, 
    it produces all possible states that erase from these and the different 
    graphs that can produce such states.
    The graphs nodes can present different degres, the edges may be duplicated
    (multigraph) and loops may also appear.
    
    Parameters
    ----------
    dimensions : array_like
        Accesible dimensions (colors) for each of the nodes of the graph.
    order : int, optional
        Orders above the minimum required to cover all nodes. If 0, the 
        output are only perfect machings, with all nodes having degree 1. 
    loops : boolean, optional
        Allow edges to connect twice the same node. These cannot appear for
        perfect matchings, thus only available for order>0.
        
    Returns
    -------
    color_dict : dictionary
        Dictionary of available states. The keys are the different combinations 
        of colored nodes (creator operators), that is, the state produced. 
        For each or state, the dictionary stores a list of all possible graphs
        that produce such state.
        The notation employed for the nodes is: (node, color/dimension).
        The notation employed for the edges is: (node1, node2, color1, color2).
        Node2 cannot be lower than Node1.
    '''
    num_nodes = len(dimensions)
    # Given a list with N different nodes, if order>0 some nodes will have degree
    # larger than 1, i.e., their creator operators will be repeated.
    # crowded_graph stores all ways in which these repeatitions may occur.
    # For N=4 and order=1: 000123, 001123, 001223, 001233, 011123, 011223...
    additions = list(itertools.combinations_with_replacement(range(num_nodes),2*order))
    crowded_graph = [list(range(num_nodes))] * len(additions)
    for ii, added in enumerate(additions):
        crowded_graph[ii] = sorted(crowded_graph[ii] + list(added))    
    color_dict = {} 
    for crowd in crowded_graph:
        # For each list of nodes in crowded_graph (with possible repetitions) 
        # we store all dimensions/coloring the nodes can have in color_nodes.
        color_nodes = []
        # We distinguish between nodes but not between repeated nodes.
        # [(node1,color1),(node1,color2)] = [(node1,color2),(node1,color1)]
        for coloring in itertools.product(*[list(range(dimensions[nn])) for nn in crowd]):
            color_nodes.append(sorted([[crowd[ii], coloring[ii]] for ii in range(len(crowd))]))
        # After sorting the list of colored nodes, the next one erase duplicities.
        color_nodes = [[tuple(ed) for ed in graph] for graph in np.unique(color_nodes,axis=0)]
        for coloring in color_nodes: 
            color_dict[tuple(coloring)] = allColorGraphs(coloring,loops)
    return color_dict 


def recursivePerfectMatchings(graph, store, matches=[], edges_left=None):
    '''
    The heavy lifting of findPerfectMatchings. Also used in recursiveEdgeCover.
    '''
    if edges_left==None: edges_left = len(np.unique(np.array(graph)[:,:2]))/2
    if len(graph)>0:
        for edge in targetEdges([nodeDegrees(graph)[0][0]],graph):
            recursivePerfectMatchings(removeNodes(edge[:2], graph), store, 
                                      matches+[edge], edges_left-1)
    elif len(graph)==0 and edges_left==0: store.append(sorted(matches))
    else: pass # Some nodes are not matched and never will


def findPerfectMatchings(graph):
    '''
    Returns all possible perfect matchings (if any) given a list of edges.
    '''
    perfect_matchings = []
    recursivePerfectMatchings(graph, perfect_matchings)
    return perfect_matchings


def recursiveEdgeCover(graph, store, matches=[], edges_left=None, nodes_left=[], order=1, loops=False):
    '''
    Pseudorecursive function that append all possible edge covers of a 
    graph in a given list 'store'. 
    It does the heavy lifting of findEdgeCovers.
    
    Parameters
    ----------
    graph : list
        List of all available edges.
    store : list
        List to store the edge covers.
    matches : list, optional
        List of edges to build a perfect matching. If it fulfils all
        requirements it will end up in store
    edges_left : int, optional
        Total number of edges of each edge cover. If None, this is computed 
        as the minimum edges required to cover all nodes plus the order.
    nodes_left : list, optional
        List of all nodes to be covered. By default, the list is obtained 
        from the nodes covered in edge_list but it can be set to target
        specific nodes.
    order : int, optional
        Orders above the minimum required to cover all nodes. If 0, the 
        output are only perfect machings, with all nodes having degree 1. 
    loops : boolean, optional
        Allow edges to connect twice the same node. These cannot appear for
        perfect matchings. 
        
    Returns
    -------
    None, but now the input list 'store' contains all possible edge covers. 
    '''
    if len(nodes_left)==0: nodes_left = np.unique(np.array(graph)[:,:2]) 
    if edges_left==None: edges_left = order + len(nodes_left)/2
    case = 2*edges_left - len(nodes_left)
    if case>1:
        for edge in graph: 
            recursiveEdgeCover(graph, store, matches+[edge], edges_left-1, 
                [node for node in nodes_left if node not in edge[:2]], loops=loops) 
    elif case==1: 
        for edge in targetEdges(nodes_left, graph):
            if edges_left>1:
                new_nodes_left = [node for node in nodes_left if node not in edge[:2]]
                recursiveEdgeCover(targetEdges(new_nodes_left,graph), store, 
                        matches+[edge], edges_left-1, new_nodes_left, loops=loops)
            if edges_left==1 and not (sorted(matches + [edge]) in store):
                store.append(sorted(matches + [edge]))
    elif case==0:
        subgraph = [ed for ed in graph if (ed[0] in nodes_left)&(ed[1] in nodes_left)]
        if loops: subgraph = [edge for edge in subgraph if edge[0]!=edge[1]]
        perfect_matchings = []
        recursivePerfectMatchings(subgraph, perfect_matchings, edges_left=edges_left)
        for pm in perfect_matchings: 
            if not (sorted(matches + pm) in store):
                store.append(sorted(matches + pm))
    else: pass
# for large orders we could compute a first pack of edges at once using itertools 


def findEdgeCovers(edge_list, edges_left=None, nodes_left=[], order=1, loops=False):#, quick_start=False):
    '''
    Given a list of edges, returns all possible edge covers, up to a certain
    order or using a certain number of edges. 
    
    Parameters
    ----------
    edge_list : list
        List of all available edges.
    edges_left : int, optional
        Total number of edges of each edge cover. If None, this is computed 
        as the minimum edges required to cover all nodes plus the order.
    nodes_left : list, optional
        List of all nodes to be covered. By default, the list is obtained 
        from the nodes covered in edge_list but it can be set to target
        specific nodes.
    order : int, optional
        Orders above the minimum required to cover all nodes. If 0, the 
        output are only perfect machings, with all nodes having degree 1. 
    loops : boolean, optional
        Allow edges to connect twice the same node. These cannot appear for
        perfect matchings.
        
    Returns
    -------
    covers : list
        List of graphs with all nodes having, at least, degree 1.
    '''
    covers = []
    recursiveEdgeCover(edge_list, covers, edges_left=edges_left, 
                       nodes_left=nodes_left, order=order, loops=loops)
    # IGNORE THE NEXT LINES FOR NOW, THEY DON'T SEEM TO HELP
    #quick_start: boolean, optional
    #    If True, it stores all edges connected to nodes with degree one. 
    # if quick_start: # this is just to speed up stuff (or at least try it)
    #     starting = deadEndEdges(edge_list)
    #     if len(starting)==0:
    #         print('There are no nodes with degree 1.')
    #         recursiveEdgeCover(edge_list, covers, edges_left=edges_left, 
    #                            nodes_left=nodes_left, order=order)
    #     else:
    #         print(starting)
    #         if len(nodes_left)==0: 
    #             nodes_left = np.unique(np.array(edge_list)[:,:2])
    #         edges_left = order + len(nodes_left)/2 - len(starting)
    #         nodes_left = [node for node in nodes_left if node not 
    #                       in np.unique(np.array(starting)[:,:2])]
    #         recursiveEdgeCover(edge_list, covers, matches=starting, 
    #                            edges_left=edges_left, 
    #                            nodes_left=nodes_left, 
    #                            order=order, loops=loops)
    return covers

def findEdgeCoversColorLater(edge_list):
    '''Extension of findEdgeCovers (atm only for order=0). Better time complexity for graphs with high dimension.'''
    
    # get uncolored graph and find perfect matchings and store what colored edges are permitted
    edgedict = edgeBleach(edge_list)
    uc_edge_list = list(edgedict.keys())
    uc_pms = findEdgeCovers(uc_edge_list,order=0)

    pms = []
    #reintroduce color to uncolored pms (all combinations that are possible with edge_list)  
    for pm in uc_pms:
        edgemult = [range(len(edgedict[edge])) for edge in pm] #how many colored edges are there for each edge in the uncolored pm
        for coloring in itertools.product(*edgemult): #go through all combinations of colored edges that produce the same PM if colorblind
            color_pm = []
            for ii , edge in enumerate(pm):
                color = edgedict[edge][coloring[ii]] #read out the color
                color_pm.append(edge+color) #add colored edge to colored PM
            pms.append(color_pm)
    return pms

###############################
###############################
###                         ###
###   HERALDING FUNCTIONS   ###
###                         ###
###############################
###############################


def findTriggerEdgeCover(edge_list, triggers, order):
    '''
    Returns all edge covers of the trigger vertices given a list of edges, up to a certain order.
    '''
    covers = []
    nodes_left = triggers 
    edges_left = order + len(nodes_left)//2
    recursiveEdgeCover(edge_list, covers, edges_left= edges_left,nodes_left= nodes_left)
    return covers


def findTriggerState(edge_list,num_nodes,num_trigger):
    '''
    Returns sympy expression of the state heralded by measuring the trigger nodes.
    Convention: num_nodes is the total number of nodes, the trigger nodes are the last num_trigger nodes
    '''
    triggers = list(range(num_nodes-num_trigger,num_nodes))

    #PREPARING TRIGGERED STATE 
    #define the range of orders we are interested in (for heralded bell states it is orders 0 and 1)
    #order 0 is the smallest order that can still cover all trigger vertices
    #highestorder is the minimum order that can cover all vertices of the entire graph
    highestorder = (num_nodes-num_trigger)//2
    allcovers = []
    state = 0
    for order in range(highestorder+1):
        #find all edge covers of the trigger with specific order
        #covers = findTriggerEdgeCover(edge_list,triggers,order) # IF EVERYTHING ELSE FAILS, THIS WORKS, THE NEXT LINE SHOULD BE COMMENTED, AND CARLOS WAS TOO COCKY
        covers = findEdgeCovers(edge_list, nodes_left=triggers, order=order)
        #save covers of each order
        allcovers.append(covers)
        #turn covers into sympy expression with theseus library functions
        cat = stateCatalog(covers)
        state += State.fromDictionary(cat)

    #dictionary for substitution: edge symbol -> edge weight
    edge_dict = {}
    for edge in edge_list:
        edge_dict[symbols(f'w_{edge[0]:02d}\,{edge[1]:02d}^{edge[2]}\,{edge[3]}')]=edge[4]

    if state != 0:
        sympy_expression = state.subs(edge_dict)
    else:
        sympy_expression = 0
        
    return sympy_expression

        
################################
################################
###                          ###
###   SYMBOLIC EXPRESSIONS   ###
###                          ###
################################
################################


# This expression could also be UNlambdified so we add some check for the Counter
factProduct = lambda lst: np.product([factorial(ii) for ii in Counter(lst).values()])


def edgeWeight(edge, padding=False): 
    if padding: return symbols(f'w_{edge[0]:02d}\,{edge[1]:02d}^{edge[2]}\,{edge[3]}')
    else: return symbols(f'w_{edge[0]}\,{edge[1]}^{edge[2]}\,{edge[3]}')


def weightProduct(graph, padding=False):
    return np.product([edgeWeight(edge,padding) for edge in graph])


def creatorState(nodes, padding=False):
    if padding: return np.product([symbols(f'v_{nn[0]:02d}^{nn[1]}') for nn in nodes])
    else: return np.product([symbols(f'v_{nn[0]}^{nn[1]}') for nn in nodes])


def targetEquation(coefficients, states, avail_states=None, padding=False):
    '''
    Introducing the coefficients for each ket of the states list, it builds a 
    non-normalized fidelity function with all the ways the state can be build 
    according to the dictionary avail_states. If no dictionary is introduced 
    it builds all possible graphs that generate the desired states.
    '''
    if len(coefficients)!=len(states):
        raise ValueError('The number of coefficients and states should be the same')
    norm = np.conjugate(coefficients)@coefficients
    if norm != 1: coefficients = np.array(coefficients)/sqrt(norm)
    if avail_states == None: 
        avail_states = {tuple(st):allColorGraphs(st) for st in states}
    equation = 0
    for coef, st in zip(coefficients,states):
        terms = 0
        for graph in avail_states[tuple(st)]:
            terms += weightProduct(graph,padding)
        equation += coef*terms
    return equation**2


class Norm:
    '''
    Set of functions to compute the normalization constant.
    '''
    def fromDictionary(states_dict, padding=False):
        '''
        Build a normalization constant with all the states of a dictionary.
        '''
        norm = 0
        for key in states_dict.keys():
            term = 0
            for graph in states_dict[key]:
                term += weightProduct(graph,padding)/factProduct(graph)
            norm += factProduct(key)*(term**2)
        return norm
    
    def fromEdgeCovers(edge_list, max_order=0, min_order=0, loops=False, padding=False):
        '''
        Returns the normalization constant (up to an arbitrary order) of all states 
        that can be build with an edge cover of the available edges.
        '''
        norm = 0
        for order in range(min_order, max_order+1):
            norm += Norm.fromDictionary(stateCatalog(findEdgeCovers(edge_list,
                                                            order=order,loops=loops)),padding)
        return norm 
        
    def fromEdgeCoversColorLater(edge_list, max_order=0, min_order=0, loops=False, padding=True):
        '''
        Returns the normalization constant (up to an arbitrary order) of all states 
        that can be build with an edge cover of the available edges.
        '''
        norm = 0
        for order in range(min_order, max_order+1):
            norm += Norm.fromDictionary(stateCatalog(findEdgeCoversColorLater(edge_list)),padding)
        return norm
    
    def fromDimensions(dimensions, max_order=0, min_order=0, loops=False, padding=False):
        '''
        Given a list of dimensions for several particles, returns the normalization
        constant for all possible states involving all the particles at least once,
        up arbitrary order.
        '''
        norm = 0
        for order in range(min_order, max_order+1):
            norm += Norm.fromDictionary(allEdgeCovers(dimensions,order,loops),padding)
        return norm  

    
class State:
    '''
    Set of functions to compute the state.
    '''
    def fromDictionary(states_dict, q_factor=False, padding=False):
        '''
        Build a superposition of all the states of a dictionary.
        '''
        state = 0
        for key in states_dict.keys():
            term = 0
            for graph in states_dict[key]:
                term += weightProduct(graph,padding)/factProduct(graph)
            term *= creatorState(key,padding)
            if q_factor: term *= sqrt(factProduct(key))
            state += term
        return state
    
    def fromEdgeCovers(edge_list, max_order=0, min_order=0, loops=False, q_factor=False, padding=False):
        '''
        Returns the superposition (up to an arbitrary order) of all states that can be 
        build with an edge cover of the available edges.
        '''
        state = 0
        for order in range(min_order, max_order+1):
            state += State.fromDictionary(stateCatalog(findEdgeCovers(edge_list,order=order,loops=loops)),
                                          q_factor,padding)
        return state
    
    def fromDimensions(dimensions, max_order=0, min_order=0, loops=False, q_factor=False, padding=False):
        '''
        Given a list of dimensions for several particles, returns the superposition of all
        possible states involving all the particles at least once, up arbitrary order.
        '''
        state = 0
        for order in range(min_order, max_order+1):
            state += State.fromDictionary(allEdgeCovers(dimensions,order,loops),q_factor,padding)
        return state  

    
dumbFunction = lambda inputs, function: function(*inputs)
def sympyMinimizer(loss_function,variables=[],initial_values=[],bounds=None,method=None,options=None):
    '''
    Apply scipy minimizer on a sympy expression.
    '''
    if len(variables)==0: 
        variables = list(loss_function.free_symbols)
        variables.sort(key=lambda s: str(s))
    if len(initial_values)==0: initial_values = 2*np.random.random(len(variables)) - 1
    loss_scipy = lambdify(variables, loss_function, modules="numpy")
    return optimize.minimize(dumbFunction, x0=initial_values, bounds=bounds, method=method, options=options, args=(loss_scipy))

###################################
###################################
###                             ###
###   MAIN CLASS (FIRST CODE)   ###
###                             ###
###################################
###################################
    
    
class Graph:
    def __init__(self, Dimensions):
        self.num_vertices = len(Dimensions)
        self.max_modes = np.array(Dimensions) 
        self.vertices = self.node_paths()
        self.combinations = self.AllCombinations()
        self.TriggerableState = self.paths()
        
    def AllCombinations(self):
        """
        Creating all possible paths
        Parameters
        ----------
        abcd : List of sympy symbols
    
        Returns
        -------
        prod : np.array of sympy symbols
            
    
        """
        prod = 1
        for i in self.vertices:
            prod = np.kron(prod, i)
        return prod
    
    def node_paths(self):
        nodes = []
        for n in range(self.num_vertices):
            nodes.append(symbols(f"n{n}_:{self.max_modes[n]}"))
        return nodes
    
    def state_symbol(self, state):
        symb = 1
        for n, i in enumerate(state):
            symb *= self.vertices[n][i]
        return symb
    
    def paths(self):
        """
        Create the second order state, i.e. two excited edges
    
        Parameters
        ----------
        vertices : List of sympy symbol
        localDim : list of integers
    
        Returns
        -------
        FullState : List of sympy equations
            Create a full state based on the edges combinations
    
        """
        n = len(self.vertices)
        localDim = self.max_modes
        FullState = 0
        for state in self.iterables():
            whole_list = 1
            for i in range(0,n,2):
                ij = [state[i], state[i+1]]
                lis = []
                for l in itertools.product(range(localDim[ij[0]]), range(localDim[ij[1]])):
                    lis.append(symbols(f"w^{ij[0]}{ij[1]}_{l[0]}{l[1]}")*self.vertices[ij[0]][l[0]]*self.vertices[ij[1]][l[1]])
                whole_list = np.kron(whole_list, np.array(lis))
            FullState += whole_list.sum()
        return FullState


    def fidelity(self, desired_state):
        AllEquations = []
        TargetEquations = []
        for comb in self.combinations:
            newEq = np.sum([i for i in self.TriggerableState.args if str(comb) in str(i)]).subs([(comb,1)])
            for state in desired_state:
                if comb == self.state_symbol(state):
                    # This term is part of the quantum state we want
                    TargetEquations.append(newEq)
            AllEquations.append(newEq)
        # Run the Optimization 
        self.TargetEquation = TargetEquations
        NormalisationConstant2 = np.sum(np.array(AllEquations)**2)
        self.NormalisationConstant = NormalisationConstant2
        Fidelity = np.sum(np.array(TargetEquations))**2/(len(TargetEquations)*NormalisationConstant2)
        return Fidelity

    def LossFun(self, variables, Loss2fun):
        return Loss2fun(*variables)
    
    def minimize(self, Fidelity, initial_weights=[], alpha=0):
        """
        
        Parameters
        ----------
        alpha : float
            Regularization constant.
        Returns
        -------
        None.
        """
        variables = list(Fidelity.free_symbols)
        if len(initial_weights) == 0:
            initial_weights = 2 * np.random.rand(len(variables)) - 1
        loss = (1 - Fidelity) + alpha * np.sum(np.abs(variables))
        loss2fun = lambdify(variables, loss, modules="numpy")
        sol = optimize.minimize(self.LossFun, initial_weights, args=(loss2fun))
        vars_ = [(key,value) for key, value in zip(variables,sol.x)]
        return sol, vars_
        
    def ampltiudes(self, vars_):
        """
        Returns the w's of the problem: for example the GHZ of 4 qubits. This function
        return's: w_|0000> and w_|1111>
        Parameters
        ----------
        vars_ : List of tuples
            value of the w's variables based on the minimization problem.
        Returns
        -------
        list
            As in the example above [w_|0000>, w_|1111>] 
        """
        return [i.subs(vars_) for i in self.TargetEquation]
        
    def topological_optimization(self, Fidelity, initial_weights=[], alpha=0.5, loss_min=5e-2,
                                 max_counts=100, w_limit=1):
        sol, vars_ = self.minimize(Fidelity, initial_weights, alpha=0)
        weight_last = sol.x
        count = 0
        while count < max_counts:
            self.Fidelity_last = Fidelity
            variables = list(Fidelity.free_symbols)
            ith_rid = np.random.choice(np.arange(len(variables)))
            Fidelity = Fidelity.subs([(variables[ith_rid], 0)])
            new_vars = list(Fidelity.free_symbols)
            initial_weights = [i for n, i in enumerate(weight_last) if variables[n] in new_vars]
            if len(initial_weights) == 0:
                Fidelity = self.Fidelity_last
                count += 1
            else:
                sol, vars_ = self.minimize(Fidelity, initial_weights, alpha=alpha)
                w_sum = np.abs(sol.x).sum()
                if (sol.fun < loss_min) and (w_sum < w_limit):
                    count = 0
                    weight_last = sol.x
                else:
                    Fidelity = self.Fidelity_last
                    count += 1
            print(f"The solution:{sol.fun} - count:{count} - w_sum:{w_sum}")
        self.weights = {str(i[0]):i[1] for i in vars_}
        return sol, vars_
        
    # def iterables(self):
    #     n = self.num_vertices
    #     combinations = list(itertools.combinations(range(n),2))
    #     Comb = []
    #     for num, c0 in enumerate(combinations):
    #         if c0[0] == 0:
    #             cT = list(c0)
    #             for c1 in combinations:
    #                 c1 = list(c1)
    #                 flag = any(item in c1 for item in cT)
    #                 if not flag:
    #                     cT += c1
    #             Comb.append(cT)
    #     return Comb
    
    def iterables(self):
        n = self.num_vertices
        combinations = list(itertools.combinations(range(n),2))
        Comb = []
        for comb in itertools.combinations(combinations, n // 2):
            new = []
            for ii in comb:
                flag = any(item in new for item in ii)
                if not flag:
                    new += list(ii)
                else:
                    break
            if len(new) == n:
                Comb.append(new)
        return Comb
    
    def plot(self, optimization=False, filename=None):
        # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if optimization:
            if "weights" not in dir(self):
                print("--------------------------------------------------------------")
                print("Error: first, you should execute 'topological_optimization'")
                print("--------------------------------------------------------------")
                return
        n = self.num_vertices
        dim_max = max(self.max_modes)
        colorsfun = plt.cm.get_cmap("Set1",lut=dim_max)
        colors = [colorsfun(i) for i in range(dim_max)]
        angle = np.linspace(0,2*np.pi*(n-1)/n,n)
        x = np.cos(angle)
        y = np.sin(angle)
        fig, ax = plt.subplots(figsize=(5,5))
        circle = plt.Circle((0, 0), 1.2, color='sienna', alpha=0.1,edgecolor="black")
        ax.add_patch(circle)
        ref = 0.3 # Separation between weights
        edge_pos = np.linspace(-ref,ref,dim_max)
        r = 0
        for i in range(dim_max-1):
            for j in range(i+1,dim_max):
                if i != j:
                    r += 1
                    self.edge(ax, x, y, ref + 0.2*r, [colors[i], colors[j]], i, j, optimization)
                    self.edge(ax, x, y, -ref - 0.2*r, [colors[j], colors[i]], j, i, optimization)
        for i in range(dim_max):
            self.edge(ax, x, y, edge_pos[i], colors[i], i, i, optimization)

        for i in range(n):
            ax.plot(x[i], y[i], "o", markersize=20, markeredgecolor="black", markerfacecolor="white")
            ax.text(x[i] - 0.025, y[i] - 0.01, str(i))
        ax.axis('off')
        if isinstance(filename, str):
            fig.savefig(filename)
        
    def edge(self, ax, x, y, a, color, state1, state2, optimization):
        n = len(x)
        dims = self.max_modes
        for i, j in itertools.combinations(range(n), 2):
            h = np.sqrt((x[j] - x[i])**2 + (y[j] - y[i])**2)
            r = h * np.linspace(0, 1, 50)
            fr = a * ((1 - r/h)**2 - (1 - r/h))
            if  y[j] - y[i] > 1e-6:
                theta = -np.arccos((x[j]-x[i])/h)
            else:
                theta = np.arccos((x[j]-x[i])/h)
            xp = r * np.cos(theta) + fr * np.sin(theta) + x[i]
            yp = -r * np.sin(theta) + fr * np.cos(theta) + y[i]
            w_name = f"w^{i}{j}_{state1}{state2}"
            if (optimization and w_name in self.weights.keys()) or (
                    state1 < dims[i] and state2 < dims[j] and not optimization):
                if optimization:
                    norm = np.sqrt(np.sum(np.array(list(self.weights.values()))**2))
                    args = {"linewidth":3*abs(self.weights[w_name])/norm}
                else:
                    args = {}
                if isinstance(color, list):
                    ax.plot(xp[:len(xp)//2+1], yp[:len(xp)//2+1], color=color[1], **args)
                    ax.plot(xp[len(xp)//2:], yp[len(xp)//2:], color=color[0], **args)
                else:
                    ax.plot(xp, yp, color=color, **args)
            

##########################
##########################
###                    ###
###   IGNORE FOR NOW   ###
###                    ###
##########################
##########################
    
    
# class Graph:
#     def __init__(self, dimensions):
#         '''
#         Initialize a fully connected graph with all possible edges given the dimension.
#         It computes only the minimum order such that all detectors are triggered.
        
#         Instances
#         ---------
#         num_vertices: number of vertices (particles/detectors) of the network (system)
#         max_modes: number of dimensions accessible to each vertix
#         vertices: list of employed dimensions for each vertix. It starts with all possible 
#             dimensions only once. One can add/remove afterwords, even repeating them. The 
#             repeated nodes/creators can be used to compute higher order terms.
#         combinations: given a system of {num_vertices} vertices with a given dimension for
#             each of them, these enumerate all possible basic kets that can be produced. It 
#             only computes lowest order terms so it may not help for high order terms.
#         perf_matchings: compute all posible pairs that can be formed with an even number of 
#             vertices. It does not consider dimensions.
#         edges: list of network edges written as sympy symbols. It starts with all possible 
#             edges given the dimensions of each node.
#         '''
#         self.num_vertices = len(dimensions)
#         self.max_modes = np.array(dimensions) 
#         self.vertices = [[(nn,ii) for ii in range(dim)] for nn, dim in enumerate(dimensions)]
#         self.combinations = list(itertools.product(*[list(range(dim)) for dim in dimensions]))
#         self.perf_matchings = list(allPairSplits(list(range(self.num_vertices))))
#         self.edges = buildAllEdges(dimensions) # A dictionary counting how many we use could help
        
#     def lowOrderTerms(self):
#         '''
#         Given all posible kets described in self.combinations and all possible perfect matchings,
#         write all the edge contributions that leads to each of the states that trigger all 
#         detectors, each with only one particle (i.e. it does not consider higher order terms).
#         The terms are written with symbols, from Sympy library.
#         '''
#         low_ord_terms = []
#         for comb in self.combinations:
#             term_sumation = 0
#             for pm in self.perf_matchings:
#                 term = 1
#                 for pair in pm:
#                     term *= symbols(f'w^{pair[0]}{pair[1]}_{comb[pair[0]]}{comb[pair[1]]}')
#                 term_sumation += term
#             yield term_sumation
#         return low_ord_terms
        
#     def allNodes(self):
#         '''
#         List of all current nodes (creator operators) including repeated ones.
#         '''
#         lst_nodes = [item for sublist in self.vertices for item in sublist]
#         return lst_nodes
    
#     # def add_nodes(self, nodes_list):
#     #     '''
#     #     Add new nodes to the existing ones. Usually in pairs for the higher order equations.
#     #     '''
#     #     for node in nodes_list:
#     #         self.vertices[node[0]].append(node)
#     #         self.vertices[node[0]].sort()    
        
#     def allCombinations(self):
#         """
#         Creating all possible paths
#         Parameters
#         ----------
#         abcd : List of sympy symbols
    
#         Returns
#         -------
#         prod : np.array of sympy symbols
       
    
#         """
#         prod = 1
#         for i in self.vertices:
#             prod = np.kron(prod, i)
#         return prod
    
#     def node_paths(self):
#         nodes = []
#         for n in range(self.num_vertices):
#             nodes.append(symbols(f"n{n}_:{self.max_modes[n]}"))
#         return nodes
    
#     def state_symbol(self, state):
#         symb = 1
#         for n, i in enumerate(state):
#             symb *= self.vertices[n][i]
#         return symb
    
#     def paths(self):
#         """
#         Create the second order state, i.e. two excited edges
    
#         Parameters
#         ----------
#         vertices : List of sympy symbol
#         localDim : list of integers
    
#         Returns
#         -------
#         FullState : List of sympy equations
#             Create a full state based on the edges combinations
    
#         """
#         n = len(self.vertices)
#         localDim = self.max_modes
#         FullState = 0
#         for state in self.iterables():
#             whole_list = 1
#             for i in range(0,n,2):
#                 ij = [state[i], state[i+1]]
#                 lis = []
#                 for l in itertools.product(range(localDim[ij[0]]), range(localDim[ij[1]])):
#                     lis.append(symbols(f"w^{ij[0]}{ij[1]}_{l[0]}{l[1]}")*self.vertices[ij[0]][l[0]]*self.vertices[ij[1]][l[1]])
#                 whole_list = np.kron(whole_list, np.array(lis))
#             FullState += whole_list.sum()
#         return FullState


#     def fidelity(self, desired_state):
#         AllEquations = []
#         TargetEquations = []
#         for comb in self.combinations:
#             newEq = np.sum([i for i in self.TriggerableState.args if str(comb) in str(i)]).subs([(comb,1)])
#             for state in desired_state:
#                 if comb == self.state_symbol(state):
#                     # This term is part of the quantum state we want
#                     TargetEquations.append(newEq)
#             AllEquations.append(newEq)
#         # Run the Optimization 
#         self.TargetEquation = TargetEquations
#         NormalisationConstant2 = np.sum(np.array(AllEquations)**2)
#         self.NormalisationConstant = NormalisationConstant2
#         Fidelity = np.sum(np.array(TargetEquations))**2/(len(TargetEquations)*NormalisationConstant2)
#         return Fidelity

#     def LossFun(self, variables, Loss2fun):
#         return Loss2fun(*variables)
    
#     def minimize(self, Fidelity, initial_weights=[], alpha=0):
#         """
        

#         Parameters
#         ----------
#         alpha : float
#             Regularization constant.

#         Returns
#         -------
#         None.

#         """
#         variables = list(Fidelity.free_symbols)
#         if len(initial_weights) == 0:
#             initial_weights = 2 * np.random.rand(len(variables)) - 1
#         loss = (1 - Fidelity) + alpha * np.sum(np.abs(variables))
#         loss2fun = lambdify(variables, loss, modules="numpy")
#         sol = optimize.minimize(self.LossFun, initial_weights, args=(loss2fun))
#         vars_ = [(key,value) for key, value in zip(variables,sol.x)]
#         return sol, vars_
        
#     def ampltiudes(self, vars_):
#         """
#         Returns the w's of the problem: for example the GHZ of 4 qubits. This function
#         return's: w_|0000> and w_|1111>

#         Parameters
#         ----------
#         vars_ : List of tuples
#             value of the w's variables based on the minimization problem.

#         Returns
#         -------
#         list
#             As in the example above [w_|0000>, w_|1111>] 

#         """
#         return [i.subs(vars_) for i in self.TargetEquation]
        
#     def topological_optimization(self, Fidelity, initial_weights=[], alpha=0.5, loss_min=5e-2,
#                                  max_counts=100, w_limit=1):
#         sol, vars_ = self.minimize(Fidelity, initial_weights, alpha=0)
#         weight_last = sol.x
#         count = 0
#         while count < max_counts:
#             self.Fidelity_last = Fidelity
#             variables = list(Fidelity.free_symbols)
#             ith_rid = np.random.choice(np.arange(len(variables)))
#             Fidelity = Fidelity.subs([(variables[ith_rid], 0)])
#             new_vars = list(Fidelity.free_symbols)
#             initial_weights = [i for n, i in enumerate(weight_last) if variables[n] in new_vars]
#             if len(initial_weights) == 0:
#                 Fidelity = self.Fidelity_last
#                 count += 1
#             else:
#                 sol, vars_ = self.minimize(Fidelity, initial_weights, alpha=alpha)
#                 w_sum = np.abs(sol.x).sum()
#                 if (sol.fun < loss_min) and (w_sum < w_limit):
#                     count = 0
#                     weight_last = sol.x
#                 else:
#                     Fidelity = self.Fidelity_last
#                     count += 1
#             print(f"The solution:{sol.fun} - count:{count} - w_sum:{w_sum}")
#         self.weights = {str(i[0]):i[1] for i in vars_}
#         return sol, vars_
        
#     # def iterables(self):
#     #     n = self.num_vertices
#     #     combinations = list(itertools.combinations(range(n),2))
#     #     Comb = []
#     #     for num, c0 in enumerate(combinations):
#     #         if c0[0] == 0:
#     #             cT = list(c0)
#     #             for c1 in combinations:
#     #                 c1 = list(c1)
#     #                 flag = any(item in c1 for item in cT)
#     #                 if not flag:
#     #                     cT += c1
#     #             Comb.append(cT)
#     #     return Comb
    
#     def iterables(self):
#         n = self.num_vertices
#         combinations = list(itertools.combinations(range(n),2))
#         Comb = []
#         for comb in itertools.combinations(combinations, n // 2):
#             new = []
#             for ii in comb:
#                 flag = any(item in new for item in ii)
#                 if not flag:
#                     new += list(ii)
#                 else:
#                     break
#             if len(new) == n:
#                 Comb.append(new)
#         return Comb
    
#     def plot(self, optimization=False, filename=None):
#         # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#         if optimization:
#             if "weights" not in dir(self):
#                 print("--------------------------------------------------------------")
#                 print("Error: first, you should execute 'topological_optimization'")
#                 print("--------------------------------------------------------------")
#                 return
#         n = self.num_vertices
#         dim_max = max(self.max_modes)
#         colorsfun = plt.cm.get_cmap("Set1",lut=dim_max)
#         colors = [colorsfun(i) for i in range(dim_max)]
#         angle = np.linspace(0,2*np.pi*(n-1)/n,n)
#         x = np.cos(angle)
#         y = np.sin(angle)
#         fig, ax = plt.subplots(figsize=(5,5))
#         circle = plt.Circle((0, 0), 1.2, color='sienna', alpha=0.1,edgecolor="black")
#         ax.add_patch(circle)
#         ref = 0.3 # Separation between weights
#         edge_pos = np.linspace(-ref,ref,dim_max)
#         r = 0
#         for i in range(dim_max-1):
#             for j in range(i+1,dim_max):
#                 if i != j:
#                     r += 1
#                     self.edge(ax, x, y, ref + 0.2*r, [colors[i], colors[j]], i, j, optimization)
#                     self.edge(ax, x, y, -ref - 0.2*r, [colors[j], colors[i]], j, i, optimization)
#         for i in range(dim_max):
#             self.edge(ax, x, y, edge_pos[i], colors[i], i, i, optimization)

#         for i in range(n):
#             ax.plot(x[i], y[i], "o", markersize=20, markeredgecolor="black", markerfacecolor="white")
#             ax.text(x[i] - 0.025, y[i] - 0.01, str(i))
#         ax.axis('off')
#         if isinstance(filename, str):
#             fig.savefig(filename)
        
#     def edge(self, ax, x, y, a, color, state1, state2, optimization):
#         n = len(x)
#         dims = self.max_modes
#         for i, j in itertools.combinations(range(n), 2):
#             h = np.sqrt((x[j] - x[i])**2 + (y[j] - y[i])**2)
#             r = h * np.linspace(0, 1, 50)
#             fr = a * ((1 - r/h)**2 - (1 - r/h))
#             if  y[j] - y[i] > 1e-6:
#                 theta = -np.arccos((x[j]-x[i])/h)
#             else:
#                 theta = np.arccos((x[j]-x[i])/h)
#             xp = r * np.cos(theta) + fr * np.sin(theta) + x[i]
#             yp = -r * np.sin(theta) + fr * np.cos(theta) + y[i]
#             w_name = f"w_{i}{j}_{state1}{state2}"
#             if (optimization and w_name in self.weights.keys()) or (
#                     state1 < dims[i] and state2 < dims[j] and not optimization):
#                 if optimization:
#                     norm = np.sqrt(np.sum(np.array(list(self.weights.values()))**2))
#                     args = {"linewidth":3*abs(self.weights[w_name])/norm}
#                 else:
#                     args = {}
#                 if isinstance(color, list):
#                     ax.plot(xp[:len(xp)//2+1], yp[:len(xp)//2+1], color=color[1], **args)
#                     ax.plot(xp[len(xp)//2:], yp[len(xp)//2:], color=color[0], **args)
#                 else:
#                     ax.plot(xp, yp, color=color, **args)