#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cruizgo, soerenarlt, janpe
"""
import itertools, numbers, random
from math import factorial
from collections import Counter
import numpy as np
import warnings


import logging

log = logging.getLogger(__name__)


# # Auxiliary Functions
# Helpful functions used in many processes but not by the final user. 

def allPairSplits(lst):
    '''     
    Generate all sets of unique pairs from a list `lst`.
    This is equivalent to all partitions of `lst` (considered as an indexed set) which have 2 elements
    in each partition.
    
    Recall how we compute the total number of such partitions. Starting with a list [1, 2, 3, 4, 5, 6]
    one takes off the first element, and chooses its pair [from any of the remaining 5]. For example, 
    we might choose our first pair to be (1, 4). Then, we take off the next element, 2, and choose 
    which element it is paired to (say, 3). So, there are 5 * 3 * 1 = 15 such partitions.
    That sounds like a lot of nested loops (i.e. recursion), because 1 could pick 2, in which case our 
    next element is 3. But, if one abstracts "what the next element is", and instead just thinks of what 
    index it is in the remaining list, our choices are static and can be aided by the product function.
    
    From selfgatoatigrado: https://stackoverflow.com/a/13020502
    '''
    N = len(lst)
    choice_indices = itertools.product(*[range(k) for k in range(N-1, 0, -2)])

    for choice in choice_indices:
        # calculate the list corresponding to the choices
        tmp = lst[:]
        result = []
        for index in choice:
            result.append((tmp.pop(0), tmp.pop(index)))
        yield result  # use yield and then turn it into a list is faster than append


def targetEdges(nodes, graph, isolate=False):
    '''
    Returns all graph's edges which connect to the input nodes.
    '''
    if isolate:
        return [edge for edge in graph if (edge[0] in nodes) and (edge[1] in nodes)]
    else:
        return [edge for edge in graph if (edge[0] in nodes) or (edge[1] in nodes)]


def removeNodes(nodes, graph):
    '''
    Removes all graph's edges which connect to the input nodes.
    '''
    return [edge for edge in graph if not ((edge[0] in nodes) or (edge[1] in nodes))]


def deadEndEdges(graph):  # Unused
    '''
    Returns all edges connecting nodes with degree one.
    '''
    unq, counts = np.unique(np.array(graph)[:, :2], return_counts=True)
    return targetEdges(unq[counts == 1], graph)


def edgeBleach(color_edges):
    '''
    Given a list of colored edges, returns a dictionary with the available colors (values) for each 
    of the edges (keys).
    
    This function helps to speed up the function `findPerfectMatchings`.
    
    Parameters
    ----------
    color_edges : list of tuples
        List of all colored edges: [(node1, node2, color1, color2), ...].
        
    Returns
    -------
    bleached_edges : dictionary
        Dictionary with the color combinations available for each edge:
        {(node1, node2): [(color1, color2), ...], (node1, node3): [(color2, color1), ...], ...}
    '''
    raw_edges = np.unique(np.array(color_edges)[:, :2], axis=0)
    bleached_edges = {tuple(edge): [] for edge in raw_edges}
    for edge in color_edges:
        bleached_edges[edge[:2]].append(edge[2:])
    return bleached_edges


def edgePainting(graph_list, avail_colors):
    '''
    Takes a list of uncolored graphs and 'paint them' with the
    dictionary of `avail_colors`.
    
    Parameters
    ----------
    graph_list : list of tuples
        List of subgraphs: perfect matchings, edge covers... 
        The colors of the edges, if any, are ignored.
    avail_colors : dictionary
        Dictionary with the color combinations available for each edge (see edgeBleach).
        {(node1, node2): [(color1, color2), ...], (node1, node3): [(color2, color1), ...], ...}

    Returns
    -------
    colored_graphs : list of tuples
        List of graphs with all possible color combinations.
    '''
    graph_list = np.unique(np.array(graph_list)[:,:,:2],axis=0)
    painted_subgraphs = []
    for subgraph in graph_list:
        for coloring in itertools.product(*[avail_colors[tuple(edge)] for edge in subgraph]):
            color_subgraph = [tuple(edge) + color for edge, color in zip(subgraph, coloring)]
            painted_subgraphs.append(sorted(color_subgraph))
    return [tuple(map(tuple,graph)) for graph in np.unique(painted_subgraphs, axis=0)]


def allColorGraphs(color_nodes, loops=False):
    '''
    Given a list of colored nodes, i.e., an state. It uses AllPairSplits to generate all graphs that
    leads to such state. This function is a building block of the function 'allEdgeCovers'.
    
    Parameters
    ----------
    color_nodes : list of tuples
        List of all colored nodes: [(node,color)...] Some may be repeated.
    loops : boolean, optional
        Allow edges to connect twice the same node: (node1, node1, color1, color2).
        
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
    if loops:
        color_graph = [[[nd[0][0], nd[1][0], nd[0][1], nd[1][1]] for nd in graph]
                       for graph in color_graph]
    else:
        color_graph = [[[nd[0][0], nd[1][0], nd[0][1], nd[1][1]] for nd in graph
                        if nd[0][0] != nd[1][0]] for graph in color_graph]
        color_graph = [graph for graph in color_graph if len(graph) == len(color_nodes) / 2]
    return [tuple(map(tuple,graph)) for graph in np.unique(color_graph, axis=0)]


# # Informative Functions
# Return some information about dimensions, edges, graphs...

def nodeDegrees(edge_list, nodes_list=[], increasing=True):
    '''
    Compute the degree of each node of a graph. Returning a list of tuples such that: 
    [(node1, degree1), (node2, degree2) ...]
    By default, it sorts the nodes by their degree in increasing order.
    
    Parameters
    ----------
    edge_list : list of tuples
        List of all available edges: [(node1, node2, color1, color2), ...]
    nodes_list : list, optional
        List of all nodes. By default, the list is obtained from the nodes that appear in edge_list.
    increasing : boolean, optional
        If True, the nodes are ordered by degree.
    
    Returns
    -------
    links : list of tuples
        List of nodes and their degrees: [(node1, degree1), ...]
    '''
    if len(nodes_list) == 0: nodes_list = np.unique(np.array(edge_list)[:, :2])
    links = {ii: 0 for ii in nodes_list}
    for edge in edge_list:
        links[edge[0]] += 1
        links[edge[1]] += 1
    if increasing:
        return sorted(links.items(), key=lambda item: item[1])
    else:
        return [(k, v) for k, v in links.items()]


def creatorList(edge_list, allow_rep = False):
    '''
    From a list of edges, returns a list of the employed creator operators.
    '''
    creator_list = []
    for ed in edge_list: creator_list += [(ed[0], ed[2]), (ed[1], ed[3])]
    if allow_rep:
        return sorted(creator_list)
    else:
        return sorted(set(creator_list))


def graphDimensions(edge_list):
    '''
    Estimate the dimensions of a graph based on its edges.

    The output dimensions are in decreasing order: we assume the dimension of a node N, is equal or
    larger than the dimension of a node N+1.
    
    Parameters
    ----------
    edge_list : list of tuples
        List of all available edges: [(node1, node2, color1, color2), ...]
    
    Returns
    -------
    dimensions : list
        List with the dimensions accessible to each node in decreasing order. 
    '''
    color_nodes = np.array(creatorList(edge_list))
    num_nodes = color_nodes[-1,0] + 1
    return [1 + color_nodes[color_nodes[:,0]==ii,1].max() for ii in range(num_nodes)]


def stateDimensions(ket_list):
    '''
    Given a state, it returns the dimensions accessible to each particle.
    
    Parameters
    ----------
    ket_list : list of tuples
        List of kets written as: [((node1, dimension1), (node2, dimension2), ...), ...].
    
    Returns
    -------
    dimensions : list
        List with the dimensions accessible to each particle. 
    '''
    dimensions = [1] * len(ket_list[0])
    for ket in ket_list:
        for ii, local_dim in enumerate(ket):
            dimensions[ii] = np.maximum(dimensions[ii], local_dim[1] + 1)
    return dimensions


# # Generators
# Used to produce edges, graphs, states...

def stateCatalog(graph_list):
    '''
    Given a list of colored graphs, returns a dictionary with graphs grouped by the state they generate.
    Each of these states is a dictionary key.
    
    Parameters
    ----------
    graph_list : list
        List of graphs.
    
    Returns
    -------
    state_catalog : dictionary
        Dictionary with all the kets created by each graph as keys. If more than one graph leads to
        the same state, they are listed together in the corresponding entrance of the dictionary.
    '''
    state_catalog = {}
    for graph in graph_list:
        coloring = []
        for ed in graph: coloring += [(ed[0], ed[2]), (ed[1], ed[3])]
        coloring = tuple(sorted(coloring))
        try:
            state_catalog[coloring] += [graph]
        except KeyError:
            state_catalog[coloring] = [graph]
    return state_catalog


def mixedCatalog(traced_nodes, state_catalog):
    '''
    Given a catalog of states, it groups them based on the state on the traced nodes.
    This function helps in the computation of mixed states.
    
    Parameters
    ----------
    traced_nodes : list of integers
        List of the nodes which are traced out.
    state_catalog : dictionary, optional
        Dictionary with all the kets created by each graph as keys. If more than one graph leads to
        the same state, they are listed together in the corresponding entrance of the dictionary.
    
    Returns
    -------
    mixed_catalog : dictionary
        Dictionary of state_catalogs grouped by the state of the traced nodes.
    '''
    mixed_catalog = dict()
    for ket, graph_list in state_catalog.items():
        traced_ket = tuple(node for node in ket if node[0] in traced_nodes)
        try:
            mixed_catalog[traced_ket][ket] = graph_list
        except KeyError:
            mixed_catalog[traced_ket] = {ket:graph_list}
    return mixed_catalog   


def buildAllEdges(dimensions, string=False, imaginary=False, loops=False):
    '''
    Given a collection of nodes, each with several possible colors/dimensions, returns all possible
    edges of the graph.
    
    Parameters
    ----------
    dimensions : list
        Accesible dimensions/colors for each of the nodes of the graph.
    string : boolean, optional
        If True, it returns a list of strings instead of tuples. 
    imaginary : boolean, str ('cartesian' or 'polar'), optional
        If False, it returns real weights.
    loops : boolean, optional
        Allow edges to connect twice the same node.
        
    Returns
    -------
    all_edges : list
        List of all possible edges given the dimensions of the nodes. 
        If string, it returns a list of strings.
        Else, it returns a list of tuples: (node1, node2, color1, color2).
    '''
    num_vertices = len(dimensions)
    all_edges = []
    if loops:
        combo_function = itertools.combinations_with_replacement
        for pair in combo_function(range(num_vertices), 2):
            if pair[0] == pair[1]:  # (node1, node1, 1, 0) is not stored
                for dims in combo_function(range(dimensions[pair[0]]), 2):
                    all_edges.append((pair[0], pair[1], dims[0], dims[1]))
            else:
                for dims in itertools.product(*[range(dimensions[ii]) for ii in pair]):
                    all_edges.append((pair[0], pair[1], dims[0], dims[1]))
    else:
        for pair in itertools.combinations(range(num_vertices), 2):
            for dims in itertools.product(*[range(dimensions[ii]) for ii in pair]):
                all_edges.append((pair[0], pair[1], dims[0], dims[1]))
    # returns edges whether as tuples or as sympy symbols
    if string:
        if imaginary == False:
            return [edgeWeight(edge) for edge in all_edges]
        else:
            return (['r_{}_{}_{}_{}'.format(*edge) for edge in all_edges] +
                    ['th_{}_{}_{}_{}'.format(*edge) for edge in all_edges])
    else:
        return all_edges


def stringEdges(edge_list, imaginary=False):
    '''
    Given a list of edges written as tuples it returns them in string format.
    
    Parameters
    ----------
    edge_list : list of tuples
        List of edges: [(node1, node2, color1, color2), ...]   
    imaginary : boolean, str ('cartesian' or 'polar'), optional
        If False, it returns real weights. Else, the weights are complex and defined by string pairs. 
        
    Parameters
    ----------
    list of str
        List of weight' strings.
    '''
    if imaginary == False:
        return ['w_{}_{}_{}_{}'.format(*edge) for edge in edge_list]
    else:
        return (['r_{}_{}_{}_{}'.format(*edge) for edge in edge_list]
                + ['th_{}_{}_{}_{}'.format(*edge) for edge in edge_list])


def buildRandomGraph(dimensions, num_edges, cover_all=True, loops=False):
    '''
    Given a set of nodes with different dimensions it build a random graph with a given number of edges.
    
    Parameters
    ----------
    dimensions : list
        Accesible dimensions/colors for each of the nodes of the graph.
    num_edges : int
        Number of edges of the random graph.
    cover_all : boolean, optional
        If True, it makes sure the selected edges cover all the nodes of the graph.   
        
    Returns
    -------
    graph : list of tuples
        List of randomly chosen edges: [(node1, node2, color1, color2), ...]
    '''
    all_edges = buildAllEdges(dimensions, loops=loops)
    if cover_all:
        num_nodes = len(dimensions)
        if 2 * num_edges >= num_nodes:
            all_covered = False
        else:
            raise ValueError('num_edges is too low to cover all nodes')
        while not all_covered:
            graph = random.sample(all_edges, num_edges)
            all_covered = (len(np.unique(np.array(graph)[:, :2])) == num_nodes)
        return sorted(graph)
    else:
        return sorted(random.sample(all_edges, num_edges))


def allPerfectMatchings(dimensions):
    '''
    Given a collection of nodes with different dimensions/colors available, it produces all possible
    states that erase from these and the different graphs that can produce such states.
    The graphs nodes can present different colors/dimensions and be connected by more than one edge.

    A fully connected (uncolored) graph with 2n nodes has (2n)!/((2**n)⋅n!) perfect matchings.

    Parameters
    ----------
    dimensions : list
        Accesible dimensions/colors for each of the nodes of the graph.
        
    Returns
    -------
    color_dict : dictionary
        Dictionary of available states. The keys are the different combinations of colored nodes
        (creator operators), that is, the state produced. 
        For each or state, the dictionary stores a list of all possible graphs that produce such state.
        The notation employed for the nodes is: (node, color/dimension).
        The notation employed for the edges is: (node1, node2, color1, color2).
        Node2 cannot be lower than Node1.
    '''
    num_nodes = len(dimensions)
    crowded_graph = [list(range(num_nodes))]
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
        color_nodes = [[tuple(ed) for ed in graph] for graph in np.unique(color_nodes, axis=0)]
        for coloring in color_nodes:
            color_dict[tuple(coloring)] = allColorGraphs(coloring)
    return color_dict


def allEdgeCovers(dimensions, order=0, loops=False): # This function should replace allPerfectMatchings
    '''
    Given a collection of nodes with different dimensions/colors available, it produces all possible
    states that erase from these and the different graphs that can produce such states.
    The graphs nodes may have different colors/dimensions. The edges may be duplicated or form loops.

    Parameters
    ----------
    dimensions : list
        Accesible dimensions/colors for each of the nodes of the graph.
    order : int, optional
        Orders above the minimum required to cover all nodes.
        If 0, the output are only perfect machings, with all nodes having degree 1.
    loops : boolean, optional
        Allow edges to connect twice the same node. Only available for order>0.

    Returns
    -------
    color_dict : dictionary
        Dictionary of available states. The keys are the different combinations of colored nodes
        (creator operators), that is, the state produced.
        For each or state, the dictionary stores a list of all possible graphs that produce such state.
        The notation employed for the nodes is: (node, color/dimension).
        The notation employed for the edges is: (node1, node2, color1, color2).
        Node2 cannot be lower than Node1.
    '''
    num_nodes = len(dimensions)
    # Given a list with N different nodes, if order>0 some nodes will have degree
    # larger than 1, i.e., their creator operators will be repeated.
    # crowded_graph stores all ways in which these repeatitions may occur.
    # For N=4 and order=1: 000123, 001123, 001223, 001233, 011123, 011223...
    additions = list(itertools.combinations_with_replacement(range(num_nodes), 2 * order))
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
        color_nodes = [[tuple(ed) for ed in graph] for graph in np.unique(color_nodes, axis=0)]
        for coloring in color_nodes:
            color_dict[tuple(coloring)] = allColorGraphs(coloring, loops)
    return color_dict


def allFockStates(photons_per_modes, ancillas=None): 
    '''
    Generation of all possible Fock states. Loops are allowed.

    Parameters
    ----------
    photons_per_modes : list of lists of integers
        List of the photons to create in different modes:
        [[N1 ph, M1 modes], [N2 ph, M2 modes], [N3 ph, M3 modes], ...]
    ancillas : int, optional
        Ancilla detectors reached by one photon each.

    Returns
    -------
    fock_dict : dictionary
        Dictionary of available Fock states. For each or state, the dictionary 
        stores a list of all possible graphs that produce such state.
        The notation employed for the nodes is: (node, color/dimension).
        The notation employed for the edges is: (node1, node2, color1, color2).
        Node2 cannot be lower than Node1.
        We keep the notation of other dictionaries, but the color is always 0.
    '''
    assert all(isinstance(num,numbers.Integral) for num in sum(photons_per_modes, []))
    total_photons = sum(pair[0] for pair in photons_per_modes)
    total_modes = sum(pair[1] for pair in photons_per_modes)
    if ancillas is None:
        ancillas = total_photons % 2
    elif (total_photons + ancillas) % 2 == 0:
        pass # all good
    else:
        raise ValueError('The total number of photons must be even. Modify photons/ancillas.')
    ancilla_nodes = tuple(range(total_modes,total_modes + ancillas))
    ancilla_nodes_long = tuple((anc,0) for anc in ancilla_nodes)
    
    prod_states = []
    mode_count = 0
    for photons, modes in photons_per_modes:
        prod_states.append(list(itertools.combinations_with_replacement(range(mode_count,mode_count+modes), photons)))
        mode_count += modes
        
    fock_dict = dict()
    for prod in itertools.product(*prod_states):
        state = sum(prod,())
        terms = sorted(sorted(graph) for graph in allPairSplits(list(state+ancilla_nodes)))
        terms = [tuple(tuple(edge)+(0,0) for edge in graph) for graph in np.unique(terms,axis=0)]
        fock_dict[tuple((node,0) for node in state) + ancilla_nodes_long] = terms
    return fock_dict


def recursivePerfectMatchings(graph, store, matches=[], edges_left=None):
    '''
    The heavy lifting of findPerfectMatchings.
    '''
    if edges_left is None: edges_left = len(np.unique(np.array(graph)[:, :2])) / 2
    if len(graph) > 0:
        for edge in targetEdges([nodeDegrees(graph)[0][0]], graph):
            recursivePerfectMatchings(removeNodes(edge[:2], graph), store,
                                      matches + [edge], edges_left - 1)
    elif len(graph) == 0 and edges_left == 0:
        store.append(sorted(matches))
    else:
        pass  # Some nodes are not matched and never will


def findPerfectMatchings(graph):
    '''
    Returns all possible perfect matchings (if any) of a graph.
    
    It starts by taking away the color/dimensions of the edges. Once the perfect matchings are build
    for the uncolored edges, it 'paint' them again with all possible combinations.
    
    Parameters
    ----------
    graph : list of tuples
        List of edges: [(node1, node2, color1, color2), ...].
        
    Returns
    -------
    painted_matchings : list of tuples
        Lists of all the possible perfect matchings of the input graph.
    '''
    avail_colors = edgeBleach(graph)
    raw_matchings = []
    recursivePerfectMatchings(list(avail_colors.keys()), raw_matchings)
    painted_matchings = []
    for match in raw_matchings:
        for coloring in itertools.product(*[avail_colors[edge] for edge in match]):
            color_match = [edge + color for edge, color in zip(match, coloring)]
            painted_matchings.append(tuple(color_match))
    return painted_matchings


def findEdgeCovers(graph, edges_left=None, nodes_left=None, order=1, loops=False):
    '''
    Given a list of edges, returns all possible edge covers, up to a certain
    order or using a certain number of edges.

    Parameters
    ----------
    graph : list
        List of all available edges.
    edges_left : int, optional
        Total number of edges of each edge cover. If None, this is computed
        as the minimum edges required to cover all nodes plus the order.
    nodes_left : iterable of int, optional
        List of all nodes to be covered. By default, the list is obtained
        from the nodes covered in graph but it can be set to target
        specific nodes.
    order : int, optional
        Orders above the minimum required to cover all nodes. If 0, the
        output are only perfect machings, with all nodes having degree 1.
    loops : boolean, optional NOT USED ANYMORE
        Allow edges to connect twice the same node. These cannot appear for
        perfect matchings.

    Returns
    -------
    covers : list
        List of graphs with all nodes having, at least, degree 1.
    '''
    # We ignore the edge colors until the very end
    avail_colors = edgeBleach(graph)
    raw_edges = sorted(avail_colors.keys())

    if nodes_left is None: 
        nodes_left = set(np.array(raw_edges)[:, :2].flat)
    if edges_left is None: 
        edges_left = order + (len(nodes_left) // 2)

    # Create a set of all possible subsets of edges
    raw_covers = set()
    for subgraph in itertools.combinations_with_replacement(raw_edges, edges_left):
        nodes_covered = set(sum(subgraph, ()))
        if all(node in nodes_covered for node in nodes_left):
            raw_covers.add(subgraph)

    painted_covers = set()
    for cover in raw_covers:
        for coloring in itertools.product(*[avail_colors[edge] for edge in cover]):
            painted_covers.add(tuple(sorted(edge + color
                                            for edge, color in zip(cover, coloring))))
    return sorted(painted_covers)


# # String Expressions
# Write the string expressions used in the optimization.

def edgeWeight(edge, imaginary=False):
    '''
    It writes the edge's weight as needed for `writeNorm`, `targetEquation`, and others.
    
    Parameters
    ----------
    edge : tuple
        Edge in tuple notation: (node1, node2, color1, color2).
    imaginary : boolean, str ('cartesian' or 'polar'), optional
        If False, the final expression uses only real weights. If 'cartesian' or 'polar', the weights 
        are complex values written in the corresponding notation.
        
    Returns
    -------
    str
        Edge's weight written as string for contributions on a particular ket.
    '''
    if imaginary == False:
        return 'w_{}_{}_{}_{}'.format(*edge)
    elif imaginary == 'cartesian':
        return '(r_{}_{}_{}_{}+1j*th_{}_{}_{}_{})'.format(*edge * 2)
    elif imaginary == 'polar':
        return 'r_{}_{}_{}_{}*np.exp(1j*th_{}_{}_{}_{})'.format(*edge * 2)
    else:
        raise ValueError('Introduce a valid input `imaginary`.')


def weightProduct(edge_list, imaginary=False):
    '''
    It writes the product of edge's weights as needed for `writeNorm`, `targetEquation`, and others.
    
    Parameters
    ----------
    edge_list : list of tuples
        List of edges: [(node1, node2, color1, color2), ...].
    imaginary : boolean, str ('cartesian' or 'polar'), optional
        If False, the final expression uses only real weights. If 'cartesian' or 'polar', the weights 
        are complex values written in the corresponding notation.
        
    Returns
    -------
    str
        Product of edge's weights written as string for contributions on a particular ket.
    '''
    return '*'.join([edgeWeight(edge, imaginary) for edge in edge_list])


# This expression could also be UNlambdified so we add some check for the Counter
factorialProduct = lambda lst: np.prod([factorial(ii) for ii in Counter(lst).values()])

def writeNorm(state_catalog, imaginary=False, hot=False):
    '''
    Write a normalization function with all the states of a dictionary.
    
    Parameters
    ----------
    state_catalog : dictionary
        Dictionary with all the kets created by each graph as keys. If more than one graph leads to
        the same state, they are listed together in the corresponding entrance of the dictionary.
    imaginary : boolean, str ('cartesian' or 'polar'), optional
        If False, the final expression uses only real weights. If 'cartesian' or 'polar', the weights 
        are complex values written in the corresponding notation.
    hot : boolean, optional
        Higher Order Terms. If True, it computes the coefficients derived from edge/node repetition.
        
    Returns
    -------
    str
        Norm string with the contributions of each perfect matching to their corresponding ket.
    '''
    norm_sum = []
    if hot:
        for key, values in state_catalog.items():
            # The division with factorialProduct comes from combinatoric reasons: wikipedia.org/wiki/Multinomial_theorem
            term_sum = [f'{weightProduct(graph, imaginary)}/{factorialProduct(graph)}' for graph in values]
            term_sum = ' + '.join(term_sum)
            if imaginary == False:
                # The multiplication with factorialProduct from creator operators: wikipedia.org/wiki/Creation_operators
                # The factor should be squared or, in this case, left outside the square
                norm_sum.append(f'{factorialProduct(key)}*(({term_sum})**2)')
            else:
                norm_sum.append(f'{factorialProduct(key)}*(abs({term_sum})**2)')
        return ' + '.join(norm_sum).replace('/1 +', ' +').replace('/1)', ')').replace('+ 1*', '+ ')
    else:
        for key, values in state_catalog.items():
            term_sum = ' + '.join([f'{weightProduct(graph, imaginary)}' for graph in values])
            if term_sum == '':
                term_sum = '0'
            if imaginary == False:
                # The multiplication with factorialProduct from creator operators: wikipedia.org/wiki/Creation_operators
                # The factor should be squared or, in this case, left outside the square
                norm_sum.append(f'(({term_sum})**2)')
            else:
                norm_sum.append(f'(abs({term_sum})**2)')
        norm_sum = ' + '.join(norm_sum)
        return norm_sum


def targetEquation(ket_list, amplitudes=None, state_catalog=None, imaginary=False):
    '''
    Given a list of target kets, it writes a non-normalized fidelity function.
    
    If no state_catalog is introduced it builds all possible graphs that generate the desired kets.
    
    Parameters
    ----------
    ket_list : list of tuples
        List of target kets written as: [((node1, dimension1), (node2, dimension2), ...), ...].   
    amplitudes : list, optional
        List of the amplitudes we aim for each ket. If None, all will be the same.
    state_catalog : dictionary, optional
        Dictionary with all the kets created by each graph as keys. If more than one graph leads to
        the same state, they are listed together in the corresponding entrance of the dictionary.
    imaginary : boolean, str ('cartesian' or 'polar'), optional
        If False, the final expression uses only real weights. If 'cartesian' or 'polar', the weights 
        are complex values written in the corresponding notation.
        
    Returns
    -------
    str
        Non-normalized fidelity string with all the contributing weights.
    '''
    if amplitudes is None:
        amplitudes = [1] * len(ket_list)
    else:
        if len(amplitudes) != len(ket_list):
            raise ValueError('The number of amplitudes and kets should be the same')
    if imaginary == 'polar':
        amplitudes = [amp[0] * np.exp(1j * amp[1]) for amp in amplitudes]
    norm2 = abs(np.conjugate(amplitudes) @ amplitudes)
    # if norm2 != 1: # Is this useless? I think so (Carlos)
    #     norm2 = str(norm2)
    if state_catalog is None:
        state_catalog = {tuple(ket): allColorGraphs(ket) for ket in ket_list}
    equation_sum = []
    for coef, ket in zip(np.conjugate(amplitudes), ket_list):
        try:
            # The division with factorialProduct comes from combinatoric reasons: wikipedia.org/wiki/Multinomial_theorem
            if len(state_catalog[tuple(ket)]) == 0:
                raise KeyError
            term_sum = [f'{weightProduct(graph, imaginary)}/{factorialProduct(graph)}' for graph in state_catalog[tuple(ket)]]
            term_sum = ' + '.join(term_sum)
            # The multiplication with factorialProduct from creator operators: wikipedia.org/wiki/Creation_operators
            creation_term = factorialProduct(ket)
            if (creation_term**.5) % 1 == 0:
                creation_term = int(creation_term**.5)
            else:
                creation_term = f'{creation_term}**.5'
            equation_sum.append(f'({coef})*({creation_term})*({term_sum})')
        except KeyError:
            equation_sum.append('0')
            log.info(f'The ket {tuple(ket)} of the target state was not available in the state catalog, it was set to 0. This could be because there is no perfect matching or edgecover that produces this ket.')
    # equation_sum = ' + '.join(equation_sum).replace('0 + ','').replace(' + 0','')
    # if equation_sum == '0':
    equation_sum = ' + '.join(term for term in equation_sum if term != '0')
    if equation_sum == '':
        return '0'
    elif imaginary == False:
        return f'(({equation_sum})**2)/{norm2}'.replace('/1 +', ' +').replace('/1)', ')').replace('(1)*', '')
    else:
        return f'(abs({equation_sum})**2)/{norm2}'.replace('/1 +', ' +').replace('/1)', ')').replace('(1)*', '') 


def mixedTarget(ket_list, mixed_catalog, amplitudes=None, imaginary=False, target_dict=False):
    '''
    Given a list of target kets which were NOT traced out in the mixed catalog. It produces the 
    target equation for the different possible states which were traced.
    
    Parameters
    ----------
    ket_list : list of tuples
        List of target kets written as: [((node1, dimension1), (node2, dimension2), ...), ...].
        The kets must include ALL nodes which are NOT traced out.
    mixed_catalog : dictionary
        Dictionary of state_catalogs that produce the kets from ket_list. The catalogs are splitted by 
        the ancilla creators, which are the keys of the mixed_catalog. 
        It can be obtained from the input `state_catalog`.  
    amplitudes : list, optional
        List of the amplitudes we aim for each ket. If None, all will be the same.
    imaginary : boolean, str ('cartesian' or 'polar'), optional
        If False, the final expression uses only real weights. If 'cartesian' or 'polar', the weights 
        are complex values written in the corresponding notation.
    target_dict : boolean, optional
        If True, it returns a dictionary with target equations as values. Else, it returns one str.
    
    Returns
    -------
    str (if target_dict False)
        Non-normalized fidelity string with all the contributions from the reduced density matrix.
    dictinary of str (if target_dict True)
        Dictionary whose values are non-normalized fidelity string with all the contributions from the 
        reduced density matrix.
    
    '''
    mixed_target = dict()
    for traced_ket, catalog in mixed_catalog.items():
        full_kets = [tuple(sorted(ket+traced_ket)) for ket in ket_list]
        mixed_target[traced_ket] = targetEquation(full_kets, amplitudes=amplitudes, 
                                                  state_catalog=catalog, imaginary=imaginary)
    if target_dict:
        return mixed_target
    else:
        return '  +  '.join(target for target in mixed_target.values() if target != '0')


def buildLossString(loss_function, variables):
    loss_string = 'lambda ' + ', '.join(variables) + f': {loss_function}'
    loss_string = f'func = lambda inputs: ({loss_string})(*inputs) '
    exec(loss_string, globals())
    return func, loss_string  # we can keep the second as a security check


def stringFunction(str_function, variables):
    '''
    Turn a string into a function, given a list of variables (strings).
    
    It does the same than buildLossString.
    '''
    func_string = 'lambda ' + ', '.join(variables) + f': {str_function}'
    func_string = f'func = lambda inputs: ({func_string})(*inputs) '
    exec(func_string, globals())
    return func


# # String Expressions
# Write the string expressions used in the optimization.

def ptrace(u, keep, dims, optimize=False):
    """Calculate the partial trace of an outer product

    ρ_a = Tr_b(|u><u|)

    From slek120: https://scicomp.stackexchange.com/a/30057
    
    Parameters
    ----------
    u : array
        Vector to use for outer product
    keep : array
        An array of indices of the spaces to keep after being traced. For instance, if the space is
        A x B x C x D and we want to trace out B and D, keep = [0,2]
    dims : array
        An array of the dimensions of each space. 
        For instance, if the space is A x B x C x D, dims = [dim_A, dim_B, dim_C, dim_D]

    Returns
    -------
    ρ_a : 2D array
        Traced matrix
    """
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim + i if i in keep else i for i in range(Ndim)]
    u = u.reshape(dims)
    rho_a = np.einsum(u, idx1, u.conj(), idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)


def compute_entanglement(qstate: np.array, sys_dict: dict, var_factor=0) -> float:
    """
    calculate for a set of bipartions given in config the mean of trace[ rho_A ], where rho_A is 
    reduced density matrix of given state for the given bipartitions

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
    try:  # for checking if norm is not zero -> if so return 1 cause no ket
        qstate *= 1 / (np.linalg.norm(qstate))
    except TypeError:
        return 1

    def calc_con(mat, par):
        red = ptrace(mat, par, dimi, False)
        return np.einsum('ij,ji', red, red)  # is equivalent to trace( red**2 ) but faster

    loss_vec = [calc_con(qstate, par[0]) for par in sys_dict['bipar_for_opti']]
    lenght = len(loss_vec)
    mean = sum(loss_vec) / lenght
    if var_factor == 0:  # no need to compute variance if factor = 0
        return mean
    else:
        var = sum([(x - mean) ** 2 for x in loss_vec]) / (lenght)
        return mean + var_factor * var


def entanglement_fast(avail_states: dict, sys_dict: dict,var_factor):
    """
    compute the entanglement according to compute_entanglement() of the state given by the graph
    according to given avail_states

    Parameters
    ----------
    sys_dict : dict
        that stores essential infos of quantum system (see helpfunctions.get_sysdict).
    avail_states : dict
        storing graphs(value) for each state(key)

    Returns
    -------
    str
        returns entanglement of graph only in terms of graphs weights.

    """

    state_vector = sys_dict['dim_total'] * [0]
    num_anc = sys_dict['num_ancillas']  # count ancillas
    if num_anc != 0:
        temp_dic = dict()
        for key, val in avail_states.items():
            try:
                extsting_graph = temp_dic[key[:-num_anc]]
                extsting_graph.append(val[0])
                temp_dic[key[:-num_anc]] = extsting_graph 
            except KeyError:
                temp_dic[key[:-num_anc]] = val
        avail_states = temp_dic
    for idx, state in enumerate(sys_dict["all_states"]):
        try:
            term_sum = [weightProduct(graph, sys_dict['imaginary'])
                        for graph in avail_states[state]]
            term_sum = '+'.join(term_sum)
            state_vector[idx] = f'{(term_sum)}'
        except KeyError:
            pass

    if sys_dict['imaginary'] is False:
        return f'compute_entanglement(np.array({state_vector} ),'.replace("'", "") \
               + f' {sys_dict}, {var_factor} )'
    else:
        return f'abs(compute_entanglement(np.array({state_vector} ),'.replace("'", "") \
               + f' {sys_dict}, {var_factor} ))'
