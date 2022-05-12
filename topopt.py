import os
import sys
import numpy as np
import theseus as th


def edgeNum(pdv, uc):
    '''
    calculate number of edges from pdv
    '''
    p, d, v = (pdv)
    a = v - p
    if uc:
        return p * (p - 1) * d // 2 + p * a * d + a * (a - 1) // 2
    else:
        return p * (p - 1) * d * d // 2 + p * a * d + a * (a - 1) // 2


def stateToString(state):
    '''
    for readability, turn state array into string
    '''
    termlist = np.array(state)[:, :, 1]
    termstringlist = []
    for term in termlist:
        termstringlist.append(''.join([str(item) for item in term]))
    return '+'.join(termstringlist)


def stringToTerm(termstring):
    '''
    used by makeState. turn ket given by string into tuples representing creator operators.
    example: 
        input: "0210"
        output: ((0, 0), (1, 2), (2, 1), (3, 0))
    '''
    return tuple([tuple([i, int(col)]) for i, col in enumerate(termstring)])


def makeState(statestring):
    '''
    turn state given as string into state to be used by theseus
    example:
        input: "0000+1111+2222"
        output: [((0, 0), (1, 0), (2, 0), (3, 0)),
                 ((0, 1), (1, 1), (2, 1), (3, 1)),
                 ((0, 2), (1, 2), (2, 2), (3, 2))]
    '''
    terms = statestring.split('+')
    return [stringToTerm(term) for term in terms]


def makeGHZ(pdv):
    '''
    construct GHZ state from (p,d,v)
    '''
    state = []
    data, dim, verts = pdv
    for ii in range(dim):
        term = []
        for jj in range(verts):
            if jj < data:
                term.append((jj, ii))
            else:
                term.append((jj, 0))
        state.append(tuple(term))
    return state


def makeUnicolor(edge_list, num_nodes):
    '''
    simplify edge list by deleting all multicolor edges between data nodes
    '''
    return [edge for edge in edge_list if
            (((edge[0] not in range(num_nodes)) or (edge[1] not in range(num_nodes))) or (edge[2] == edge[3]))]


def makeEdgesFromPDV(pdv, unicolor=False):
    '''
    input vector [p,d,v]
    make edge_list for a graph with v vertices. the first p vertices are data particles with local dimensions d.
    unicolor allows multicolored edges between data vertices.
    '''
    locdim = [pdv[1]] * pdv[0] + [1] * (pdv[2] - pdv[0])  # local dimensions
    edge_list = th.buildAllEdges(locdim)  # make edge list from local dimension
    if unicolor: edge_list = makeUnicolor(edge_list, pdv[0])
    return edge_list


def defineGHZ(pdv, unicolor=False):
    '''
    returns state and starting graph for a GHZ search specified by (particles, dimension, vertices).
    '''
    state = makeGHZ(pdv)
    edge_list = makeEdgesFromPDV(pdv, unicolor=unicolor)
    return state, edge_list


def addAncillas(state, num):
    '''
    takes a state and add num ancillas to it. this way ancillas don't have to be written in state string.
    
    example:
    input: [((0, 0), (1, 0), (2, 0), (3, 0)),
                 ((0, 1), (1, 1), (2, 1), (3, 1))]
    output: [((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)),
                 ((0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1))]
    '''
    particles = len(state[0])
    for ii, ket in enumerate(state):
        tempket = list(ket)
        for jj in range(num):
            tempket.append(tuple([jj + particles, 0]))
        state[ii] = tempket
    return state


def removeConnections(edge_list, connection_list):
    '''
    removes all edges that connect certain pairs of vertices.
    
    example:
    input: edge_list, [[0,1],[3,5]]
    output: edge_list without any edges that connect 0-1 or 3-5.
    '''
    new_edge_list = edge_list
    for connection in connection_list:
        new_edge_list = [edge for edge in new_edge_list if (edge[0] != connection[0] or edge[1] != connection[1])]
    return new_edge_list


def setDeletedIndexThr(x, thr, real=True):
    '''
    returns indices of all edges that have abs(weight) smaller than a threshold.
    '''
    if real:
        delind = [ii for ii, xi in enumerate(x) if abs(xi) < thr]
    else:
        delind = [ii for ii, xi in enumerate(x[:(len(x) // 2)]) if abs(xi) < thr]
    return delind


def setDeletedIndexSingle(x, rep, real=True):
    '''
    returns the index of the edge with the rep'th smallest weight.
    '''
    if real:
        idx = np.argsort(abs(np.array(x)))
        delind = idx[rep]
    else:
        idx = np.argsort(abs(np.array(x[:len(x) // 2])))
        delind = idx[rep]
    return delind


def deleteEdges(edge_list, x, delind, real=True):
    '''
    deletes specified indices from edge list and weights.
    '''
    if (type(delind) is int) or (type(delind) is np.int64):
        delind = [delind]
    edge_list_new = [edge for ii, edge in enumerate(edge_list) if ii not in delind]
    if real:
        x_new = [elem for ii, elem in enumerate(x) if ii not in delind]
    else:
        r_cur = x[:len(edge_list)]
        r_new = [r for ii, r in enumerate(r_cur) if ii not in delind]
        th_cur = x[len(edge_list):]
        th_new = [th for ii, th in enumerate(th_cur) if ii not in delind]
        x_new = np.array(r_new + th_new)

    return edge_list_new, x_new


def makeLossString(state, edge_list, mode="cr", real=True):
    '''
    define loss as lambda function from target state and available edges. this is done by writing its definition out as string and executing it, without needing sympy.
    '''
    cat = th.stateCatalog(th.findPerfectMatchings(edge_list))

    target = th.targetEquation(state, avail_states=cat, real=real)
    norm = th.Norm.fromDictionary(cat, real=real)
    if real:
        variables = ["w_{}_{}_{}_{}".format(*edge) for edge in edge_list]
    else:
        variables = ["r_{}_{}_{}_{}".format(*edge) for edge in edge_list] + ["th_{}_{}_{}_{}".format(*edge) for edge in
                                                                             edge_list]

    if mode == "cr":
        lambdaloss = "".join(["1-", target, "/(1+", norm, ")"])
        func, lossstring = th.buildLossString(lambdaloss, variables)
    if mode == "fid":
        lambdaloss = "".join(["1-", target, "/(0+", norm, ")"])
        func, lossstring = th.buildLossString(lambdaloss, variables)

    return func, lossstring


def prepOptimizer(numweights, x=[], real=True):
    '''
    returns initial values and bounds for use in optimization.
    '''
    if real == True:
        bounds = numweights * [(-1, 1)]
        if len(x) == 0:
            initial_values = 2 * np.random.random(numweights) - 1
        else:
            initial_values = x
    else:
        bounds = numweights * [(-1, 1)] + numweights * [(-np.pi, np.pi)]
        if len(x) == 0:
            rands_r = 2 * np.random.random(numweights) - 1
            rands_th = 2 * np.pi * np.random.random(numweights) - np.pi
            initial_values = np.concatenate([rands_r, rands_th])
        else:
            initial_values = x

    return initial_values, bounds


def txtAppend(file_name, record):
    '''
    writes record into file.
    creates data directory if it does not exist yet.
    '''
    try:
        os.makedirs(os.path.join(sys.path[0], 'data'))
    except FileExistsError:
        pass

    try:
        with open(os.path.join(sys.path[0], 'data', file_name), 'a') as f:
            f.write(str(record))
            f.write('\n')
    except:
        print('write error.')


def writeSol(edge_list, x, pdv, fid, real=True):
    '''
    write solution to txt file.
    '''
    pdvstr = '(' + str(pdv[0]) + '-' + str(pdv[1]) + '-' + str(pdv[2]) + ')'
    if not real:
        pdvstr = 'c' + pdvstr
    if all(abs(x[:len(edge_list)]) > 0.95):
        clean = 'clean'
    else:
        clean = 'rough'
    edgenum = str(len(edge_list))
    pmnum = str(len(th.findPerfectMatchings(edge_list)))
    fidelity = str(round(float(1 - fid(x)), 2))

    if real:
        combined_data = str(list(map(list, zip(*[edge_list, x]))))
    else:
        x = list(zip(x[:len(edge_list)], x[len(edge_list):]))
        combined_data = str(list(map(list, zip(*[edge_list, x]))))
    txtAppend(pdvstr + '-' + clean + '-' + edgenum + '-' + pmnum + '-' + fidelity + '.txt', combined_data)
