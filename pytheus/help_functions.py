import itertools
import numpy as np


def flatten_lists(the_lists):
    """
    takes a list as argument return flatten one

    """
    result = []
    for _list in the_lists:
        result += _list
    return result


def stateToString(state, ket=False):
    '''
    for readability, turn state array into string
    '''
    termlist = np.array(state)[:, :, 1]
    termstringlist = []
    for term in termlist:
        termstringlist.append(''.join([str(item) for item in term]))
    if ket:
        termstringlist = ['|' + term + '>' for term in termstringlist]
    return '+'.join(termstringlist)


def readableState(state):
    readable_dict = {}
    for key in state.kets:
        readable_dict[stateToString([key], ket=True)] = state[key]
    return readable_dict


def stringToTerm(termstring):
    '''
    used by makeState. turn ket given by string into tuples representing creator operators.
    example: 
        input: "0210"
        output: ((0, 0), (1, 2), (2, 1), (3, 0))
    '''
    return tuple([tuple([i, int(col)]) for i, col in enumerate(termstring)])


def makeState(statestring: str) -> list:
    '''
    turn state given as string into state to be used by pytheus
    example:
        input: "0000+1111+2222"
        output: [((0, 0), (1, 0), (2, 0), (3, 0)),
                 ((0, 1), (1, 1), (2, 1), (3, 1)),
                 ((0, 2), (1, 2), (2, 2), (3, 2))]
    '''
    terms = statestring.split('+')
    return [stringToTerm(filter(str.isdigit, term)) for term in terms]


def makeUnicolor(edge_list, num_nodes):
    '''
    simplify edge list by deleting all multicolor edges between data nodes
    '''
    return [edge for edge in edge_list if
            (((edge[0] not in range(num_nodes)) or (edge[1] not in range(num_nodes))) or (edge[2] == edge[3]))]


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


def prepEdgeList(edge_list, cnfg):
    """
    Restrict starting graph as given by config.
    """
    try:
        if cnfg['unicolor']:
            num_data_nodes = len(cnfg['target_state'][0])
            edge_list = makeUnicolor(edge_list, num_data_nodes)
    except KeyError:
        pass

    removed_connections = []
    disjoint_nodes = []
    try:
        disjoint_nodes += cnfg['in_nodes']
    except KeyError:
        pass
    try:
        disjoint_nodes += cnfg['single_emitters']
    except KeyError:
        pass
    removed_connections += list(itertools.combinations(disjoint_nodes,2))
    try:
         removed_connections += cnfg['removed_connections']
    except KeyError:
        pass
    edge_list = removeConnections(edge_list,removed_connections)
    return edge_list


def get_all_bi_partions(num_par: int, lenght=None):

    """
    returns all bi-partions as a generator for a given number of particles:

        e.g. : num_par = 3 : [([0], [1, 2]), ([1], [0, 2]), ([2], [0, 1])]

    """
    if lenght is None or lenght == 'all':
        def check_len(bipar):
            return True
    else:
        assert (type(lenght) is int and int(num_par / 2) >= lenght), \
            "invalid lenght given(or Typeerror): int(num_par/2) > given lenght"

        def check_len(bipar):
            return len(bipar) == lenght

    S = {i for i in range(num_par)}
    doubles = []
    for ll in range(1, int(len(S) / 2) + 1):
        combinations = set(itertools.combinations(S, ll))
        for oneC in combinations:
            if sorted(list(oneC)) not in doubles:
                bipar = sorted(list(oneC))
                if check_len(bipar):
                    yield (bipar, sorted(list(S - set(oneC))))
            doubles.append(sorted(list(S - set(oneC))))


def get_all_kets_for_given_dim(dimension: list, type_return=int):
    """
    get all possible kets for given dimension  (1=ancilla)
    e.g: input = [2,2,1,1] -> [0, 1, 10, 11]  for int 
                       -> ['00', '01', '10', '11'] for str
    """
    if dimension.count(1) != 0:
        dimension = dimension[:-dimension.count(1)]
    if type_return == int:
        return list([int("".join(map(str, x))) for x in
                     itertools.product(*[[t for t in range(i)]
                                         for i in dimension])])
    if type_return == str:
        return list(["".join(map(str, x)) for x in itertools.product(
            *[[t for t in range(i)] for i in dimension])])


def get_complement(ls: tuple, whole_list: list):
    return tuple(set([x for x in whole_list]) - set(ls))


def get_sysdict(dimensions_of_H: list, bipar_for_opti='all', imaginary = False):
    """
    a dict to store
        - number of particles: sysdict['num_particles']
        - all possible bipartions: sysdict['all_biparations']
        - all dimension for each particle: sysdict['dimensions']
        - how many solution should be produced: sysdict["samples"]
    """
    sysdict = dict()
    sysdict['dimensions'] = dimensions_of_H
    sysdict['num_ancillas'] = dimensions_of_H.count(1)
    sysdict['num_particles'] = len(dimensions_of_H) - sysdict['num_ancillas']
    sysdict['all_states'] = [tuple([(idx, int(ket[idx])) for idx in
                                    range(sysdict['num_particles'])])
                             for ket in
                             get_all_kets_for_given_dim(dimensions_of_H, str)]

    sysdict['dim_total'] = np.product(dimensions_of_H)
    sysdict['bipar_for_opti'] = list(
        get_all_bi_partions(sysdict['num_particles'], bipar_for_opti))
    sysdict['imaginary'] = imaginary
    return sysdict
