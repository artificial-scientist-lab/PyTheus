import numpy as np
import pytheus.theseus as th
from copy import deepcopy

# +
DEFAULT_NORM = 'Not stored yet. You have to run getNorm().'
DEFAULT_STATE = 'Not stored yet. You have to run getState().'
WRONG_IMAGINARY = 'The property `imaginary` is NOT defined correctly.'


def invalidInput(input_name):
    return f'Introduce a valid input `{input_name}`.'


def defaultValues(length, imaginary):
    if imaginary in [False, 'cartesian']:
        return [True] * length
    elif imaginary == 'polar':
        return [(True, False)] * length
    else:
        raise ValueError(invalidInput('imaginary'))


# -

# # The Graph class

class Graph():  # should this be an overpowered dictionary? NOPE
    def __init__(self,
                 edges,  # list/tuple of edges, dictionary with weights, 'full' or 'empty'
                 dimensions=[],
                 weights=[],  # list of values or tuples encoding imaginary values
                 imaginary=False,  # 'cartesian' or 'polar'
                 norm=False,  # For the sake of perfomance, compute
                 state=False,  # norm and state only when needed.
                 state_cat=True
                 ):
        self.dimensions = dimensions
        self.imaginary = imaginary
        self.full = True if edges == 'full' else False
        # The next line may redefine previous properties
        self.graph = self.graphStarter(edges, weights)  # MAIN PROPERTY
        # This may not be elegant, but it works
        self.state_catalog = None
        if self.state_catalog == None and state_cat: self.getStateCatalog()
        self.norm = DEFAULT_NORM
        if norm: self.getNorm()
        self.state = DEFAULT_STATE
        if state: self.getState()

    # Long, cumbersome, and very necessary function.
    def graphStarter(self, edges, weights):
        '''
        To facilitate the use of the library, this function transforms different kinds of graph
        inputs into a suitable Graph instance, redefining the properties if needed.
        
        Once the edges and weights are correct, it returns an property graph.
        '''
        if edges == 'full':  # If True, dimensions must be defined.
            if len(self.dimensions) == 0:
                raise ValueError('Introduce the dimensions to get a fully connected graph')
            else:
                edges = th.buildAllEdges(self.dimensions)
                if len(weights) == 0:
                    weights = defaultValues(len(edges), self.imaginary)
                else:
                    raise ValueError('The input `full` does not allow to specify the weights.')
        # If the introduced edges are a list/tuple, they may include the weights           
        elif type(edges) in [list, tuple]:
            edge_shape = np.shape(edges)
            if edge_shape[1] == 4:
                pass  # seems legit
            elif edge_shape[1] == 5:  # The weights will be the last term of each item
                weights = [ed[-1] for ed in edges]
                edges = [ed[:4] for ed in edges]
            elif edge_shape[1] == 6:  # The weights will be the last 2 terms of each item
                weights = [tuple(ed[-2], ed[-1]) for ed in edges]
                edges = [ed[:4] for ed in edges]
            else:
                raise ValueError(invalidInput('edges'))
        elif type(edges) == dict:
            weights = [edges[key] for key in sorted(edges.keys())]
            edges = [kk if isinstance(kk, tuple) else eval(kk) for kk in sorted(edges.keys())]

            # Verification of appropiate edges
            if all(isinstance(edge, tuple) for edge in edges):
                if all(len(edge) == 4 for edge in edges):
                    pass  # seems legit
                else:
                    raise ValueError(invalidInput('edges'))
            else:
                raise ValueError(invalidInput('edges'))
        else:
            raise ValueError(invalidInput('edges'))

        if len(weights) == 0:  # The default option True behaves (mostly) as 1
            weights = defaultValues(len(edges), self.imaginary)
        else:
            weight_shape = np.shape(weights)
            if weight_shape[0] != len(edges):
                raise ValueError('The number of edges and weights do not match.')
            elif len(weight_shape) == 1:
                if any(isinstance(val, complex) for val in weights):
                    self.imaginary = 'cartesian'
            elif len(weight_shape) == 2:
                if weight_shape[1] == 2:
                    if self.imaginary == 'cartesian':
                        weights = [val[0] + 1j * val[1] for val in weights]
                    elif self.imaginary == 'polar':
                        weights = [tuple(val) for val in weights]
                    else:
                        raise ValueError(invalidInput('imaginary'))
                else:
                    raise ValueError('Introduce valid weights.')
            else:
                raise ValueError('Introduce valid weights.')

        if len(self.dimensions) == 0:
            self.dimensions = th.graphDimensions(edges)  # a nice function should be defined here
        # Once edges and weights are properly defined, we return a graph
        return {ed: val for ed, val in zip(edges, weights)}

    def __repr__(self):
        '''
        What you see when using print() or calling the instance.
        '''
        return '{' + ',\n '.join(f'{kk}: {vv}' for kk, vv in self.graph.items()) + '}'

    def __len__(self):
        return len(self.graph)

    def __eq__(self, other):
        return self.graph == other.graph

    def __getitem__(self, edge):
        return self.graph[tuple(edge)]

    def __setitem__(self, edge, weight):
        if isinstance(edge, (tuple, list)) and len(edge) == 4:
            if isinstance(weight, (int, float, complex)):
                if self.imaginary == 'polar':
                    self.graph[tuple(edge)] = (abs(weight), np.angle(weight))
                    print('Weight stored in polar notation.')
                else:
                    self.graph[tuple(edge)] = weight
                    if type(weight) == complex:
                        self.imaginary = 'cartesian'
            elif isinstance(weight, (tuple, list)) and len(weight) == 2:
                if self.imaginary == 'polar':
                    self.graph[edge] = tuple(weight)
                else:
                    self.graph[edge] = weight[0] * np.exp(1j * weight[1])
                    self.imaginary = 'cartesian'
                    print('Weight stored in cartesian notation.')
            else:
                raise ValueError(invalidInput('weight'))
        else:
            raise ValueError(invalidInput('edge'))

    # The method __round__ for the class State is almost the same, could it be reformat?
    def __round__(self, ndigits=0):
        if self.imaginary == False:
            for edge in self.edges:
                self[edge] = round(self[edge], ndigits)
        elif self.imaginary == 'cartesian':
            for edge in self.edges:
                self[edge] = round(self[edge].real, ndigits) + 1j * round(self[edge].imag, ndigits)
        elif self.imaginary == 'polar':
            for edge in self.edges:
                self[edge] = (round(self[edge][0], ndigits), self[edge][1])
        else:
            raise ValueError(WRONG_IMAGINARY)
        return self

    # this should be __del__ but then you would have to do: del self.graph[] etc
    def remove(self, edge, update=True):
        del self.graph[edge]
        if update:
            remove_ket_list = []
            for ket, pm_list in self.state_catalog.items():
                if ((edge[0], edge[2]) and (edge[1], edge[3])) in ket:
                    self.state_catalog[ket] = [pm for pm in pm_list if edge not in pm]
                    if len(self.state_catalog[ket]) == 0:
                        remove_ket_list.append(ket)
            for ket in remove_ket_list:
                del self.state_catalog[ket]
        # if update:
        #    if self.norm != DEFAULT_NORM: self.getNorm()
        #    if self.state != DEFAULT_STATE: self.getState()

    def purge(self, threshold=1e-4, update=True):
        '''
        It removes all edges whose weights, in absolute value, are below `threshold`.
        '''
        remove_edges = []
        if self.imaginary == 'polar':
            for edge, weight in self.graph.items():
                if abs(weight[0]) < threshold:
                    remove_edges.append(edge)
        else:
            for edge, weight in self.graph.items():
                if abs(weight) < threshold:
                    remove_edges.append(edge)
        for edge in remove_edges:
            del self.graph[edge]
        if update:
            remove_ket_list = []
            for ket, pm_list in self.state_catalog.items():
                for edge in remove_edges:
                    if ((edge[0], edge[2]) and (edge[1], edge[3])) in ket:
                        self.state_catalog[ket] = [pm for pm in pm_list if edge not in pm]
            for ket, pm_list in self.state_catalog.items():
                if len(pm_list) == 0:
                    remove_ket_list.append(ket)
            for ket in remove_ket_list:
                del self.state_catalog[ket]

    # DEFINE GET, SET AND DEL ITEM

    @property
    def edges(self):
        return list(self.graph.keys())

    @property
    def weights(self):
        return list(self.graph.values())

    @property
    def num_nodes(self):
        return len(self.dimensions)

    @property
    def perfect_matchings(self):
        return sum(self.state_catalog.values(), [])

    @property
    def is_weighted(self):
        if all(isinstance(val, tuple) for val in self.weights):
            return not all(isinstance(val, bool) for val in sum(self.weights, ()))
        elif all(isinstance(val, bool) for val in self.weights):
            return False
        elif all(isinstance(val, (int, float, complex)) for val in self.weights):
            return True
        else:
            raise ValueError('The weights are NOT defined correctly.')

    # imaginary could be redefined as one of these properties

    def copy(self):
        return deepcopy(self)

    def addEdge(self, edge, weight=None, update=True):
        if len(edge) == 4 and all(isinstance(val, int) for val in edge):
            edge = tuple(edge)
        else:
            raise ValueError(invalidInput('edge'))
        if edge in self.edges:
            print('The edge is going to be redefined.')
            update = False  # there's no need to update the state_catalog
        else:  # the edge is included
            self.graph[edge] = True
        if weight == None:
            self[edge] = defaultValues(1, self.imaginary)[0]
        else:
            self[edge] = weight
        if update:
            subgraph = th.removeNodes(edge[:2], self.edges)
            new_states = th.stateCatalog(th.findPerfectMatchings(subgraph + [edge]))
            for ket, perfect_matchings in new_states.items():
                try:
                    self.state_catalog[ket] += perfect_matchings
                except KeyError:
                    self.state_catalog[ket] = perfect_matchings
                self.state_catalog[ket] = sorted(self.state_catalog[ket])
        #     if self.norm != DEFAULT_NORM: self.getNorm()
        # if self.state != DEFAULT_STATE: self.getState()

    # This could be also a property, but then we cannot introduce arguments
    def node_degrees(self, ascending=False):  #
        '''
        Degree of each node of the graph.
        '''
        return th.nodeDegrees(self.edges, rising=ascending)

    def getStateCatalog(self):
        if self.full:
            self.state_catalog = th.allPerfectMatchings(self.dimensions)
        else:
            self.state_catalog = th.stateCatalog(th.findPerfectMatchings(self.edges))

    # the employed norm function is simplified (only pm) and uses the new imaginary notation
    def getNorm(self):
        self.norm = th.writeNorm(self.state_catalog, imaginary=self.imaginary)

    def getState(self):
        kets = list(self.state_catalog.keys())
        if self.is_weighted:
            amplitudes = []
            conversion = self.imaginary == 'polar'
            if conversion:
                self.toCartesian()
            for kt in kets:
                term = 0
                for pm in self.state_catalog[kt]:
                    term += np.prod([self.graph[edge] for edge in pm])
                amplitudes.append(term)
            if conversion:
                self.toPolar()
        else:
            amplitudes = []
        self.state = State(kets, amplitudes, self.imaginary)

        # This could also be defined as __abs__, but what do you give back? The dictionary?

    def __abs__(self):
        return_weights = len(self.graph) * [0]
        if self.is_weighted:
            if self.imaginary in [False, 'cartesian']:
                for idx, vv in enumerate(self.graph.values()):
                    return_weights[idx] = abs(vv)
            elif self.imaginary == 'polar':
                for idx, vv in enumerate(self.graph.values()):
                    return_weights[idx] = abs(vv[0])
            else:
                raise ValueError(WRONG_IMAGINARY)
            return return_weights
        else:
            ValueError('emtpy weights')

    def absolute(self):
        if self.is_weighted:
            if self.imaginary in [False, 'cartesian']:
                for kk, vv in self.graph.items():
                    self.graph[kk] = abs(vv)
            elif self.imaginary == 'polar':
                for kk, vv in self.graph.items():
                    self.graph[kk] = abs(vv[0])
            else:
                raise ValueError(WRONG_IMAGINARY)
        else:
            for kk, vv in self.graph.items():
                self.graph[kk] = True
        self.imaginary = False

    def toCartesian(self):
        if self.imaginary == 'cartesian':
            print('The weights are already in cartesian coordinates.')
            return None
        if self.is_weighted:
            if self.imaginary == False:
                for kk, vv in self.graph.items():
                    self.graph[kk] = (vv, 0)
            elif self.imaginary == 'polar':
                for kk, vv in self.graph.items():
                    self.graph[kk] = vv[0] * np.exp(1j * vv[1])
            else:
                raise ValueError('The propery `imaginary` is NOT defined correctly.')
        else:
            for kk, vv in self.graph.items():
                self.graph[kk] = True
        self.imaginary = 'cartesian'

    def toPolar(self):
        if self.imaginary == 'polar':
            print('The weights are already in polar coordinates.')
            return None
        if self.is_weighted:
            if self.imaginary == False:
                for kk, vv in self.graph.items():
                    self.graph[kk] = (vv, 0)
            elif self.imaginary == 'cartesian':
                for kk, vv in self.graph.items():
                    self.graph[kk] = (abs(vv), np.angle(vv))
            else:
                raise ValueError('The propery `imaginary` is NOT defined correctly.')
        else:
            for kk, vv in self.graph.items():
                self.graph[kk] = (True, False)
        self.imaginary = 'polar'

    def rescale(self, constant):
        conversion = self.imaginary == 'polar'
        if conversion:
            self.toCartesian()
        for edge in self.graph.keys():
            self.graph[edge] *= constant
        if conversion:
            self.toPolar()

    def minimum(self, *args):
        '''
        returns the key of the edge with the smallest weight, e.g:
            graph.minimum() smallest edge
            graph.minimum(-1) biggest edge
            graph.minimum(0,5) first 5 smallest edges 
        '''
        if len(args) == 0:
            n_th_smallest = 0  # takes smallest
        elif len(args) == 1:
            n_th_smallest = args[0]  # takes nth given
        else:
            n_th_smallest = slice(*args)

        if self.imaginary in [False, 'cartesian']:
            idx = np.argsort(abs(np.array(self.weights)))
        elif self.imaginary == 'polar':
            idx = np.argsort(abs(np.array([rr[0] for rr in self.weights])))
        # now check if input is valid
        try:
            delind = idx[n_th_smallest]
        except IndexError:
            lenght_graph = len(self.graph)
            max_given_n = max(args)
            raise ValueError(
                f'Given n_th is to large (n starts 0): {max_given_n+1=} > {lenght_graph=}')

        if type(delind) is np.int64:  # makes sure that we can iterate by return
            return self.edges[delind]

        return [self.edges[ii] for ii in delind]

    def clamp(self, maximum=1, minimum=None):  # , rescale=False):
        if maximum >= 0:
            if minimum == None:
                minimum = - maximum
            for edge, weight in self.graph.items():
                self.graph[edge] = max(minimum, min(weight, maximum))
        else:
            raise ValueError('Introduce a positive maximum.')


# # The State class

class State():

    def __init__(self,
                 kets,
                 amplitudes=[],  # list of values or tuples encoding imaginary values
                 imaginary=False,  # 'cartesian' or 'polar'
                 normalize=False,
                 ):

        self.imaginary = imaginary
        self.state = self.stateStarter(kets, amplitudes)  # This may redefine previous properties
        if normalize: self.normalize()

    def stateStarter(self, kets, amplitudes):
        '''
        Function to initiate a State instance with different inputs.
        This version is not so flexible with the input format as the analogous from Graph.
        '''
        if type(kets) == dict:
            amplitudes = [kets[key] for key in sorted(kets.keys())]
            kets = sorted(kets.keys())
        # Here, introducing the amplitudes after the kets in a list/tuple doesn't work
        elif type(kets) in [list, tuple]:
            pass  # seems legit
        else:
            raise ValueError(invalidInput('kets'))

        # Verification of appropiate kets            
        kets_shape = np.shape(kets)
        if all(isinstance(kt, str) for kt in kets):
            kets = [tuple((ii, int(dim)) for ii, dim in enumerate(kt)) for kt in kets]
        elif kets_shape[2] == 2:
            pass  # The third component must have 2 dimensions: (node, dim)
        else:
            raise ValueError('Introduce valid input `kets`.')

        # Verification and setting of appropiate amplitudes
        if len(amplitudes) == 0:  # The default option True behaves (mostly) as 1
            amplitudes = defaultValues(len(kets), self.imaginary)
        else:
            amplitude_shape = np.shape(amplitudes)
            if amplitude_shape[0] != len(kets):
                raise ValueError('The number of kets and amplitudes do not match.')
            elif len(amplitude_shape) == 1:
                if any(isinstance(val, complex) for val in amplitudes):
                    self.imaginary = 'cartesian'
            elif len(amplitude_shape) == 2:
                if amplitude_shape[1] == 2:
                    if self.imaginary == 'cartesian':
                        amplitudes = [val[0] + 1j * val[1] for val in amplitudes]
                    elif self.imaginary == 'polar':
                        amplitudes = [tuple([val[0], val[1]]) for val in amplitudes]
                    else:
                        raise ValueError(invalidInput('imaginary'))
                else:
                    raise ValueError('Introduce valid amplitudes.')
            else:
                raise ValueError('Introduce valid amplitudes.')

        # Once kets and amplitudes are properly defined, we return a state
        return {kt: val for kt, val in zip(kets, amplitudes)}

    def __repr__(self):
        '''
        What you see when using print() or calling the instance.
        '''
        return '{' + ',\n '.join(f'{kk}: {vv}' for kk, vv in self.state.items()) + '}'

    def __len__(self):
        return len(self.state)

    # def __eq__(self, other):
    # Options for establishing equivalence:
    # - Direct comparison of dictionaries: count kets as existing even if amplitude 0
    # - Using the inner product, but a@b=-.9999, is it equal? threshold needed 

    def __getitem__(self, ket):
        if type(ket) == str:
            ket = tuple((ii, int(dim)) for ii, dim in enumerate(ket))
        return self.state[ket]

    def __setitem__(self, ket, amplitude):
        if isinstance(ket, str):
            ket = tuple((ii, int(dim)) for ii, dim in enumerate(ket))
        elif isinstance(ket, (tuple, list)):
            pass  # seems legit
        else:
            raise ValueError(invalidInput('ket'))

        if isinstance(amplitude, (int, float, complex)):
            if self.imaginary == 'polar':
                self.state[tuple(ket)] = (abs(amplitude), np.angle(amplitude))
                print('Amplitude stored in polar notation.')
            else:
                self.state[tuple(ket)] = amplitude
                if type(amplitude) == complex:
                    self.imaginary = 'cartesian'
        elif isinstance(amplitude, (tuple, list)) and len(amplitude) == 2:
            if self.imaginary == 'polar':
                self.state[ket] = tuple(amplitude)
            else:
                self.state[ket] = amplitude[0] * np.exp(1j * amplitude[1])
                self.imaginary = 'cartesian'
                print('Amplitude stored in cartesian notation.')
        else:
            raise ValueError(invalidInput('amplitude'))

    def __matmul__(self, other):
        '''
        Braket operation, the inner product between state. 
        It can be used with @.
        '''
        conversion_a = self.imaginary == 'polar'
        if conversion_a:
            self.toCartesian()
        conversion_b = other.imaginary == 'polar'
        if conversion_b:
            other.toCartesian()
        amplitudes_a = []
        amplitudes_b = []
        for ket in set(self.kets + other.kets):
            try:
                amplitudes_a.append(self.state[ket])
            except KeyError:
                amplitudes_a.append(0)
            try:
                amplitudes_b.append(other.state[ket])
            except KeyError:
                amplitudes_b.append(0)
        if conversion_a:
            self.toPolar()
        if conversion_b:
            other.toPolar()
        # turning into array the second term is redundant, but somehow faster
        return np.conjugate(amplitudes_a) @ np.array(amplitudes_b)

    # The method __round__ for the class State is almost the same, could it be reformat?
    def __round__(self, ndigits=0):
        if self.imaginary == False:
            for ket in self.kets:
                self[ket] = round(self[ket], ndigits)
        elif self.imaginary == 'cartesian':
            for ket in self.kets:
                self[ket] = round(self[ket].real, ndigits) + 1j * round(self[ket].imag, ndigits)
        elif self.imaginary == 'polar':
            for ket in self.kets:
                self[ket] = (round(self[ket][0], ndigits), self[ket][1])
        else:
            raise ValueError(WRONG_IMAGINARY)
        return self

    def purge(self, threshold=1e-4):
        '''
        It removes all kets whose amplitudes, in absolute value, are below `threshold`.
        '''
        remove_kets = []
        if self.imaginary == 'polar':
            for ket, amplitude in self.state.items():
                if abs(amplitude[0]) < threshold:
                    remove_kets.append(ket)
        else:
            for ket, amplitude in self.state.items():
                if abs(amplitude) < threshold:
                    remove_kets.append(ket)
        for ket in remove_kets:
            del self.state[ket]

    @property
    def kets(self):
        return list(self.state.keys())

    @property
    def amplitudes(self):
        return list(self.state.values())

    @property
    def norm(self):
        return (self @ self) ** 0.5

    @property
    def is_weighted(self):
        if all(isinstance(val, tuple) for val in self.amplitudes):
            return not all(isinstance(val, bool) for val in sum(self.amplitudes, ()))
        elif all(isinstance(val, bool) for val in self.amplitudes):
            return False
        elif all(isinstance(val, (int, float, complex)) for val in self.amplitudes):
            return True
        else:
            raise ValueError('The amplitudes are NOT defined correctly.')

    # imaginary could be redefined as one of these properties

    def copy(self):
        return deepcopy(self)

    def addKet(self, ket, amplitude=None):
        if isinstance(ket, str):
            ket = tuple((ii, int(dim)) for ii, dim in enumerate(ket))
        elif isinstance(ket, (tuple, list)):
            pass  # seems legit
        else:
            raise ValueError('Invalid ket.')
        if ket in self.kets():
            print('The ket is going to be redefined.')
        else:  # the ket is included
            self.state[ket] = True
        if amplitude == None:
            self[ket] = defaultValues(1, self.imaginary)[0]
        else:
            self[ket] = amplitude

    def toCartesian(self):
        if self.imaginary == 'cartesian':
            print('The amplitudes are already in cartesian coordinates.')
            return None
        if self.is_weighted:
            if self.imaginary == False:
                for kk, vv in self.state.items():
                    self.state[kk] = (vv, 0)
            elif self.imaginary == 'polar':
                for kk, vv in self.state.items():
                    self.state[kk] = vv[0] * np.exp(1j * vv[1])
            else:
                raise ValueError('The propery `imaginary` is NOT defined correctly.')
        else:
            for kk, vv in self.state.items():
                self.state[kk] = True
        self.imaginary = 'cartesian'

    def toPolar(self):
        if self.imaginary == 'polar':
            print('The amplitudes are already in polar coordinates.')
            return None
        if self.is_weighted:
            if self.imaginary == False:
                for kk, vv in self.state.items():
                    self.state[kk] = (vv, 0)
            elif self.imaginary == 'cartesian':
                for kk, vv in self.state.items():
                    self.state[kk] = (abs(vv), np.angle(vv))
            else:
                raise ValueError('The propery `imaginary` is NOT defined correctly.')
        else:
            for kk, vv in self.state.items():
                self.state[kk] = (True, False)
        self.imaginary = 'polar'

    def rescale(self, constant):
        conversion = self.imaginary == 'polar'
        if conversion:
            self.toCartesian()
        for ket in self.state.keys():
            self.state[ket] *= constant
        if conversion:
            self.toPolar()

    def clamp(self, maximum=1, minimum=None):  # , rescale=False):
        if maximum >= 0:
            if minimum == None:
                minimum = - maximum
            for ket, amplitude in self.graph.items():
                self.state[ket] = max(minimum, min(amplitude, maximum))
        else:
            raise ValueError('Introduce a positive maximum.')

    def normalize(self):
        self.rescale(constant=1 / self.norm)

    def targetEquation(self, state_catalog=None, imaginary=None):
        if imaginary is None:
            return th.targetEquation(self.kets, amplitudes=self.amplitudes, state_catalog=state_catalog,
                                     imaginary=self.imaginary)
        else:
            return th.targetEquation(self.kets, amplitudes=self.amplitudes, state_catalog=state_catalog,
                                     imaginary=imaginary)
