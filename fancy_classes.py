import numpy as np
import theseus as th

# +
DEFAULT_NORM = 'Not stored yet. You have to run getNorm().'
DEFAULT_STATE = 'Not stored yet. You have to run getState().'

def defaultValues(length, imaginary):
    if imaginary in [False, 'cartesian']:
        return [True]*length
    elif imaginary == 'polar':
        return [(True,False)]*length
    else:
        raise ValueError('Introduce a valid input `imaginary`: `cartesian` or `polar`.')


# -

# # The Graph class

class Graph(): # should this be an overpowered dictionary? NOPE     
    def __init__(self,
                 edges, # list/tuple of edges, dictionary with weights, 'full' or 'empty'
                 dimensions = None,
                 weights = None, # list of values or tuples encoding imaginary values
                 imaginary = False, # 'cartesian' or 'polar'
                 norm = False,  # For the sake of perfomance, compute
                 state = False, # norm and state only when needed.
                ):
        self.dimensions = dimensions
        self.imaginary = imaginary
        self.full = True if edges=='full' else False
        # The next line may redefine previous properties
        self.graph = self.graphStarter(edges, weights) # MAIN PROPERTY
        # This may not be elegant, but it works
        self.state_catalog = self.getStateCatalog()
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
        if edges == 'full': # If True, dimensions must be defined.
            if self.dimensions == None:
                raise ValueError('Introduce the dimensions to get a fully connected graph')
            else:
                edges = th.buildAllEdges(self.dimensions)
                if weights == None:
                    weights = defaultValues(len(edges), self.imaginary)
                else:
                    raise ValueError('The input `full` does not allow to specify the weights.')
        # If the introduced edges are a list/tuple, they may include the weights           
        elif type(edges) in [list,tuple]:
            edge_shape = np.shape(edges)
            if edge_shape[1] == 4:
                pass # seems legit 
            elif edge_shape[1] == 5: # The weights will be the last term of each item
                edges = sorted(edges)
                weights = [ed[-1] for ed in edges]
            elif edge_shape[1] == 6: # The weights will be the last 2 terms of each item
                edges = sorted(edges)
                weights = [tuple(ed[-2], ed[-1]) for ed in edges]
            else:
                raise ValueError('Introduce a valid input `edges`.')
        elif type(edges) == dict:
            weights = [edges[key] for key in sorted(edges.keys())]
            edges = sorted( edges.keys() )
            # Verification of appropiate edges
            if all(isinstance(edge, tuple) for edge in edges):
                if all(len(edge)==4 for edge in edges):
                    pass # seems legit
                else: 
                    raise ValueError('Introduce a valid input `edges`.')
            else: 
                raise ValueError('Introduce a valid input `edges`.')
        else:
            raise ValueError('Introduce a valid input `edges`.')
            
        if weights == None: # The default option True behaves (mostly) as 1
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
                        weights = [val[0] + 1j*val[1] for val in weights]
                    elif self.imaginary == 'polar':
                        weights = [tuple(val[0], val[1]) for val in weights]
                    else:
                        raise ValueError('Introduce a valid input `imaginary`.') 
                else:
                    raise ValueError('Introduce valid weights.')
            else:
                raise ValueError('Introduce valid weights.')
                
        if self.dimensions == None:
            self.dimensions = th.graphDimensions(edges) # a nice function should be defined here
        # Once edges and weights are properly defined, we return a graph
        return {ed:val for ed,val in zip(edges,weights)}
    
    def __str__(self):
        '''
        What you see when using print().
        '''
        return '{' + ',\n '.join(f'{kk}: {vv}' for kk,vv in self.graph.items()) + '}'
    
    def __len__(self):
        return len(self.graph)
    
    def __eq__(self, other):
        return self.graph == other.graph
    
    def __getitem__(self, edge):
        return self.graph[tuple(edge)]
    
    def __setitem__(self, edge, weight, imaginary=False):
        if isinstance(edge, (tuple,list)) and len(edge)==4:
            if isinstance(weight, (int,float,complex)):
                if self.imaginary == 'polar':
                    self.graph[tuple(edge)] = (abs(weight), np.angle(weight))
                else:
                    self.graph[tuple(edge)] = weight
                    if type(weight) == complex:
                        self.imaginary = 'cartesian'
            # This may lead to error, the text may help as a warning
            elif isinstance(weight, (tuple,list)) and len(weight)==2:                
                if imaginary == 'cartesian':
                    weight = weight[0] + 1j*weight[1]
                    if self.imaginary == 'polar':
                        self.graph[edge] = (abs(weight), np.angle(weight))
                    else:
                        self.graph[edge] = weight
                        self.imaginary = 'cartesian'
                elif imaginary == 'polar':
                    if self.imaginary == 'polar':
                        self.graph[edge] = tuple(weight)
                    else:
                        self.graph[edge] = weight[0] * np.exp(1j*weight[1])
                        self.imaginary = 'cartesian'
                else:
                    raise ValueError('Introduce a valid input `imaginary`: cartesian or polar.')  
            else:
                raise ValueError('Invalid weight.')
        else:
            raise ValueError('Invalid edge.')
    
    # this should be __del__ but then you would have to do: del self.graph[] etc
    def remove(self, edge, update=True):
        del self.graph[edge]
        if self.state_catalog != DEFAULT_CATALOG:
            remove_ket_list = []
            for ket, pm_list in self.state_catalog.items():
                if ((edge[0], edge[2]) and (edge[1], edge[3])) in ket:
                    self.state_catalog[ket] = [pm for pm in pm_list if edge not in pm]
                    if update and len(self.state_catalog[ket]) == 0:
                        remove_ket_list.append(ket)
            for ket in remove_ket_list:
                del self.state_catalog[ket]
        # if update:
        #    if self.norm != DEFAULT_NORM: self.getNorm()
        #    if self.state != DEFAULT_STATE: self.getState()
            
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
        if self.state_catalog == DEFAULT_CATALOG:
            self.getStateCatalog() # this redefines self.state_catalog
        return sum(self.state_catalog.values(), [])  
    
    @property
    def is_weighted(self):
        if all(isinstance(val, tuple) for val in self.weights):
            return not all(isinstance(val, bool) for val in sum(self.weights, ()))
        elif all(isinstance(val, bool) for val in self.weights):
            return False
        elif all(isinstance(val, (int,float,complex)) for val in self.weights):
            return True
        else:
            raise ValueError('The weights are NOT defined correctly.')
    
    # imaginary could be redefined as one of these properties
    
    def addEdge(self, edge, weight=None, imaginary=False, update=True):
        if len(edge)==4 and all(isinstance(val, int) for val in edge):
            edge = tuple(edge)
        else:
            raise ValueError('Introduce a single valid edge.')
        if edge in self.edges():                
            print('The edge has been redefined.')
            update = False # there's no need to update the state_catalog
        else: # the edge is included
            self.graph[edge] = True
        if weight == None:
            self[edge] = defaultValues(1, self.imaginary)
        else:
            self[edge] = weight
        if update:
            other_nodes = [node for node in range(num_nodes) if node not in edge[:2]]
            subgraph = th.targetEdges(other_nodes, self.graph)
            new_states = th.findPerfectMatchings(subgraph + [edge])
            for ket, perfect_matchings in new_states.items():
                try:
                    state_catalog[ket] += perfect_matchings
                except KeyError:
                    state_catalog[ket] = perfect_matchings
        #     if self.norm != DEFAULT_NORM: self.getNorm()
        # if self.state != DEFAULT_STATE: self.getState()

    # This could be also a property, but then we cannot introduce arguments
    def node_degrees(self, ascending=False): # 
        '''
        Degree of each node of the graph.
        '''
        return th.nodeDegrees(self.edges, rising=ascending)
    
    def getStateCatalog(self):
        if self.full:
            self.state_catalog = th.allEdgeCovers(self.dimensions, order=0)
        else:
            self.state_catalog = th.stateCatalog(th.findPerfectMatchings(self.edges))
        print('Perfect matchings stored by the ket they produce in property `state_catalog`.')
    
    # the employed norm function is simplified (only pm) and uses the new imaginary notation
    def getNorm(self): 
        if self.state_catalog == DEFAULT_CATALOG:
            self.getStateCatalog()
        self.norm = th.writeNorm(self.state_catalog, self.imaginary)
        
    def getState(self):
        if self.state_catalog == DEFAULT_CATALOG:
            self.getStateCatalog()
        kets = list(self.state_catalog.keys())
        if self.is_weighted:
            ampltidues = []
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
            amplitudes = None
        self.state = State(kets, amplitudes, self.imaginary)  
        
    # This could also be defined as __abs__, but what do you give back? The dictionary?
    def absolute(self): 
        if self.is_weighted:
            if self.imaginary in [False, 'cartesian']:
                for kk, vv in self.graph.items():
                    self.graph[kk] = abs(vv)
            elif self.imaginary == 'polar':
                for kk, vv in self.graph.items():
                    self.graph[kk] = abs(vv[0])
            else:
                raise ValueError('The property `imaginary` is NOT defined correctly.')
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
            
    def clamp(maximum=1, minimum=None): #, rescale=False):
        if maximum >= 0:
            if minimum == None:
                minimum = - maximum
            for edge, weight in self.graph.items():
                self.graph[edge] = max( minimum, min(weight, maximum) )
        else:
            raise ValueError('Introduce a positive maximum.')


# # The State class

class State():
    
    def __init__(self,
                 kets, 
                 amplitudes = None, # list of values or tuples encoding imaginary values
                 imaginary = False, # 'cartesian' or 'polar'
                 normalize = False,
                ):
        
        self.imaginary = imaginary
        self.state = self.stateStarter(kets, amplitudes) # This may redefine previous properties
        if normalize: self.normalize()


    def stateStarter(self, kets, amplitudes):
        '''
        Function to initiate a State instance with different inputs.

        This version is not so flexible with the input format as the analogous from Graph.
        '''
        if type(kets) == dict:
            amplitudes = [kets[key] for key in sorted(kets.keys())]
            kets = sorted( kets.keys() )
        # Here, introducing the amplitudes after the kets in a list/tuple doesn't work
        elif type(kets) in [list, tuple]:
            pass # seems legit
        else:
            raise ValueError('Introduce a valid input `kets`.')

        # Verification of appropiate kets            
        kets_shape = np.shape(kets)
        if kets_shape[2] == 2: 
            pass # The third component must have 2 dimensions: (node, dim)
        else:
            raise ValueError('Introduce valid input `kets`.')

        # Verification and setting of appropiate amplitudes
        if amplitudes == None: # The default option True behaves (mostly) as 1
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
                        amplitudes = [val[0] + 1j*val[1] for val in amplitudes]
                    elif self.imaginary == 'polar':
                        amplitudes = [tuple(val[0], val[1]) for val in amplitudes]
                    else:
                        raise ValueError('Introduce a valid input `imaginary`.') 
                else:
                    raise ValueError('Introduce valid amplitudes.')
            else:
                raise ValueError('Introduce valid amplitudes.')

        # Once kets and amplitudes are properly defined, we return a state
        return {kt:val for kt,val in zip(kets, amplitudes)}
    
    def __str__(self):
        '''
        What you see when using print().
        '''
        return '{' + ',\n '.join(f'{kk}: {vv}' for kk,vv in self.state.items()) + '}'
    
    def __len__(self):
        return len(self.state)
    
    # def __eq__(self, other):
    # Options for establishing equivalence:
    # - Direct comparison of dictionaries: count kets as existing even if amplitude 0
    # - Using the inner product, but a@b=-.9999, is it equal? threshold needed 
    
    def __getitem__(self, ket):
        return self.state[ket]
    
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
    
    @property
    def kets(self):
        return list(self.state.keys())
    
    @property
    def amplitudes(self):
        return list(self.state.values())
    
    @property
    def norm(self):
        return ( self @ self ) ** 0.5
    
    @property
    def is_weighted(self):
        if all(isinstance(val, tuple) for val in self.amplitudes):
            return not all(isinstance(val, bool) for val in sum(self.amplitudes, ()))
        elif all(isinstance(val, bool) for val in self.amplitudes):
            return False
        elif all(isinstance(val, (int,float,complex)) for val in self.amplitudes):
            return True
        else:
            raise ValueError('The amplitudes are NOT defined correctly.')
    
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
    
    def clamp(self, maximum=1, minimum=None): #, rescale=False):
        if maximum >= 0:
            if minimum == None:
                minimum = - maximum
            for ket, amplitude in self.graph.items():
                self.state[ket] = max( minimum, min(amplitude, maximum) )
        else:
            raise ValueError('Introduce a positive maximum.')
    
    def normalize(self):
        self.rescale( constant = 1/self.norm )
