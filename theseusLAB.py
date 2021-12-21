# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import theseus as th

import time
import itertools
from math import factorial, ceil
from collections import Counter
import random

import numpy as np
import scipy.optimize as optimize
from sympy import symbols, sqrt, simplify
from sympy.solvers import solve
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
import json

# + [markdown] tags=[]
# ## Graphs, states and representations
# Here we see some examples of how graphs can be build up to arbitrary order and how, from these, we can extract expressions like the norm.
# -

dimensions = [2]*2 + [1]*2 # we usually add the ancillas at the end

# + [markdown] tags=[]
# ### Graphs with all possible edges
# -

# The function `allEdgeCovers` builds all possible edge covers given the input dimensions of the nodes and store them in a dictionary. The graphs that leads to the same state are stored in the same key. 
#
# For order=0 we have perfect matchings, for higher orders we can have duplicated edges and even loops.

full_graph0 = th.allEdgeCovers(dimensions, 0)
full_graph1 = th.allEdgeCovers(dimensions, 1, loops=True)

# The notation for the nodes is `(node, color/dimension)`, for the edges that consitute the graphs `(node1, node2, color1, color2)`.
#
# We can call the different states with a tuple of ordered nodes.

for graph in full_graph1[((0,0),(0,0),(0,1),(1,1),(2,0),(3,0))]:
    print(graph)

# Once we have the states we want to compute up to a certain order, we can join them in a single dictionary to get some expressions.

full_graph = full_graph0.copy()
full_graph.update(full_graph1)
full_norm = th.Norm.fromDictionary(full_graph)
full_state = th.State.fromDictionary(full_graph)

th.Norm.fromDictionary(full_graph0,padding=False)

th.State.fromDictionary(full_graph0,padding=False)

# **By using Notebooks, we can show the previous sympy expressions in Latex format**. However, this procedure is far more demanding than simply store them and for long expressions the file may crash. This happens quite often when we have complicated fractions with long denominators. 
#
# We can combine several expressions obtained from multiple dictionaries or combine dictionaries to get a single expression. However, the `Norm` and `State` functions also includes some options to combine states from different number of edges.

full_norm == th.Norm.fromDictionary(full_graph0,padding=False) + th.Norm.fromDictionary(full_graph1,padding=False)

full_norm_at_once = th.Norm.fromDimensions(dimensions,max_order=1,loops=True,padding=False)
full_norm == full_norm_at_once

# **Warning!** While padding symbols is irrelevant from the physics perspective, it change the sympy symbol. Be consistent or the optimization functions will fail.

full_norm == th.Norm.fromDimensions(dimensions,max_order=1,loops=True,padding=True)

# ### Set up optimization to find states

# Using the general graph we can look for concrete state like a GHZ for 2 particles + 2 ancillas: $|0000\rangle-|1100\rangle=(|00\rangle-|11\rangle)|00\rangle$.
#
# For simplicity we will do it with the lower order, only with perfect matchings.

target_states = [((0,0),(1,0),(2,0),(3,0)),((0,1),(1,1),(2,0),(3,0))]
target_eq = th.targetEquation(coefficients=[1,-1], states=target_states)

# Under the hood, we are using the `minimize` function from scipy and the object we get is an [optimize result](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult).

norm0 = th.Norm.fromDimensions(dimensions)
loss = target_eq/(1+norm0)
weights = loss.free_symbols
result = th.sympyMinimizer(loss_function=-loss,
                           variables=weights,
                           bounds=[(-1,1)]*len(weights))
solution = {key:round(value,3) for key, value in zip(weights,result.x)}

# While for higher orders new terms would appear, using only perfect matching we get the state we were looking for. Nonetheless, simpler solutions might be found with the topological optimizator.

state0 = th.State.fromDimensions(dimensions)
(state0/sqrt(norm0)).subs(solution) # solution in terms of creator operators

# + [markdown] tags=[]
# ### Getting expressions from graphs
# -

# So far, we built all possible states with all possible edges given a list of dimensions. We just had to set the maximum order we want to compute. While this is the more general system given a set of dimensions, it is also computationally expensive. Rather, we can introduce a graph with only some edges and see which states can come out from it.
#
# We start building a graph with the function `buildRandomGraph`. Then we see which perfect matchings and/or edge covers can be created from such list of edges, we can do it respectively with `findPerfectMatchings` and `findEdgeCovers`. Finally, introducing all these subgraphs in the function `stateCatalog` these get grouped by the state they produce.

new_dims = [2]*4 + [1]*2
random.seed(13)
graph = th.buildRandomGraph(new_dims, num_edges=10, loops=True, cover_all=True)
graph

graph_pm = th.findPerfectMatchings(graph)
graph_ec = th.findEdgeCovers(graph, order=1, loops=True)
graph_states = th.stateCatalog(graph_pm + graph_ec)

# Now we are ready to build the norm from the dictionary or, alternatively, use the version `fromEdgeCovers` of `Norm/State`. In which we only need to introduce the original graph.

th.Norm.fromDictionary(graph_states) 

th.Norm.fromEdgeCovers(graph,max_order=1,loops=True)
