# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:24:53 2022

@author: janpe
"""
import os

file_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_path)
from optimizer import topological_opti

from fancy_classes import Graph, State
import config_ghz as confi
import theseus as th
import help_functions as hf
import sys
from state import state1
import saver

# %%

cnfg = {item: getattr(confi, item) for item in dir(confi)
        if not item.startswith("__") and not item.endswith("__")}

sys.setrecursionlimit(1000000000)

try:
    # target state optimization
    sys_dict = None
    # add ancillas
    term_list = [term + confi.num_anc * '0' for term in confi.target_state]
    target_state = State(term_list)
    target_kets = target_state.kets
    # define local dimensions
    dimensions = th.stateDimensions(target_kets)
except:
    # concurrence optimization
    # define local dimensions
    dimensions = [int(ii) for ii in str(confi.dim)]
    target_state = None
    sys_dict = hf.get_sysdict(dimensions, bipar_for_opti=confi.K)

# build starting graph
edge_list = th.buildAllEdges(dimensions, real=confi.real)
try:
    if confi.unicolor:
        num_data_nodes = len(confi.target_state[0])
        edge_list = hf.makeUnicolor(edge_list, num_data_nodes)
except:
    pass

try:
    edge_list = hf.removeConnections(edge_list, confi.removed_connections)
except:
    pass

print(f'start graph has {len(edge_list)} edges.')
start_graph = Graph(edge_list)

# topological optimization
sv = saver.saver(config=cnfg)
for i in range(confi.samples):
    optimizer = topological_opti(start_graph, ent_dic=sys_dict, target_state=target_state, config=cnfg)
    graph_res = optimizer.topologicalOptimization()
    sv.save_graph(optimizer)

graph_res.getState()
print(f'finished with graph with {len(graph_res.edges)} edges.')
print(graph_res.edges)
