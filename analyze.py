import json
import sys
from fancy_classes import Graph, State
import graphplot as gp
import help_functions as hf

filename = sys.argv[1]

with open(filename) as data:
    sol = json.load(data)

def convert_graph_keys_in_tuple(graph: dict) -> dict:
    """
    here we can convert our Graph dict in a dict that has strings as keys
    we need this to save it in json file

    """
    # convert keys in str
    ret_dict = {}
    for key in graph.keys():
        if type(key) is str:
            try:
                ret_dict[eval(key)] = graph[key]
            except:
                pass

    return ret_dict
graph = sol['graph']
dic = convert_graph_keys_in_tuple(graph)
graph = Graph(dic)
graph.getState()
readable_state = hf.readableState(graph.state)
print(json.dumps(readable_state, indent=4))
print(graph)
gp.graphPlot(graph, scaled_weights=True, show=True, max_thickness=10)