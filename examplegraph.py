import fancy_classes
import help_functions

graph = fancy_classes.Graph({
        "(0, 1, 3, 3)": -1.0,
        "(0, 2, 0, 3)": -1.0,
        "(0, 2, 1, 1)": -1.0,
        "(0, 3, 0, 0)": -1.0,
        "(0, 4, 0, 0)": 1.0,
        "(0, 4, 2, 0)": -1.0,
        "(0, 5, 2, 0)": 1.0,
        "(1, 2, 0, 0)": 1.0,
        "(1, 3, 1, 0)": 1.0,
        "(1, 4, 1, 0)": -1.0,
        "(1, 4, 2, 0)": -1.0,
        "(1, 5, 2, 0)": 1.0,
        "(2, 3, 2, 0)": -1.0,
        "(2, 4, 2, 0)": -1.0,
        "(2, 4, 3, 0)": -1.0,
        "(2, 5, 3, 0)": -1.0,
        "(3, 4, 0, 0)": 1.0,
        "(3, 5, 0, 0)": 1.0,
        "(4, 5, 0, 0)": -1.0
    })
print('full graph info')
print(graph)
print('graph edges - list of tuples - (pos1, pos2, col1, col2)')
print(graph.edges)
print('graph weights - list')
print(graph.weights)
print('state associated with graph - dictionary ket->weight')
graph.getState()
print(help_functions.readableState(graph.state))
