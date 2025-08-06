import numpy as np


def fermion_sign(subgraph):
    """
    Compute sign factor for fermionic perfect matchings.
    Each edge in subgraph is (u, v, colour_u, colour_v). Colours are ignored.
    The sign is (-1)**crossings where crossings count pairs of edges (u1,v1) and (u2,v2) with u1 < u2 < v1 < v2 or u2 < u1 < v2 < v1.
    """
    pairs = []
    for edge in subgraph:
        u, v = edge[:2]
        if u < v:
            pairs.append((u, v))
        else:
            pairs.append((v, u))
    crossings = 0
    for i in range(len(pairs)):
        u1, v1 = pairs[i]
        for j in range(i + 1, len(pairs)):
            u2, v2 = pairs[j]
            if (u1 < u2 < v1 < v2) or (u2 < u1 < v2 < v1):
                crossings += 1
    return -1 if (crossings % 2) else 1


def fermion_fidelity(graph, target_state, cnfg):
    """
    Fermionic fidelity loss function.

    Returns a function of weights that computes 1 minus the squared overlap with the target state, including fermionic sign factors.
    """
    kets = list(graph._state_catalog.keys())
    target_unnormed = np.array([(ket in target_state.kets) * 1.0 for ket in kets])
    target_normed = target_unnormed / np.linalg.norm(target_unnormed)
    state_catalog_tensor = np.array(graph._state_catalog_tensor)
    target_normed = np.array(target_normed)

    complete_edges = graph.complete_graph_edges
    sign_array = np.zeros(state_catalog_tensor.shape[0])
    for idx, row in enumerate(state_catalog_tensor):
        subgraph = [complete_edges[i] for i in row]
        sign_array[idx] = fermion_sign(subgraph)

    def graph_state(edges):
        vals = edges[state_catalog_tensor].prod(axis=-1)
        return (vals * sign_array).sum(axis=-1)

    def normed_state(state):
        return state / np.linalg.norm(state, axis=-1)

    def overlap(state):
        return np.abs(np.dot(state, target_normed)) ** 2

    def func0(x):
        return 1 - overlap(normed_state(graph_state(x)))

    mat = np.zeros((len(graph.complete_graph_edges), len(graph.edges)))
    for i, edge in enumerate(graph.edges):
        mat[graph.complete_graph_edges.index(edge), i] = 1

    return lambda x: func0(np.dot(mat, x))


def fermion_count_rate(graph, target_state, cnfg):
    """
    Count-rate loss for fermionic systems.

    Equivalent to bosonic count_rate since sign factors cancel when computing probabilities.
    """
    return count_rate(graph, target_state, cnfg)
