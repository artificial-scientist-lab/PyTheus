import itertools
import time
import copy
import random
from random import shuffle
import numpy as np
from scipy import optimize
import pytheus.fancy_classes as fc
import pytheus.theseus as th


def flatten(l):
    return [item for sublist in l for item in sublist]


def is_connected(lst):
    # print('lst: ',lst)
    in_list = [lst[0][0], lst[0][1]]

    cnum_vertices = len(set(flatten([[vv[0], vv[1]] for vv in lst])))

    curr_len = len(in_list) - 1
    # print('len(in_list)<cnum_vertices: ',len(in_list)<cnum_vertices)
    # print('len(in_list)>curr_len: ',len(in_list)>curr_len)

    while len(in_list) < cnum_vertices and len(in_list) > curr_len:
        curr_len = len(in_list)
        for ee in lst[1:]:
            if (ee[0] in in_list) and (not ee[1] in in_list):
                in_list.append(ee[1])
            if (ee[1] in in_list) and (not ee[0] in in_list):
                in_list.append(ee[0])

    # print(in_list)
    # print(cnum_vertices)
    return len(in_list) == cnum_vertices


def make_list_unique(lst):
    unique_lst = [list(x) for x in set(tuple(x) for x in lst)]
    return unique_lst


def is_substructure(lst1, lst2):
    if len(lst1) < len(lst2):
        return False

    rr = [ll in lst1 for ll in lst2]
    if not all(rr):
        return False

    unique_lst2 = make_list_unique(lst2)
    if len(lst2) > len(unique_lst2):
        return False

    return True


def compute_all_possibilies(full_graph, all_curr_subsequence, num_vertices, num_cols):
    # print('all_curr_subsequence: ', all_curr_subsequence)
    # time.sleep(0.5)
    all_permutations = list(itertools.permutations(list(range(num_vertices))))
    all_permutations_cols = list(itertools.permutations(list(range(num_cols))))
    curr_graph = all_curr_subsequence[-1]

    # print('len(all_permutations): ', len(all_permutations))

    all_curr_subsequence_ext = all_curr_subsequence + [[ee] for ee in full_graph]
    # print('len(all_curr_subgraphs_ext): ', len(all_curr_subsequence_ext))
    all_possibilities = []
    for curr_substr in all_curr_subsequence_ext:
        if len(curr_substr) == 1:
            new_graph = curr_graph + curr_substr
            if is_substructure(full_graph, new_graph):
                if not new_graph in all_possibilities:
                    all_possibilities.append(new_graph)

        if len(curr_substr) > 1:
            if is_connected(curr_substr):
                # print('curr_substr: ',curr_substr)
                for curr_perm in all_permutations:
                    # print('all_permutations: ', all_permutations)

                    # print('curr_substr: ',curr_substr)
                    for curr_col_perm in all_permutations_cols:
                        curr_substr_perm = []
                        for cedge in curr_substr:
                            nedge = [curr_perm[cedge[0]], curr_perm[cedge[1]], curr_col_perm[cedge[2]],
                                     curr_col_perm[cedge[3]]]
                            # print('nedge: ', nedge)
                            if nedge[0] > nedge[1]:
                                nedge = [nedge[1], nedge[0], nedge[2], nedge[3]]
                            curr_substr_perm.append(nedge)
                            # time.sleep(0.1)

                        new_graph = curr_graph + curr_substr_perm
                        # print('  new_graph: ',new_graph, '(curr_perm: ', curr_perm,')')

                        # time.sleep(0.1)
                        if is_substructure(full_graph, new_graph):
                            # print('is_substructure')
                            if not new_graph in all_possibilities:
                                # print('not new_graph in all_possibilities')
                                all_possibilities.append(new_graph)

    return all_possibilities


def compute_assembly_index(full_graph, all_curr_subsequence, assembly_index_col, num_vertices, num_cols):
    # print(' - - - - -')
    # print('in compute_assembly_index')
    # print('  full_graph: ',full_graph)
    # for csg in all_curr_subsequence:
    #    print('    csg: ', csg)
    # time.sleep(0.25)

    if len(full_graph) == 0:
        return 0, []
    if len(full_graph) == 1:
        return 1, [[full_graph[0]]]
    if len(full_graph) == 2:
        return 2, [[full_graph[0]], full_graph]

    global min_assembly_idx, min_ai_structure, assembly_index_collection
    num_vertices = max([max(ll[0:2]) for ll in full_graph]) + 1
    num_cols = max([max(ll[2:4]) for ll in full_graph]) + 1

    all_possibilies = compute_all_possibilies(full_graph, all_curr_subsequence, num_vertices, num_cols)
    # print('len(all_possibilies): ', len(all_possibilies))
    for new_graph in all_possibilies:
        new_curr_subgraphs = copy.deepcopy(all_curr_subsequence)
        new_curr_subgraphs.append(new_graph)
        # print('new_graph: ', new_graph)
        # time.sleep(1)

        if len(new_curr_subgraphs) < min_assembly_idx:
            if is_substructure(full_graph, new_graph) and is_substructure(new_graph, full_graph):
                # print(' DONE !!! Assembly Index: ', len(new_curr_subgraphs))
                assembly_index_col.append(len(new_curr_subgraphs))
                if len(new_curr_subgraphs) < min_assembly_idx:
                    min_assembly_idx = len(new_curr_subgraphs)
                    min_ai_structure = new_curr_subgraphs
                    # print('new best value: ', min_assembly_idx)
            else:
                compute_assembly_index(full_graph, new_curr_subgraphs, assembly_index_collection, num_vertices,
                                       num_cols)

    return min_assembly_idx, min_ai_structure


def assembly_index_unweighted(gg, num_vertices, num_cols):
    global min_assembly_idx, min_ai_structure, assembly_index_collection
    min_assembly_idx = 666
    min_ai_structure = []
    assembly_index_collection = []

    if len(gg) == 0:
        # print('assembly_index_collection: ', 0)
        return 0

    init_structure = [gg[0]]
    # print(gg)

    min_assembly_idx, min_ai_structure = compute_assembly_index(gg, [init_structure], assembly_index_collection,
                                                                num_vertices, num_cols)
    # print('assembly_index_collection: ', min_assembly_idx)
    # for ii in min_ai_structure:
    # print(ii)

    return min_assembly_idx


def sample_subgraph(graph, size_of_graph):
    all_edges = graph.edges
    all_weights = graph.weights

    curr_edges = []
    while len(curr_edges) < size_of_graph:
        ridx = random.randint(0, len(all_edges) - 1)
        if random.random() < abs(all_weights[ridx]):
            if not all_edges[ridx] in curr_edges:
                curr_edges.append(all_edges[ridx])

    curr_graph = fc.Graph(curr_edges)
    for edge in curr_edges:
        curr_graph[edge] = graph[edge]

    return curr_graph


def assembly_index(graph, cnfg):
    print("computing assembly index")
    num_vertices = cnfg["num_vertices"]
    num_cols = cnfg["num_cols"]
    size_of_graph = cnfg["size_of_graph"]

    all_sampled_assembly_indices = []

    for ii in range(cnfg["sample_size"]):
        sampled_graph = sample_subgraph(graph, size_of_graph)
        sampled_graph = sampled_graph.edges
        sampled_graph = [list(edge) for edge in sampled_graph]
        sampled_graph.sort()
        min_assembly_idx = assembly_index_unweighted(sampled_graph, num_vertices, num_cols)
        all_sampled_assembly_indices.append(min_assembly_idx)

    weighted_assembly_index = sum(all_sampled_assembly_indices) / len(all_sampled_assembly_indices)
    return weighted_assembly_index


def sample_top(graph, cnfg, ii):
    sorted_inds = list(np.argsort(graph.weights))
    sorted_inds.reverse()
    sorted_edges = [graph.edges[ind] for ind in sorted_inds]
    # weight of the edge that gets promoted
    weight = graph[sorted_edges[ii + cnfg["size_of_graph"]]]
    sampled_edges = sorted_edges[:cnfg["size_of_graph"] - 1] + [sorted_edges[ii + cnfg["size_of_graph"]]]
    sampled_edges = [list(edge) for edge in sampled_edges]
    sampled_edges.sort()
    return sampled_edges, weight


def sample_bottom(graph, cnfg, ii):
    sorted_inds = list(np.argsort(graph.weights))
    sorted_inds.reverse()
    sorted_edges = [graph.edges[ind] for ind in sorted_inds]
    # weight of the edge that gets demoted
    weight = graph[sorted_edges[ii]]
    sampled_edges = sorted_edges[:ii] + sorted_edges[ii + 1:cnfg["size_of_graph"]] + [sorted_edges[
                                                                                          cnfg["size_of_graph"] + ii]]
    sampled_edges = [list(edge) for edge in sampled_edges]
    sampled_edges.sort()
    return sampled_edges, weight


def top_n_assembly(graph, cnfg):
    print("computing top_n_assembly loss")
    num_vertices = cnfg["num_vertices"]
    num_cols = cnfg["num_cols"]
    size_of_graph = cnfg["size_of_graph"]

    for e in graph.edges:
        graph[e] = np.abs(graph[e])

    sorted_inds = list(np.argsort(graph.weights))
    sorted_inds.reverse()
    sorted_edges = [graph.edges[ind] for ind in sorted_inds]
    top_edges = sorted_edges[:cnfg["size_of_graph"]]
    top_edges = [list(edge) for edge in top_edges]
    top_edges.sort()
    curr_assembly = assembly_index_unweighted(top_edges, num_vertices, num_cols)

    lossfunc = 0

    for ii in range(len(graph) - cnfg["size_of_graph"]):
        # check the assembly index if smallest of top edges was switched out for each of the bottom edges
        # get weight of bottom edge to be promoted
        sampled_edges, weight = sample_top(graph, cnfg, ii)
        sample_assembly = assembly_index_unweighted(sampled_edges, num_vertices, num_cols)
        lossfunc += (sample_assembly - curr_assembly) * weight

    for ii in range(cnfg["size_of_graph"]):
        # check the assembly index if biggest of bottom edges was switched out for each of the top edges
        # get weight of top edge to be demoted
        sampled_edges, weight = sample_bottom(graph, cnfg, ii)
        sample_assembly = assembly_index_unweighted(sampled_edges, num_vertices, num_cols)
        lossfunc += (curr_assembly - sample_assembly) * weight

    return lossfunc


if __name__ == "__main__":
    cnfg = {
        "num_vertices": 4,
        "num_cols": 2,
        "size_of_graph": 8}
    gg = fc.Graph(th.buildAllEdges([2, 2, 2, 2]))
    for e in gg.edges:
        gg[e] = random.random()
    ai = top_n_assembly(gg, cnfg)
    print(ai)
