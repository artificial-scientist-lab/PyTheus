# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import theseus as th
import topopt as top
import graphplot as gp
import numpy as np
import time
from scipy import optimize
import sys
import matplotlib.pyplot as plt

plt.ion()

sys.setrecursionlimit(10000000)

# + tags=[]
# define target state and starting graph. can use defineGHZ or any arbitrary combination of state and edge_list.
pdv = (4, 4, 8)
state, edge_list = top.defineGHZ(pdv, unicolor=True)
coeff = None
real = True  # define if weights should be real or complex numbers

print('target state:')
print('    ψ =', top.stateToString(state), '\n')
print('starting graph:')
print('    #edges =', len(edge_list))
# -


graph_init = gp.graphPlot(edge_list, scaled_weights=True, show=True, max_thickness=10)

# + tags=[]
# optimization parameters
samples = 1  # set how many solutions to produce
bulk_thr = 0.01  # threshold for truncating graph after optimization (set to zero if you don't want to truncate)
fid_thr = 0.1  # (1- fidelity) needs to be below this for good solution
cr_thr = 0.3  # (1-count rate) needs to be below this for good solution
ftol = 1e-05
preoptweights = []  # set preoptimized parameters. empty list means no preoptimization

# defining loss functions (count rate and fidelity)
cr, cr_string = top.makeLossString(state, edge_list, coeff=coeff, real=real)
fid, fid_string = top.makeLossString(state, edge_list, coeff=coeff, mode="fid", real=real)

while samples > 0:
    ttot = time.perf_counter()
    print("--- new sample ---")
    samples -= 1
    condition = False

    # INITIAL OPTIMIZATION
    # GENERAL IDEA:
    #     * do one initial optimization on full starting graph
    #     * check if initial optimization is good
    #     * delete bulk of small edges in initial solution (to speed things up before we delete edge by edge)
    #     * optimize truncated graph
    #     * check if truncated optimization is good
    #     * if any of the checks fail, redo the previous step until good truncated solution comes out

    print("INITIAL OPTIMIZATION")
    print("starting with", len(edge_list), "edges")
    while not condition:  # keep going until truncated optimization is good
        cont = True
        firsttry = True
        while cont:  # keep going until first optimization is good
            print("- optimizing starting graph")
            # initial optimization with random initial variables
            if firsttry and len(preoptweights) != 0:
                initial_values, bounds = top.prepOptimizer(len(edge_list), x=preoptweights, real=real)
                firsttry = False
            else:
                initial_values, bounds = top.prepOptimizer(len(edge_list), real=real)
            result = optimize.minimize(cr, x0=initial_values, bounds=bounds, method='L-BFGS-B',
                                       options={'ftol': ftol})  # use this to show bfgs progress: options={'disp':99}

            # checking solution
            fid_check = fid(result.x)
            if result.fun < cr_thr or fid_check < fid_thr: cont = False

            # count weights below bulk threshold
            numtrunc = np.sum([abs(result.x) < bulk_thr])

        if numtrunc == 0:
            print("- no truncation")
            edge_list_new = edge_list
            result_new = result
            cr_new = cr
            fid_new = fid
            condition = True
        else:
            print("- optimizing truncated graph, deleted edges:", str(numtrunc))
            # delete edges with small weight below a threshold
            thr = bulk_thr
            delind = top.setDeletedIndexThr(result.x, thr, real=real)
            edge_list_new, x_new = top.deleteEdges(edge_list, result.x, delind, real=real)
            # redefine loss functions for truncated graph
            cr_new, _ = top.makeLossString(state, edge_list_new, coeff=coeff, real=real)
            fid_new, _ = top.makeLossString(state, edge_list_new, coeff=coeff, mode="fid", real=real)

            # optimization of truncated graph with initial values given by initial solution
            initial_values, bounds = top.prepOptimizer(len(edge_list_new), x=x_new, real=real)
            result_new = optimize.minimize(cr_new, x0=initial_values, bounds=bounds, method='L-BFGS-B',
                                           options={'ftol': ftol})

            # check fidelity of truncated solution
            condition = fid_new(result_new.x) < fid_thr

    # setting up for stepwise topological optimization
    edge_list_cur = edge_list_new
    x_cur = result_new.x
    cr_cur = cr_new
    fid_cur = fid_new
    rep = 0
    rep2 = 0

    # STEP-BY-STEP TOPOLOGICAL OPTIMIZATION
    # GENERAL IDEA:
    #     * --0--
    #     * delete smallest edge and optimize with weights from previous solution
    #     * check solution 
    #         * if good: continue with deleting next edge (go back to --0--)
    #         * --1--
    #         * if bad: retry 5 times with random initial weights
    #             * if good: update edge list and continue with deleting next edge (go back to --0--)
    #             * if bad: leave edge and try deleting next biggest edge, go back to --1--
    #     * if there are no more edges that can be deleted and still giving good solution, save last good solution

    print('TOPOLOGICAL OPTIMIZATION')
    written = False
    print("starting with", len(edge_list_cur), "edges (after truncation)")
    while rep < min(len(edge_list_cur), 18):  # iterating through the edges (starting from smallest)

        if rep2 == 0:  # FIRST TIME TRYING REMOVAL NEW EDGE (update loss and use previous weights as initialization)
            # set up new edge list with weights after deleting one edge
            delind = top.setDeletedIndexSingle(x_cur, rep, real=real)
            edge_list_new, x_new = top.deleteEdges(edge_list_cur, x_cur, delind, real=real)
            contains_target = True
            try:
                # redefine loss functions for reduced graph
                cr_new, _ = top.makeLossString(state, edge_list_new, coeff=coeff, real=real)
                fid_new, _ = top.makeLossString(state, edge_list_new, coeff=coeff, mode="fid", real=real)

                # optimization of reduced graph
                initial_values, bounds_new = top.prepOptimizer(len(edge_list_new), x=x_new, real=real)
                result_new = optimize.minimize(cr_new, x0=initial_values, bounds=bounds_new, method='L-BFGS-B',
                                               options={'ftol': ftol})
            except KeyError:
                contains_target = False


        else:  # try same edge with random initial value
            initial_values, bounds_new = top.prepOptimizer(len(edge_list_new), real=real)
            result_new = optimize.minimize(cr_new, x0=initial_values, bounds=bounds_new, method='L-BFGS-B',
                                           options={'ftol': ftol})

            # CHECKING IF NEW SOLUTION IS GOOD
        if not contains_target:
            rep += 1  # increase edge index
            rep2 = 0  # reset attempt counter
        elif result_new.fun > cr_thr:  # checking if count rate fails
            rep2 += 1  # increase attempt counter
            if rep2 >= 5:  # same edge has been tried 6 times --> seems to be necessary, trying next biggest edge instead
                rep += 1  # increase edge index
                rep2 = 0  # reset attempt counter, new edge also gets 6 tries

        elif fid_new(result_new.x) > fid_thr:  # checking if fidelity fails
            rep2 += 1  # increase attempt counter
            if rep2 >= 5:  # same edge has been tried 6 times --> seems to be necessary, trying next biggest edge instead
                rep += 1  # increase edge index
                rep2 = 0  # reset attempt counter, new edge also gets 6 tries

        else:  # no checks failed
            print("edge deletion successful, edges left:", len(edge_list_new))

            # update graph, ready for next step 
            edge_list_cur = edge_list_new.copy()
            x_cur = result_new.x

            # not strictly necessary to save these, loss functions can be restored from edge_list and state 
            # still sometimes useful
            result_cur = result_new
            fid_cur = fid_new
            cr_cur = cr_new

            # this catches any clean solutions that occur before the smallest solution
            # if this is not done, sometimes a clean solution will appear, but be lost because there is a smaller rough solution that still passes the checks
            cleancheck = all(abs(x_cur[:len(edge_list_cur)]) > 0.95)
            if cleancheck:
                # write solution to file
                top.writeSol(edge_list_cur, x_cur, pdv, fid_cur, real=real)
                written = True
                print('clean solution saved to file')
            # reset counters to start with next step
            rep = 0
            rep2 = 0

    if not written:  # checking if solution is rough (clean solutions are saved as soon as detected)
        # write solution to file
        top.writeSol(edge_list_cur, x_cur, pdv, fid_cur, real=real)
        print('rough solution saved to file')
    print(
        'FINISHED with ' + str(len(edge_list_cur)) + ' edges. took ' + str(round(time.perf_counter() - ttot, 2)) + 's')
# -
graph_sol = gp.graphPlot(edge_list_cur, scaled_weights=True, show=True, max_thickness=10)

plt.ioff()
plt.show()
