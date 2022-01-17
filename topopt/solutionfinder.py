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

import theseus as th
import topopt as top
import numpy as np
import time
import random
import itertools

# +
curr_ID=str(random.randint(10000,99999))

print('session '+str(curr_ID)+' init',flush=True)
tinit = time.perf_counter()

#define parameters of system
pdv =(5,3,8)
unicolor = True
samples = 10
print('#edges: '+str(top.edgeNum(pdv,unicolor)))
        
ghz = top.makeGHZ(pdv)
#build starting graph for system and find all perfect matchings 
locdim = [pdv[1]]*pdv[0]+[1]*(pdv[2]-pdv[0])
uncolored_edge_list = th.buildAllEdges([1]*pdv[2])
edge_list = th.buildAllEdges(locdim)

if unicolor:
    edge_list = top.makeUnicolor(edge_list)
    

print(str(curr_ID)+' '+str(len(uncolored_edge_list))+' uncolored edges',flush=True)

pms = th.findEdgeCoversColorLater(uncolored_edge_list)

tinit = time.perf_counter()
#define important functions
target = th.targetEquation([1]*pdv[1],ghz)
norm = th.Norm.fromEdgeCoversColorLater(edge_list)

#define loss
sc_loss = 1 - target/(1+norm)
fid_loss = 1 - target/norm
loss = sc_loss

print(str(curr_ID)+' init took '+str(round(time.perf_counter()-tinit,2))+'s',flush=True)
# -

ttot = time.perf_counter()
while samples>0:
    samples -= 1

    #initial optimization with all weights
    variables = top.variablesFromLoss(loss)
    vartoedge = dict(zip(variables,edge_list)) #CAREFUL this relies on both lists being sorted
    bounds = len(variables)*[(-1,1)]

    topt = time.perf_counter()
    condition = False
    while not condition: #fidelity of reoptimization is good and there are small edges
        print(str(curr_ID)+' opt of starting graph',flush=True)
        cont = True
        while cont: #go until fidelity and CR of optimization are good
            tstep = time.perf_counter()
            sol = th.sympyMinimizer(loss,variables = variables,bounds=bounds,method = 'L-BFGS-B',options={'maxfun': 15000})
            fid_check = top.evalLoss(fid_loss,variables,sol)
            if sol.fun<0.1 or fid_check<0.1: cont = False
            print(str(curr_ID)+' - took '+str(round(time.perf_counter()-tstep,2))+'s')

        print(str(curr_ID)+' opt of truncated graph',flush=True)
        #dictionary associating variables with values
        vals_start = dict(np.transpose([variables,sol.x]))

        #dictionary keeping track of deleted edges
        zero_vals_start = dict()

        #delete edges below a threshold
        thr_initial = 0.1 # can change this to 0.01
        for variable in variables:
            if abs(vals_start[variable]) < thr_initial:
                #vals_start[variable]= 0
                zero_vals_start[variable] = 0


        # set some variables to zero with dict zero_vals_start
        edge_list_start = edge_list.copy()
        delvars = list(zero_vals_start.keys())
        for delvar in delvars:
            edge_list_start.remove(vartoedge[delvar])

        target_start = target.subs(zero_vals_start)
        norm = th.Norm.fromEdgeCovers(edge_list_start)
        loss_start = 1 - target_start/(1+norm)
        fid_start = 1 - target_start/norm

        #make new optimization with truncated graph
        variables_start = top.variablesFromLoss(loss_start)
        bounds_start = len(variables_start)*[(-1,1)]
        sol_start = th.sympyMinimizer(loss_start,variables = variables_start,bounds=bounds_start,method = 'L-BFGS-B',options={'maxfun': 15000})

        cond1 = len(delvars)!=0
        cond2 = top.evalLoss(fid_start,variables_start,sol_start)<0.1
        condition = cond1 and cond2


    loss_cur = loss_start
    fid_cur = fid_start
    variables_cur = variables_start
    sol_cur = sol_start
    vals_cur = dict(np.transpose([variables_cur,sol_cur.x]))
    zero_vals_cur = zero_vals_start
    target_cur = target_start
    edge_list_cur = edge_list_start.copy()

    success = False
    rep = 0
    rep2 = 0
    tstep = time.perf_counter()
    print(str(curr_ID)+' init opt took '+str(round(time.perf_counter()-topt,2))+'s',flush=True)
    print(str(curr_ID)+' topopt',flush=True)
    
    
    while rep <16: #iterating through the smallest edges

        if rep2 ==0: #FIRST TIME TRYING REMOVAL NEW EDGE (update loss and use previous weights as initialization)

            

            #dictionary of zero weights, ready to add new entries
            zero_vals_new = zero_vals_cur.copy()

            #delete rep'th smallest weight
            vals_ordered = top.orderVals(vals_cur)
            variables_ordered = list(vals_ordered.keys())
            del_var = variables_ordered[-1-rep]

            zero_vals_new[del_var] = 0
            edge_list_new = edge_list_cur.copy()
            edge_list_new.remove(vartoedge[del_var])

            target_new = target_cur.subs(zero_vals_new)
            norm = th.Norm.fromEdgeCovers(edge_list_new)
            loss_new = 1 - target_new/(1+norm)

            #instead of getting variables from loss (can run into problems for small number of edges -> nonzero edges could disappear from loss. this problem could be solved by also deleting edges that disappear from loss out of vals_new, but it's also ok like it is now)
            variables_new = variables_cur.copy()
            variables_new.remove(del_var)
            bounds_new = len(variables_new)*[(-1,1)]

            #use previous weights as initialization
            vals_new = vals_cur.copy()
            vals_new.pop(del_var,None)
            init_vals = list(vals_new.values())

            sol_new = th.sympyMinimizer(loss_new,variables = variables_new,initial_values = init_vals,bounds=bounds_new,method = 'SLSQP')#,options={'disp':99})

        else: #try random initial values
            sol_new = th.sympyMinimizer(loss_new,variables = variables_new,bounds=bounds_new,method = 'SLSQP')#,options={'disp':99})

        #CHECKING IF NEW SOLUTION IS GOOD      
        if sol_new.fun<0.1: #CHECKING COUNT RATE
            fid_new = 1 - target_new/norm
            fid_check = top.evalLoss(fid_new,variables_new,sol_new)

            if fid_check < 0.05: #CHECKING FIDELITY
                print(str(curr_ID)+' edges left: '+str(len(variables_new))+'... step took '+str(round(time.perf_counter()-tstep,2))+'s',flush=True)
                tstep = time.perf_counter()
                success = True
                zero_vals_cur = zero_vals_new.copy()
                target_cur = target_new
                loss_cur = loss_new
                fid_cur = fid_new
                variables_cur = variables_new.copy() #list of varibale names
                sol_cur = sol_new #variables_new was used to generate sol_new --> they have the same order
                vals_cur = dict(np.transpose([variables_cur,sol_cur.x])) #dictionary  dict[variable name]=weight
                edge_list_cur = edge_list_new.copy()

                clean = top.cleanCheck(fid_cur,variables_cur,sol_cur,vals_cur)
                if clean == 'clean':
                    top.writeSol(fid_cur,variables_cur,sol_cur,vals_cur,pdv)

                rep = 0
                rep2 = 0
            else:
                rep2 += 1
                if rep2 >=5:
                    rep += 1
                    rep2 =0

        else:
            rep2 += 1
            if rep2 >=5:
                rep += 1
                rep2 =0

    top.writeSol(fid_cur,variables_cur,sol_cur,vals_cur,pdv)
    print(str(curr_ID)+' finished with '+str(len(variables_cur))+' edges. took '+str(round(time.perf_counter()-ttot,2))+'s',flush=True)


