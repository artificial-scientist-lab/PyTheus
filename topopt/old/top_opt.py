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
import numpy as np

#for checking computation times (time.perf_counter())
import time
import os
import sys
import json
import random


# +
#construct GHZ state from (p,d,v)
def makeGHZ(pdv):
    state = []
    data, dim, verts = pdv
    for ii in range(dim):
        term = []
        for jj in range(verts):
            if jj<data:
                term.append((jj,ii))
            else:
                term.append((jj,0))
        state.append(term)
    return state

#function for simplifying edge_list if wanted
def makeUnicolor(edge_list):
    return [edge for edge in edge_list if (((edge[0] not in range(5)) or (edge[1] not in range(5))) or (edge[2]==edge[3]))]

#function for manually checking the values
def orderVals(vals):
    vallist = list(vals.values())
    vallist = [[x,i] for i,x in enumerate(vallist)]
    vallist.sort(key=lambda s: abs(s[0]),reverse=True)
    x_reordered, inds = np.array(vallist)[:,0], np.array(vallist)[:,1].astype(int)
    keylist = list(vals.keys())
    variables_reordered = [keylist[i] for i in inds]
    vals_reordered = dict(np.transpose([variables_reordered,x_reordered]))
    return vals_reordered

def variablesFromLoss(loss):
    variables = list(loss.free_symbols)
    variables.sort(key=lambda s: str(s))
    return variables

def subsLoss(function, sol, zero_vals=dict()):
    simple_loss = function.subs(zero_vals)
    variables = variablesFromLoss(simple_loss)
    return simple_loss.subs(dict(np.transpose([variables,sol.x])))

def evalLoss(function,variables,sol):
    return function.evalf(subs = dict(zip(variables,sol.x)))

def cleanCheck(fid_cur,variables_cur,sol_cur,vals_cur):
    clean = 'clean'
    for v in list(vals_cur.values()):
        if abs(v)<0.95:
            clean = 'rough'
    return clean

def writeSol(fid_cur,variables_cur,sol_cur,vals_cur):
    #writing file
    pdvstr = '('+str(pdv[0])+'-'+str(pdv[1])+'-'+str(pdv[2])+')'
    clean = cleanCheck(fid_cur,variables_cur,sol_cur,vals_cur)
    edgenum = str(len(variables_cur))
    fid = evalLoss(fid_cur,variables_cur,sol_cur)
    fidelity = str(round(float(1-fid) ,2))
    combined_data=str(vals_cur)
    json_append(pdvstr+'-'+clean+'-'+edgenum+'-'+fidelity+'.json',combined_data)
    
def json_append(file_name, record):
    try:
        with open(os.path.join(sys.path[0],file_name), 'a') as f:
            json.dump(record, f)
            f.write('\n')
    except:
        print('write error.')


# +
#define parameters of system
pdv =(5,3,8)
unicolor = True
        
ghz = makeGHZ(pdv)

#build starting graph for system and find all perfect matchings 
locdim = [pdv[1]]*pdv[0]+[1]*(pdv[2]-pdv[0])
edge_list = th.buildAllEdges(locdim)
pms = th.findEdgeCovers(edge_list,order = 0)

if unicolor:
    edge_list = makeUnicolor(edge_list)

#define important functions
target = th.targetEquation([1]*pdv[1],ghz)
norm = th.Norm.fromEdgeCovers(edge_list)

#define loss
sc_loss = 1 - target/(1+norm)
fid_loss = 1 - target/norm
loss = sc_loss
# -

# # Optimization 

# +
curr_ID=str(random.randint(10000,99999))

t0 = time.perf_counter()

#initial optimization with all weights
variables = variablesFromLoss(loss)
vartoedge = dict(zip(variables,edge_list)) #CAREFUL this relies on both lists being sorted
bounds = len(variables)*[(-1,1)]

cont = True
condition = False
while not condition: #fidelity of reoptimization is good and there are small edges
    print('opt')
    while cont: #go until fidelity and CR of optimization are good
        print('-')
        sol = th.sympyMinimizer(loss,variables = variables,bounds=bounds,method = 'L-BFGS-B',options={'disp':99,'maxfun': 15000})
        fid_check = evalLoss(fid_loss,variables,sol)
        if sol.fun<0.1 or fid_check<0.1: cont = False


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
    variables_start = variablesFromLoss(loss_start)
    bounds_start = len(variables_start)*[(-1,1)]
    sol_start = th.sympyMinimizer(loss_start,variables = variables_start,bounds=bounds_start,method = 'L-BFGS-B',options={'disp':99,'maxfun': 15000})
    
    cond1 = len(delvars)!=0
    cond2 = evalLoss(fid_start,variables_start,sol_start)<0.1
    condition = cond1 and cond2

print('time for initial optimization ' +str(round(time.perf_counter()-t0,2)))
# -

# # topopt

loss_cur = loss_start
fid_cur = fid_start
variables_cur = variables_start
sol_cur = sol_start
vals_cur = dict(np.transpose([variables_cur,sol_cur.x]))
zero_vals_cur = zero_vals_start
target_cur = target_start
edge_list_cur = edge_list_start.copy()

# +
t0=time.perf_counter()
tstep = t0
success = False
rep = 0
rep2 = 0

while rep <16: #iterating through the smallest edges
        
    if rep2 ==0: #FIRST TIME TRYING REMOVAL NEW EDGE (update loss and use previous weights as initialization)
        
        print('--------')
    
        #dictionary of zero weights, ready to add new entries
        zero_vals_new = zero_vals_cur.copy()
    
        #delete rep'th smallest weight
        vals_ordered = orderVals(vals_cur)
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
        fid_check = fid_new.evalf(subs = dict(zip(variables_new,sol_new.x)))
        
        if fid_check < 0.05: #CHECKING FIDELITY
            print('edges left: '+str(len(variables_new))+' step took '+str(round(time.perf_counter()-tstep,2))+'s')
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
            
            clean = cleanCheck(fid_cur,variables_cur,sol_cur,vals_cur)
            if clean == 'clean':
                writeSol(fid_cur,variables_cur,sol_cur,vals_cur)
            
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


total_time = time.perf_counter()-t0
print(total_time)

writeSol(fid_cur,variables_cur,sol_cur,vals_cur)
# -


