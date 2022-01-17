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
import numpy as np
import os
import sys
import json

def edgeNum(pdv,uc):
    #calculate number of edges from pdv
    p,d,v = (pdv)
    a = v-p
    if uc:
        return p*(p-1)*d//2 + p*a*d + a*(a-1)//2
    else:
        return p*(p-1)*d*d//2 + p*a*d + a*(a-1)//2

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

def jsonAppend(file_name, record):
    try:
        with open(os.path.join(sys.path[0],'data',file_name), 'a') as f:
            json.dump(record, f)
            f.write('\n')
    except:
        print('write error.')

def writeSol(fid_cur,variables_cur,sol_cur,vals_cur,pdv):
    #writing file
    pdvstr = '('+str(pdv[0])+'-'+str(pdv[1])+'-'+str(pdv[2])+')'
    clean = cleanCheck(fid_cur,variables_cur,sol_cur,vals_cur)
    edgenum = str(len(variables_cur))
    fid = evalLoss(fid_cur,variables_cur,sol_cur)
    fidelity = str(round(float(1-fid) ,2))
    combined_data=str(vals_cur)
    jsonAppend(pdvstr+'-'+clean+'-'+edgenum+'-'+fidelity+'.json',combined_data)


# -

