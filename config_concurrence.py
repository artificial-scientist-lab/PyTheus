# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:24:33 2022

@author: janpe
"""


### Quantum infos ###
dim = 3333
real = True

### loss function ###
# L = mean_of_concurrence + variance * variance_of_concurence
var_factor = 0
K = 2 # lenght of bipar optimize for (K=2 for 2-uniform state or K = 'all' )
loss_func = 'ent'

### safe directory ###
suffix = 'test' # suffix for directory eg. for 2222 name of savedirectory is 2222/suffix
foldername = 'conc'

### opti infos ###
num_pre = 2
samples = 10
min_edge = 20
thresholds = [0.0001]
treshold = thresholds[0]

optimizer= "SLSQP"
norm_constrain = 0
after_threshold = 0.001
ftol = 1e-6

# extra calculations
dimensions = [int(ii) for ii in str(dim)]
if len(dimensions) % 2 != 0:
    dimensions.append(1)