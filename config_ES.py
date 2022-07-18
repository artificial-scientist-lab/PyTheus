# -*- coding: utf-8 -*-

###############################
##### general settings  ######
###############################

safe_hist =  True
real = True
loss_func = 'cr'  # which lossfunciton: 'ent' , 'fid' or 'cr'

### opti infos ###
num_pre = 1 # amount of preoptis before topological opti
samples = 10 
thresholds = [0.2,0.1]
bulk_thr = 0.01
# e.g. : lossfunc = 'cr' [cr_threshold, fid_threshold]
# e.g. : lossfunc = 'ent' [max_loss_increase_for_one_edge_deletion]

### optimizer settings ###
optimizer= "L-BFGS-B"
ftol = 1e-5

### safe directory ###
suffix = 'test' # suffix for directory eg. for 2222 name of savedirectory is 2222/suffix
foldername = '4d-ES'

###############################
### entanglement settings  ####
###############################

### Quantum infos  ###

###############################
###  statefinder settings   ###
###############################

target_state = ['00','11','22','33']
num_anc = 4
num_data_nodes = 2
unicolor = True
removed_connections = [[0,1]]



### loss function ###
epsilon = 0 #?maybe?
minimal_cycles = 1

















