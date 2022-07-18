# -*- coding: utf-8 -*-

###############################
##### general settings  ######
###############################

safe_hist = True
real = True
loss_func = 'cr'  # which lossfunciton: 'ent' , 'fid' or 'cr'

### opti infos ###
num_pre = 1  # amount of preoptis before topological opti
samples = 10
thresholds = [0.2, 0.1]
bulk_thr = 0.01

### optimizer settings ###
optimizer = "L-BFGS-B"
ftol = 1e-5

### save directory ###
foldername = '4-4-8'

###############################
###  statefinder settings   ###
###############################

target_state = ['0000', '1111', '2222', '3333']
num_anc = 4
unicolor = True

edges_tried = 10
tries_per_edge = 5
