import pytheus
from pytheus.main import run_main
from pytheus.fancy_classes import Graph
from pytheus.graphplot import leiwandPlotBulk
import os
import json
from IPython.utils import io

foldername = 'graphs_new/FockStates'
walk = os.walk(foldername)
base = os.getcwd()
# go through all subdirectories of example folder

ccount = 0
pcount = 0
bcount = 0
ocount = 0

# skip = ['cluster_4',  # inconsistency plot needs more ancillas than given by config
#       'cnot22_sp',  # inconsistency plot needs more ancillas than given by config
#      'cnot23_sp',  # inconsistency plot needs more ancillas than given by config
##    ]
skip = []
startingcount = 0  # set this to a number to skip previous directories
directorycount = 0
for root, dirs, files in walk:
    directorycount += 1
    base = os.getcwd()
    config = False
    plot = False
    print(root)
    print(files)
    for file in files:
        if file.startswith('config'):
            config = True
            configname = file
            ccount += 1
        if file.startswith('plot'):
            plot = True
            plotname = file
            pcount += 1
    if config or plot:
        ocount += 1
    name = root.split('/')[-1]
    print(config, plot, directorycount, startingcount)
    if config and plot and (name not in skip) and (directorycount >= startingcount):
        bcount += 1
        print(name)
        print('directorycount', directorycount)
        filename = root + '/' + configname
        with io.capture_output() as captured:  # doing this to prevent print spam from run_main
            cnfg = run_main(filename, False, run_opt=False, state_cat=False)
        if cnfg["loss_func"] in ["cr", "fid", "fockcr", "fockfid "]:
            # define ancilla nodes
            nonanc = cnfg["out_nodes"] + cnfg["in_nodes"] + cnfg["single_emitters"]
            cnfg["anc_nodes"] = [vert for vert in cnfg["verts"] if vert not in nonanc]
            vert_types = {}
            for vert in cnfg["verts"]:
                if vert in cnfg["out_nodes"]:
                    vert_types[vert] = 'out'
                elif vert in cnfg["in_nodes"]:
                    vert_types[vert] = 'in'
                elif vert in cnfg["single_emitters"]:
                    vert_types[vert] = 'sps'
                else:
                    vert_types[vert] = 'anc'
            if 'mixed' in cnfg["description"]:
                mixind = len(cnfg["out_nodes"]) - 1
                vert_types[mixind] = 'mix'
            cnfg['vert_types'] = vert_types
            # load graph
            with open(root + '/' + plotname) as input_file:
                sol_dict = json.load(input_file)
            graph = Graph(sol_dict['graph'], imaginary=cnfg['imaginary'])
            leiwandPlotBulk(graph, cnfg, root, name='graph_' + name)
        elif cnfg["loss_func"] == 'ent':
            cnfg["vert_types"] = {}
            for ii, dim in enumerate(cnfg["dimensions"]):
                if dim > 1:
                    verttype = 'out'
                else:
                    verttype = 'anc'
                cnfg["vert_types"][ii] = verttype
            with open(root + '/' + plotname) as input_file:
                sol_dict = json.load(input_file)
            graph = Graph(sol_dict['graph'], imaginary=cnfg['imaginary'])
            leiwandPlotBulk(graph, cnfg, root, name='graph_' + name)
        elif cnfg["loss_func"] == 'fockcr':
            # TODO: implement (missing graph json atm)
            print('not implemented yet')
        else:
            print('skipped')
    # print(ccount, pcount, bcount, ocount)
    os.chdir(base)  # moving back to directory to continue walk
print('finished')
