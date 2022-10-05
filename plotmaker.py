import theseus
from theseus.main import run_main
from theseus.fancy_classes import Graph
from theseus.graphplot import leiwandPlotBulk
import os
import json
from IPython.utils import io

walk = os.walk('examples')
base = os.getcwd()
# go through all subdirectories of example folder
for root, dirs, files in walk:
    base = os.getcwd()
    if 'config.json' in files and 'plot.json' in files:
        print('example', root)
        filename = root + '/config.json'
        with io.capture_output() as captured: # doing this to prevent print spam from run_main
            cnfg = run_main(filename, False, run_opt=False)
        if cnfg["loss_func"] in ["cr", "fid"]:
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
            with open(root + '/plot.json') as input_file:
                sol_dict = json.load(input_file)
            graph = Graph(sol_dict['graph'], imaginary=cnfg['imaginary'])
            leiwandPlotBulk(graph, cnfg, root)
        elif cnfg["loss_func"] == 'ent':
            cnfg["vert_types"] = {}
            for ii, dim in enumerate(cnfg["dimensions"]):
                if dim > 1:
                    verttype = 'out'
                else:
                    verttype = 'anc'
                cnfg["vert_types"][ii] = verttype
            with open(root + '/plot.json') as input_file:
                sol_dict = json.load(input_file)
            graph = Graph(sol_dict['graph'], imaginary=cnfg['imaginary'])
            leiwandPlotBulk(graph, cnfg, root)
        elif cnfg["loss_func"] == 'fockcr':
            #TODO: implement (missing graph json atm)
            print('not implemented yet')
        else:
            print('skipped')
    os.chdir(base) # moving back to directory to continue walk
