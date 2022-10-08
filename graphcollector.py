import theseus
from theseus.main import run_main
from theseus.fancy_classes import Graph
from theseus.graphplot import leiwandPlotBulk
import os
import json
from IPython.utils import io
import shutil

foldername = 'theseus/graphs_cool'
walk = os.walk(foldername)
theseusbase = os.getcwd()
print(theseusbase)
# go through all subdirectories of example folder

for root, dirs, files in walk:
    base = os.getcwd()
    if not root.endswith('plots'):
        for file in files:
            if file.startswith('graph') and file.endswith('pdf'):
                shutil.copy(root+'/'+file, theseusbase +'/theseus/graphs_cool/plots/'+ file)
    os.chdir(base)  # moving back to directory to continue walk
