# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:19:47 2022

@author: janpe
"""

from fancy_classes import State
from pathlib import Path
import json
import matplotlib.pyplot as plt


def convert_graph_keys_in_tuple(graph: dict) -> dict:
    """
    here we can convert our Graph dict in a dict that has strings as keys
    we need this to save it in json file
    """
    # convert keys in str
    ret_dict = {}
    for key in graph.keys():
        if type(key) is str:
            try:
                ret_dict[eval(key)] = graph[key]
            except:
                pass

    return ret_dict


def convert_file_path_to_dic(abs_file_path: Path) -> dict:
    with abs_file_path.open() as file:
        dictionary = json.load(file)
    file.close()
    return dictionary


class analyser():

    def __init__(self, folder):
        self.folder = Path(folder)
        self.check_folder_name()
        self.get_all_states_in_folder_and_summary_file()

    def check_folder_name(self):
        if len(list(Path(self.folder).rglob('*.json'))) == 0:
            raise ValueError(
                f'The given path {self.folder} does not contain jsons')
        elif self.folder.exists() is False:
            raise ValueError(f'The given path {self.folder} does not exist')

    def get_all_states_in_folder_and_summary_file(self):
        # iterate through all files in folder ending with json and append to files
        self.files = []
        for path in Path(self.folder).rglob('*.json'):
            if 'summary' in str(path):
                self.summary = convert_file_path_to_dic(path)
            elif 'best' in str(path):
                self.best = convert_file_path_to_dic(path)
            else:
                self.files.append(convert_file_path_to_dic(path))
        self.files = sorted(self.files, key = lambda dic: dic["loss"])    
        
    def plot_losses(self, loss_idx=0):
        """
        plots y-axis: concurrence, x-axis: number of edges for a whole
        list of state dicts, plotvline plots a vertical line for a given positon

        Parameters
        ----------
        sysdic_list : list
            of state dicts
        plotvline : float, optional
            Position of vertical line. The default is 0.
        plot_mean : Boolean, optional
            IF mean should be plotted. The default is True.

        """

        fig = plt.figure()
        ax = fig.add_subplot(111)
        NUM_COLORS = len(self.files)
        cm = plt.get_cmap('gist_rainbow')
        ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS)
                          for i in range(NUM_COLORS)])

        for idx, st in enumerate(self.files):
            loss_history = st['history']
            min_edge = len(st['graph'])
            edges = [ii for ii in range(
                min_edge, min_edge+len(loss_history))]
            edges.reverse()
            loss = [ll[loss_idx] for ll in loss_history]
            ax.plot(edges, loss, label=idx, alpha=0.4, lw=3)
            ax.plot(edges, loss, alpha=0.6, lw=0.7, color="black")
            ax.set_xlabel("Amount of edges left")
            ax.set_ylabel("Loss function")
        ax.invert_xaxis()
        ax.grid()
        plt.show()
        
    def turn_graph_in_state(self,graph):
        ancillas = self.summary['dimensions'].count(1)
        graph.getState()
        if ancillas != 0:
            end_res = dict()
            for kets,ampl in graph.state.state.items():
                end_res[kets[:-ancillas]] = ampl
        else: end_res = graph.state.state
        ## TODO: ...


path = r'C:/Users/janpe/Google Drive/6.Semester/Bachlorarbeit/Code/public_git/Theseus/data/conc_4-3/try'
path = r'C:/Users/janpe/Google Drive/6.Semester/Bachlorarbeit/Code/public_git/Theseus/data/aklt_3/AKLT_3'
a = analyser(path)
a.plot_losses(1)

# %%
# graph = sol['graph']
# dic = convert_graph_keys_in_tuple(graph)
# graph = Graph(dic)
# graph.getState()
# readable_state = hf.readableState(graph.state)
