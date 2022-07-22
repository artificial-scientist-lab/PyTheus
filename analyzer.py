# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:19:47 2022

@author: janpe
"""

from fancy_classes import State
from pathlib import Path
import json

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

def convert_file_path_to_dic(abs_file_path: Path)->dict:
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
        if len( list(Path(self.folder).rglob('*.json'))) == 0:
            raise ValueError(f'The given path {self.folder} does not contain jsons')
        elif self.folder.exists() is False:
            raise ValueError(f'The given path {self.folder} does not exist')
        
    def get_all_states_in_folder_and_summary_file(self):
        #iterate through all files in folder ending with json and append to files
        self.files = []
        for path in Path(self.folder).rglob('*.json'): 
            if 'summary'  in str(path):
                self.summary = convert_file_path_to_dic(path)
            elif 'best' in str(path):
                self.best = convert_file_path_to_dic(path)
            else:
                self.files.append( convert_file_path_to_dic(path) )
        
    


a = analyser('C:/Users/janpe/Google Drive/6.Semester/Bachlorarbeit/Code/public_git/Theseus/data/conc')
print(a.files[0])



    

    #%%
# graph = sol['graph']
# dic = convert_graph_keys_in_tuple(graph)
# graph = Graph(dic)
# graph.getState()
# readable_state = hf.readableState(graph.state)