# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:15:59 2022

@author: janpe
"""
import os.path

from .fancy_classes import Graph
import numpy as np
from pathlib import Path
import json


def read_json(abspath: Path) -> dict:
    """
    return the dictonary for a given abspath that is a json file

    """
    with abspath.open('r', encoding="UTF-8") as openfile:
        # Reading from json file
        dictonary = json.load(openfile)
        openfile.close()

    return dictonary


def write_json(abspath: Path, dictionary: dict,
               overwrite_existing_file = False) -> None:
    """
    getting an Path object and a dictonary then write the dictionary
    in a json file for the given path

    """

    # Serializing json
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.int32):
                return int(obj)
            if isinstance(obj, np.int64):
                return int(obj)
            return json.JSONEncoder.default(self, obj)

    if not isinstance(abspath, Path):
        abspath = Path(abspath)

    # convert json object
    json_object = json.dumps(dictionary, cls=NumpyEncoder,
                             ensure_ascii=False, indent=4)

    ### write json to new folder ###
    # make json ending
    abspath_save = str(abspath)
    if not abspath_save.endswith('.json'):
        abspath_save += '.json'
    # check if same state exists
    if overwrite_existing_file is False:
        counter = 0
        while Path(abspath_save).exists():
            abspath_save = str(abspath) + f'({counter})' + '.json'
            counter += 1
        
    abspath = Path(abspath_save)
    with abspath.open("w+", encoding="UTF-8") as file:
        file.write(json_object)
        file.close()


class saver:

    def __init__(self, config=None, name_config_file = '', dim = []):

        self.name_config_file = Path(name_config_file).name
        self.config = config
        self.config['dimensions'] = dim
        self.best_state = None
        self.save_path = self.get_and_create_save_directory()
        self.best_opt = None

    def get_folder_name(self) -> str:
        """
        return the folder name: dimension of the system or confi

        """
        folder = self.name_config_file
        if self.name_config_file.endswith('.json'):
            folder = self.name_config_file[:-5]
        return Path(folder) / self.config.get('foldername', 'try')

    def get_and_create_save_directory(self):
        """
        look if folder ~/output/(2-2-2-2) exists otherwise creates it
        when exists it looks if config files are the same then our
        safe directory is ~/output/(2-2-2-2) otherwise: we chose
        ~/data/(2-2-2-2)(x) for a x where the summary files are matching
        or a new directory
        """
        folder_name = self.get_folder_name()

        i = 0
        while True:  # iterate as long as one could find a proper safe folder
            pt = Path.cwd() / 'output' / folder_name  # move data directory
            pt.mkdir(parents=True, exist_ok=True)
            summary_path = pt / 'summary.json'
            if summary_path.exists():
                if self.check_if_summary_is_same(summary_path):
                    self.foldername = Path(folder_name)
                    return pt
                else:  # adapt folder name to avoid same folder with different summaries
                    folder_name = str(self.get_folder_name()) + f'_{i}'
                    i += 1
            else:
                self.foldername = Path(folder_name)
                write_json(pt / 'summary.json', self.config)
                return pt

    def check_if_summary_is_same(self, path_exst_summary: Path) -> bool:
        """
        just checks if the existing summary and the current have the
        same parameters, the compare function is needed for
        distinguishing iterable and non iterables
        """

        def compare(obj_a, obj_b):
            try:
                __ = iter(obj_a)
                return all([a == b for a, b in zip(obj_a, obj_b)])
            except TypeError:
                return obj_a == obj_b

        existing_summary = read_json(path_exst_summary)
        try:
            same = all([compare(self.config[key], existing_summary[key])
                        for key in self.config.keys() if key != 'samples'] )
        except KeyError:
            same = False
        return same

    def convert_graph_keys_in_str(self, graph: dict) -> dict:
        """
        here we can convert our Graph dict in a dict that has strings as keys
        we need this to save it in json file

        """
        # convert keys in str
        ret_dict = {}
        for key in graph.keys():
            if type(key) is not str:
                try:
                    ret_dict[str(key)] = graph[key]
                except:
                    try:
                        ret_dict[repr(key)] = graph[key]
                    except:
                        pass

        return ret_dict

    def check_best_opt(self, topo):
        # check if opt is better than previous best_opt
        # this may not only depend on the loss function. for example, a solution with fidelity 0.994 and 16 edges
        # is 'better' in some cases than fidelity 0.999 and 23 edges
        # TODO: implement the choice of which criteria is used
        return self.best_opt is None or topo.loss_val < self.best_opt.loss_val
    
    def get_dictonary_storing_in_json(self,topo_obj):
        """
        converts the topological optimizer object to a dict

        Parameters
        ----------
        topo_obj : Object of topological_opti

        Returns
        -------
        safe_dic : dict

        """
        
        
        safe_dic = {'graph': self.convert_graph_keys_in_str(topo_obj.graph.graph),
                    'loss': topo_obj.loss_val,
                    'seed': self.config['seed'],
                    'history': topo_obj.history}

        try:
            safe_dic['graph_hist'] = [self.convert_graph_keys_in_str(xx.graph)
                                      for xx in topo_obj.graph_hist]
            safe_dic['loss_hist'] = topo_obj.loss_hist
        except AttributeError:
            pass
        return safe_dic
    
    def save_graph(self, topo: object) -> None:
        """
        we use an object from the class topological_opti to save all
        infos: - the optimized graph
               - corresponding loss
               - if confi.safe_hist is True we also safe the loss during
                 each deletion and the corresponding graph

        """

        abs_path = self.save_path / self.get_file_name(topo.graph, topo.loss_val)
        # update best graph
        if self.check_best_opt(topo):
            self.best_opt = topo
            safe_dic_best = self.get_dictonary_storing_in_json(topo)
            print('new best state saved to best.json')
            write_json(self.save_path / 'best' , safe_dic_best, 
                       overwrite_existing_file = True)
            # TODO: rewrite best_graph to file on this line (then we get best graph even if run is cancelled before it is finished)
            
        safe_dic = self.get_dictonary_storing_in_json(topo)

        write_json(abs_path, safe_dic)

    def get_file_name(self, graph: Graph, loss: float) -> str:
        """
        takes as input object Graph and generate a file name out of it

        """
        if all(np.array(abs(graph)) > 0.95):
            clean = 'clean'
        else:
            clean = 'rough'
        file_name = clean + '-' + str(len(graph.graph)) + '-'
        graph.getStateCatalog()
        file_name += str(len(graph.perfect_matchings)) + '-'
        for lo in loss:
            file_name += f'{lo:.4f}_'
        return file_name[:-1] # delet last _
