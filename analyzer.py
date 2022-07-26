# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:19:47 2022

@author: janpe
"""
from cmath import polar
from fancy_classes import State, Graph
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from state import state1 as sst
import matplotlib.gridspec as gridspec
import graphplot as gp
import help_functions as hf
from scipy.linalg import logm
from theseus import ptrace
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{braket}\usepackage{xcolor}')



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


def num_in_str(num, dec=2, change_sign=False):
    if np.round(num.imag, dec) != 0:
        polar_num = polar(num)
        if not change_sign:
            num = np.round([polar_num[0], polar_num[1]/np.pi],  dec)
        else:
            num = np.round([-polar_num[0], polar_num[1]/np.pi],  dec)
        return '{} * exp( {}*i*π )'.format(*num).replace('1.0 * ', '')
    else:
        rounded = np.round(num.real, dec)
        if not change_sign:
            if rounded == 1 or rounded == -1:
                return ''
            return f'{np.round(num.real,dec)}'
        else:
            if rounded == 1 or rounded == -1:
                return ''
            return f'{-np.round(num.real,dec)}'


entanglement_functions = {
    'con': lambda matrix: abs((2*(1 - min(np.trace(matrix@matrix), 1)))**0.5),
    'entropy': lambda matrix: -np.trace(matrix@logm(matrix)),
    'dense': lambda matrix: np.trace(matrix@matrix)}


class entanglement_measure():

    def __init__(self, dimensions: list, measure_kind='con'):
        if dimensions.count(1) == 0:
            self.dims = dimensions
        else:
            self.dims = dimensions[:-dimensions.count(1)]
        self.num_par = len(self.dims)
        self.densitys = self.maximal_mixed_densitys()
        self.func = entanglement_functions[measure_kind]
        self.m = measure_kind
        self.len = self.amount_bipar()

    def maximal_mixed_densitys(self) -> list:
        """
        return a list containing all reduced density matrices for that
        are maximally mixed:
            e.g: for dim = [2,2,2,2] we have 2 splittings:
                - one particle and the rest
                - and 2 particles and 2 particles
        -> returns list  maximal mixed density matrix for those two splits
        """
        densitys = []
        for kk in range(1, int(self.num_par/2)+1):
            dim_subsystem = np.prod(self.dims[:kk])
            matrix = 1/dim_subsystem * np.identity(dim_subsystem)
            densitys.append(matrix)
        return densitys

    def max_value_given_len_bipar(self, k: int):
        red_density = self.densitys[k-1]
        return self.func(red_density)

    def max_all(self, k_uniform=0) -> float:
        """
        calculate the maximal sum of given entanglement measurement 

        k_uniform : int, optional
            which bipartion of lenght k one consider,
            if k = 0:
            the sum of all possible bipartion and the corrosponding
            max value will be returned

        """

        if k_uniform == 0:
            return sum([self.max_value_given_len_bipar(min(len(bipar[0]), len(bipar[1])))
                        for bipar in
                        hf.get_all_bi_partions(self.num_par)])

        else:
            return sum([self.max_value_given_len_bipar(len(bipar[0])) for bipar in
                        hf.get_all_bi_partions(self.num_par)
                        if len(bipar[0]) == k_uniform])

    def amount_bipar(self):
        """
        calculate amount bipars for all ks
        eg:    dim = [2,2,2,2] :
            [4,3] we have 4 splits with 3|1 particles
            and 3 splits with 2|2 particles

        """
        lenghts = []
        for kk in range(1, 1+int(self.num_par/2)):
            ls = [1 for bipar in hf.get_all_bi_partions(self.num_par)
                  if len(bipar[0]) == kk]
            lenghts.append(len(ls))
        return lenghts

    def max_values(self,  k_uniform=0, full_vec=False) -> list:
        """
        return vector containing all maximal values for all possible
        bipartions (when k_uniform=0) else returns a vector with all
        possible bipartions having len(k_uniform), 
        if  full_vec = True: returns vec for all bipartions
        """

        if k_uniform == 0:
            max_val_all_bipar = [self.max_value_given_len_bipar(len(bipar[0]))
                                 for bipar in
                                 hf.get_all_bi_partions(self.num_par)]
            if full_vec:
                return np.array(max_val_all_bipar)
            else:
                return np.unique(max_val_all_bipar)

        else:
            max_val_k_bipar = [self.max_value_given_len_bipar(len(bipar[0]))
                               for bipar in
                               hf.get_all_bi_partions(self.num_par)
                               if len(bipar[0]) == k_uniform]
            if full_vec:
                return np.array(max_val_k_bipar)
            else:
                return np.unique(max_val_k_bipar)[0]

    def info(self, returne=False):
        string = f'For {self.num_par} particles with dimension {self.dims[0]}:\n'
        string += f'max {self.m} : {self.max_all():.3f}\n'
        for kk in range(1, 1+len(self.densitys)):
            max_val = self.max_value_given_len_bipar(kk)
            string += f'For k = {kk}: {max_val:.3f} ( {self.len[kk-1]} )  \n'

        if returne:
            return string
        else:
            print(string)



class state_dic_class(dict):
    """
    class to make sure that we can use both inputs:
        dic['001'] = x or
        dic[ [(0,0),(1,0),(2,1)] ] = x
    """

    def __init__(self, dic=None):
        super().__init__()
        if dic is not None:
            for kk, vv in dic.items():
                self.__setitem__(kk, vv)

    def __getitem__(self, keys):
        if isinstance(keys, str):
            return super().__getitem__(keys)
        if isinstance(keys, list):
            return super().__getitem__(hf.stateToString(keys))
        else:
            raise ValueError('only keystrings or state list allowed as keys')

    def __setitem__(self, key, item):
        if isinstance(key, str):
            super().__setitem__(key, item)
        else:
            super().__setitem__(hf.stateToString([key]), item)

    def items(self, as_string=True):
        if as_string:
            return super().items()
        else:
            ket = [hf.makeState(kk) for kk in super().keys()]
            return zip(ket, super().values())




class state_analyzer():

    def __init__(self, state: State,  weights=[], dim=[], precision=1e-3,
                 measure_kind='con'):
        self.dic = state_dic_class(state.state )
        self.norm = state.norm 
        self.pre = precision
        self.len = len(self.dic)  # number of summands
        if len(weights) != 0:
            assert self.len == len(weights), "lenght of weight should match"

        if len(dim) == 0:
            ni = len(list(self.dic.keys())[0])
            self.dim = ni * [1 + max([int(x) for ket in
                                               self.dic.keys() for x in ket])]
        else:
            self.dim = dim
        self.check_ancillas()
        self.num_par = len(self.dim)   # number particles
        self.func = entanglement_functions[measure_kind]
        self.allbipar = list(hf.get_all_bi_partions(self.num_par))
        self.max = entanglement_measure(self.dim)

    def check_ancillas(self):
        # check if given dic has more particles than dim (means we have ancilla)
        # (e.g dim = [2,2,1]) -> reduce to dim=[2,2] because 1 <-> ancilla
        num_ancillas = self.dim.count(1)
        if num_ancillas != 0:
            temp_dic = dict()
            for key, val in self.dic.items():
                temp_dic[key[:-num_ancillas]
                         ] = temp_dic.setdefault(key[:-num_ancillas], 0) + val
            self.dic = temp_dic
            self.dim = self.dim[:-num_ancillas]

    def state_vec(self, normalized = False)->dict:
        if any(isinstance(ampl, complex) for ampl in self.dic.values()):
            state_vec = np.zeros(np.product(self.dim), dtype=np.complex64)
        else:
            state_vec = np.zeros(np.product(self.dim))
        for idx, ket in enumerate(hf.get_all_kets_for_given_dim(self.dim, str)):
            try:
                state_vec[idx] = self.dic[ket]
            except KeyError:
                pass
        if all(state_vec == 0) is True:
            raise TypeError('State Vector is zero for all entrys, hmm?')
        self.norm = np.linalg.norm(state_vec)
        if normalized:
            return state_vec * 1/(self.norm)
        else: return state_vec 
        
    def ent(self):
        qstate = self.state_vec(normalized= True)
        def calc_ent(mat, par):
            red = ptrace(mat, par, self.dim)
            return self.func(red)
        self.con_vec = np.array([calc_ent(qstate, par[0]) for par in
                                 self.allbipar])
        self.c = sum(self.con_vec)
        self.con_vec_norm = 1/(self.max.max_values(full_vec=True))*self.con_vec
        return self.c

    def get_reduced_density(self, bipar, normalized = False):
        return ptrace(self.state_vec(normalized), bipar, self.dim)

    def print_red_densitys(self,k):
        self.calc_k_uniform() # for getting  k mask
        for bipar in np.array(self.allbipar)[self.k_mask[k-1]]:
            print(f'{bipar} :')
            print(self.get_reduced_density(bipar[0],True) )
        
            
    def string_wrapper(self,string:str,max_chars_per_line = 100)->str:
        new_string = '$'
        counts = 0
        for ss in string:
            if counts >= max_chars_per_line and  ss == '+':
                new_string += '$' + '\n' + '$' + ss
                counts = 0
            else:
                new_string += ss 
                counts += 1
    
        new_string+= '$'
        return new_string
            
    def state_string(self, dec=2, filter_zeros=False, with_color = False):
        if with_color:
            st_col = lambda strg, col: colored(strg,col)
        else: 
            st_col = lambda strg, col: strg
            
        ampls = [np.round(amp/self.norm, dec) for amp in self.dic.values()]
        most = np.round(
            abs(max(set([polar(aa)[0] for aa in ampls]), key=ampls.count)), dec)
        if most < 0.001:
            most = 1

        strs_plus, strs_min = [], []
        cet_counts = 0
        for ket, ampl in sorted(self.dic.items()):
            ampl = np.round(ampl/self.norm, dec)
            if ampl.real > 0 or ampl.imag != 0:
                ampl = np.round(1/most * ampl, dec)
                if abs(ampl) < float('10e-{}'.format(dec)):
                    col = 'red'
                else:
                    col = 'black'
                if abs(ampl) != 0 or not filter_zeros:
                    strs_plus.append(st_col(f' + {num_in_str(ampl)} \cdot', col))
                    strs_plus.append(st_col(r'\ket{{{0}}}'.format(ket), col))
                    cet_counts += 1
            else:
                ampl = np.round(1/most * ampl, dec)
                if abs(ampl) < float('10e-{}'.format(dec)):
                    col = 'red'
                else:
                    col = 'black'
                if abs(ampl) != 0 or not filter_zeros:
                    strs_min.append(
                        st_col(f' + {num_in_str(ampl,change_sign=True)} \cdot', col))
                    strs_min.append(st_col(r'\ket{{{0}}}'.format(ket), col))
                    cet_counts += 1

        strs_plus.append(st_col(' - (', 'black'))
        strs_plus += strs_min
        strs_plus.append(st_col(') ', 'black'))
        if filter_zeros:
            strs_plus.append(
                f'\n -- filtered {len(self.dic)-cet_counts} kets with amplitudes zero')
        string_end = "".join(strs_plus).replace('- ()', '').replace('(+', '(')
        string_end = self.string_wrapper( string_end)
        #string_end = r'\textbf{' + string_end + r' }'
        return string_end

    def calc_k_uniform(self) -> (int,np.array):
        try: 
            self.con_vec_norm
        except AttributeError:
            self.ent()
            
        # amount bipars for each k (as list)
        self.k_mask = []
        for kk in range(int(self.num_par/2)):
            self.k_mask.append([idx for idx, bi in enumerate(self.allbipar)
                                if min(len(bi[0]), len(bi[1])) == kk + 1])
        k = 0
        k_levels = []
        for kk in range(int(self.num_par/2)):
            bipar_kk = self.con_vec_norm[self.k_mask[kk]]
            # calc mean for each k for return
            k_levels.append(np.mean(bipar_kk))
            if all(np.isclose(bipar_kk, 1, rtol=self.pre)) is True:
                k += 1
        return k, k_levels

    def info_string(self, filter_zeros=False, ret=False, dec=3,
             with_color = True):
        info_string = self.state_string(filter_zeros=filter_zeros)
        
        return info_string


class analyser():

    def __init__(self, folder):
        self.folder = Path(folder)
        self.check_folder_name()
        self.get_all_states_in_folder_and_summary_file()
        self.dim = [ int(xx )for xx in str(self.summary['dim'] ) ]
    def check_folder_name(self):
        if len(list(Path(self.folder).rglob('*.json'))) == 0:
            raise ValueError(
                f'The given path {self.folder} does not contain jsons')
        elif self.folder.exists() is False:
            raise ValueError(f'The given path {self.folder} does not exist')

    def get_lossfunc_names(self):
        los = self.summary["loss_func"] 
        if los == 'ent':
            k_unif = str(self.summary["K"]) 
            return [r'$\mathcal{{L}}_{{ K ={0} }}$'.format(k_unif)]
        elif los == 'cr':
            thres = self.summary["thresholds"] 
            return [r'$1 - Countrate_{{ THR = {0} }}$'.format(thres[0]),
                    r'$1 - Fidelity_{{ THR ={0} }}$'.format(thres[1])]
        elif los == 'fid':
            thres = self.summary["thresholds"]
            return [r'$1 - Fidelity_{{ THR = {0} }}$'.format(thres[0]),
                    r'$1 - Countrate_{{ THR = {0} }}$'.format(thres[1])]
            
            

    def get_all_states_in_folder_and_summary_file(self):
        # iterate through all files in folder ending with json and append to files
        self.files = []
        for path in Path(self.folder).rglob('*.json'):
            if 'summary' in str(path):
                self.summary = convert_file_path_to_dic(path)
            elif 'best' in str(path):
                self.best = convert_file_path_to_dic(path)
            else:
                dic = convert_file_path_to_dic(path)
                dic['file_name'] = str(path.name)
                self.files.append(dic)
                
        self.files = sorted(self.files, key = lambda dic: dic["loss"])    
        for idx,file in enumerate(self.files):
            print(f'({idx}): {file["file_name"]}')
        
        self.loss_func_names = self.get_lossfunc_names()
        
    def plot_losses(self):
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
        def plotter(axe,num_loss):
            NUM_COLORS = len(self.files)
            cm = plt.get_cmap('gist_rainbow')
            axe.set_prop_cycle(color=[cm(1.*i/NUM_COLORS)
                              for i in range(NUM_COLORS)])
        
            for idx, st in enumerate(self.files):
                loss_history = st['history']
                min_edge = len(st['graph'])
                edges = [ii for ii in range(
                    min_edge, min_edge+len(loss_history))]
                edges.reverse()
                loss = [ll[num_loss] for ll in loss_history]
                axe.plot(edges, loss, label=idx, alpha=0.4, lw=3)
                axe.plot(edges, loss, alpha=0.6, lw=0.7, color="black")
            axe.set_xlabel("Amount of edges left")
            axe.set_ylabel(self.loss_func_names[num_loss])
            axe.invert_xaxis()
            axe.grid()
        fig, axs = plt.subplots(len(self.best['loss']), 1)
        try:
            for idx_loss,ax in enumerate(axs):
                plotter(ax,idx_loss)
        except TypeError:
            plotter(axs,0)
            
        plt.tight_layout()
        plt.show()
        
    def turn_graph_in_state(self,graph_dict: dict, thresholds_amplitudes = np.inf):
        graph = Graph(graph_dict,dimensions=self.dim) 
        graph.getState()
        return state_analyzer(graph.state)
    
    def info_statex(self, idx = 0):
        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure(str(idx))
        graph_ax = plt.subplot(gs[:, 0]) 
        state_ax =  plt.subplot(gs[0, 1])
        loss_ax = plt.subplot(gs[1, 1])
        state = self.files[idx]['graph']
        gp.graphPlot(Graph(state), ax_fig = (fig,graph_ax))
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
        ### state ###
        st_dic = self.files[idx]
        string_info = self.turn_graph_in_state(st_dic['graph']).state_string(filter_zeros=True)
        print(string_info)
        state_ax.text(0,1,string_info,fontsize = 20)
        state_ax.axis('off')
        ### loss ###
        loss_history = st_dic['history']
        min_edge = len(st_dic['graph'])
        edges = [ii for ii in range(
            min_edge, min_edge+len(loss_history))]
        edges.reverse()
        for ii in range(len(loss_history[0])):
            loss = [ll[ii] for ll in loss_history]
            loss_ax.plot(edges, loss, label=ii, alpha=0.4, lw=3)
            loss_ax.plot(edges, loss, alpha=0.6, lw=0.7, color="black")

        loss_ax.set_ylabel("Loss functions")

        loss_ax.set_xlabel("Amount of edges left")
        loss_ax.invert_xaxis()
        loss_ax.grid()
    def get_x_state(self, idx_files = 'all'):
        if idx_files == 'all':
            idx_files = [ ii for ii in range(len(self.files))]
        if isinstance(idx_files, int):
            idx_files = [idx_files]
        
        fig, axs = plt.subplots(len(idx_files), 2, sharex='col')
        fig.set_size_inches(15, 8)
        for idx,xth_file in enumerate(idx_files):
            st_dic = self.files[xth_file]
            string_info = self.turn_graph_in_state(st_dic['graph'])
            
            loss_history = st_dic['history']
            min_edge = len(st_dic['graph'])
            edges = [ii for ii in range(
                min_edge, min_edge+len(loss_history))]
            edges.reverse()
            for ii in range(len(loss_history[0])):
                loss = [ll[ii] for ll in loss_history]
                axs[idx][0].plot(edges, loss, label=ii, alpha=0.4, lw=3)
                axs[idx][0].plot(edges, loss, alpha=0.6, lw=0.7, color="black")
            if idx == int(len(idx_files)/2):
                axs[idx][0].set_ylabel("Loss functions")
            if idx == len(idx_files) - 1:
                axs[idx][0].set_xlabel("Amount of edges left")
            axs[idx][0].invert_xaxis()
            axs[idx][0].grid()
            axs[idx][1].text(0,0,string_info)
            axs[idx][1].axis('off')
        axs[0][0].get_shared_x_axes().join(axs[0][0], *axs[0,:])
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
        
        
        
path = r'C:/Users/janpe/Google Drive/6.Semester/Bachlorarbeit/Code/public_git/Theseus/data/conc_4-3/try'
#path = r'C:/Users/janpe/Google Drive/6.Semester/Bachlorarbeit/Code/public_git/Theseus/data/aklt_3/AKLT_3'
#path = r'C:/Users/janpe/Google Drive/6.Semester/Bachlorarbeit/Code/public_git/Theseus/data/ghz_346/try'
a = analyser(path)
a.plot_losses()
a.info_statex(0)
#a.get_x_state([ii for ii in range(3)])










