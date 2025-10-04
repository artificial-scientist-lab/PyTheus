# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 10:51:51 2022

@author: janpe
"""
import os

from cmath import polar
from pytheus.fancy_classes import State, Graph
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pytheus.graphplot as gp
import pytheus.help_functions as hf
from scipy.linalg import logm
from pytheus.theseus import ptrace


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


def string_wrapper(string: str, max_chars_per_line=80, 
                   latex = '$',
                   line_break_char = '+' ) -> str:
    """
    wraps a string to a given line lenght 

    Parameters
    ----------
    string : str
        string has to be wrapped.
    max_chars_per_line : TYPE, optional
        max nums of char in one line. The default is 80.

    Returns
    -------
    str
        wrapped string

    """
    new_string = latex
    counts = 0
    for ss in string:
        if counts >= max_chars_per_line and ss == line_break_char:
            new_string += ss + latex + '\n' + latex
            counts = 0
        else:
            new_string += ss
            counts += 1

    new_string += latex
    return new_string


def num_in_str(num, dec=2, change_sign=False) -> str:
    """
    make sure to convert a number in a proper Latex string
    output depending if it is complexe or not

    Parameters
    ----------
    num : TYPE
        number to convert
    dec : TYPE, optional
        dec for rounding. The default is 2.
    change_sign : TYPE, optional
        needed in state analyzer, changes sign of num The default is False.

    Returns
    -------
    Str
        returns num in Latex string format

    """
    if np.round(num.imag, dec) != 0:
        polar_num = polar(num)
        if not change_sign:
            num = np.round([polar_num[0], polar_num[1]/np.pi],  dec)
        else:
            num = np.round([-polar_num[0], polar_num[1]/np.pi],  dec)
        return '{} \cdot e^{{ {} \cdot i \cdot \pi }}'.format(*num).replace('1.0 * ', '')
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
        """
        a class to calculate a maximal values for a given measure kind
        of entanglment:
            - all reduced density being maximal mixed 
            - best values for a given bipartion
            - the sum of best values for a given k-bipartion
            - all maximal values
        Parameters
        ----------
        dimensions : list
            Ddimension list of system
        measure_kind : func, optional
            measure function according to dict above . The default is 'con'.

        Returns
        -------
        None.

        """
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
                 measure_kind='entropy'):
        self.dic = state_dic_class(state.state)
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
        self.max = entanglement_measure(self.dim, measure_kind=measure_kind)
        self.measure_kind = measure_kind
    def check_ancillas(self):
        """
        check if given dic has more particles than dim (means we have ancilla)
        (e.g dim = [2,2,1]) -> reduce to dim=[2,2] because 1 <-> ancilla

        Returns
        -------
        None.

        """

        num_ancillas = self.dim.count(1)
        if num_ancillas != 0:
            temp_dic = dict()
            for key, val in self.dic.items():
                temp_dic[key[:-num_ancillas]
                         ] = temp_dic.setdefault(key[:-num_ancillas], 0) + val
            self.dic = temp_dic
            self.dim = self.dim[:-num_ancillas]

    def state_vec(self, normalized=False) -> np.array:
        """
        calculate the lineare algebra vector for the given state.
        e.g. |00> + |10> --> [ [1],[0],[1],[0] ] 

        Parameters
        ----------
        normalized : bool, optional
            if state vec should be normalized. The default is False.

        Raises
        ------
        TypeError
            when state vec is zero vector

        Returns
        -------
        np.array
            state vector

        """
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
        else:
            return state_vec

    def ent(self):
        """
        calculate total sum over all bipartions for given ent function 
        in self.func

        Returns
        -------
        float
            sum of ent measure for all bipartions.

        """
        qstate = self.state_vec(normalized=True)

        def calc_ent(mat, par):
            red = ptrace(mat, par, self.dim)
            return self.func(red)
        self.con_vec = np.array([calc_ent(qstate, par[0]).real for par in
                                 self.allbipar])

        self.c = sum(self.con_vec)

        def normalizer(x, min_val):
            # little function to make sure that we normalize values
            # for dense : x = min_val -> 1 and x = 1 -> 0
            # for concurrence or entropy just diveded by highest poss. value
            if self.measure_kind == 'dense':
                alpha = 1/(min_val-1)
                return alpha * x - alpha
            else:
                return x/min_val
        
        self.con_vec_norm = np.array([normalizer(calc, best) for calc, best in zip(self.con_vec,
                                                                                   self.max.max_values(full_vec=True))])

        return self.c

    def get_reduced_density(self, bipar, normalized=False) -> np.array:
        """

        Parameters
        ----------
        bipar : list/tuple
            of given bipartion, e.g [0,1] for split AB|CD
        normalized : TYPE, optional
            if normalizing state vector The default is False.

        Returns
        -------
        np.array
            reduced density matrix.

        """
        return ptrace(self.state_vec(normalized), bipar, self.dim)

    def print_red_densitys(self, k):
        self.calc_k_uniform()  # for getting  k mask
        for bipar in np.array(self.allbipar)[self.k_mask[k-1]]:
            print(f'{bipar} :')
            print(self.get_reduced_density(bipar[0], True))

    def state_string(self, dec=2, filter_zeros=False, with_color=False):
        """
        get a string of the state in Latex format

        Parameters
        ----------
        dec : int, optional
            for decimal roundings of amplitudes. The default is 2.
        filter_zeros : bool, optional
            if one displays the kets with amplitude = 0 after rounding.
            The default is False.
        with_color : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        string_end : str
            Latex string of state

        """
        if with_color:
            def st_col(strg, col): return r' \color{{{0}}} '.format(col) + strg
        else:
            def st_col(strg, col): return strg

        ampls = [np.round(amp/self.norm, dec) for amp in self.dic.values()]
        most = np.round(
            abs(max(set([polar(aa)[0] for aa in ampls]), key=ampls.count)), dec)
        if most < 0.001:
            most = 1
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
                    strs_plus.append(
                        st_col(f' + {num_in_str(ampl)} \cdot', col))
                    strs_plus.append(r'|{0} \rangle'.format(ket))
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
                    strs_min.append(r'| {0} \rangle'.format(ket))
                    cet_counts += 1
        try:
            strs_plus[0] = strs_plus[0].replace('+', '')
        except IndexError:
            pass

        strs_plus.append(st_col(' - (', 'black'))
        strs_plus += strs_min
        strs_plus.append(st_col(') ', 'black'))

        string_end = "".join(strs_plus).replace('- ()', '')
        string_end = string_wrapper(string_end[1:])

        return string_end

    def calc_k_uniform(self) -> (int, np.array):
        """
        get k uniform of state and normalized vector for each k-bipartion
        where 1 means  maximal mixed and 0 seperable for that bipartion
        e.g. :
            return is ( 1 , [ [1,1,1,1], [0,1,1] ] ) means state is 1-uniform
            and for all bipartions with cardinality 2 also maximal entangeld
            except for the first split AB|CD (= 0) it is separable

        Returns
        -------
        string_end : (int, np.array)
            num of k-uniform for given state and 
            normalized array for all bipartions

        """

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

    def info_string(self, *args, **kwargs):
        """
        returns a string with all infos specified in args and kwargs

        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        info_string : str
            info string

        """
        options = {
            'filter_zeros': False,
            'dec': 2,
            'with_color': False}
        if options['with_color']:
            def st_col(strg, col): return r'\color{0}'.format(col) + strg
        else:
            def st_col(strg, col): return strg

        if len(args) == 0:
            args = ['norm']
        options.update(kwargs)
        self.ent()
        info_string = ''
        if any(x in args for x in ['k', 'K', 'k-uniform']):

            k, k_level = self.calc_k_uniform()
            info_string += f'\nk-uniform: {k}'
            for ix, klev in enumerate(zip(k_level, self.max.len)):
                if np.isclose(klev[0], 1, rtol=self.pre):
                    col = 'green'
                else:
                    col = 'red'
                info_string += st_col(
                    f'\nk={ix + 1}: mean = {klev[0]:.3f} ({klev[1]})', col)

        if any(x in args for x in ['concurrence', 'ent']):
            info_string += f'\ntotal {self.measure_kind}: {round(self.c,options["dec"])}'
            info_string += f'\n {self.measure_kind} vector:'
            for kk, mask in enumerate(self.k_mask):
                k_info = f'\n k = {kk+1} : {[round(ii,options["dec"]) for ii in self.con_vec_norm[mask]]}'
    
                info_string += string_wrapper(k_info,55,line_break_char = ',',latex='',)
        if any(x in args for x in ['n', 'norm']):
            info_string += f'\n normalized by: 1/{round(self.norm,options["dec"])}'

        return info_string


class analyser():

    def __init__(self, folder, only_pm=False):
        if isinstance(folder, str):
            self.folder = Path(folder)
        else:
            self.folder = folder
        self.check_folder_name()
        self.get_all_states_in_folder_and_summary_file()
        self.dim = self.summary['dimensions']
        self.imaginary = self.summary['imaginary']
        self.only_pm = only_pm

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
        else:
            return [f'loss {ii}' for ii in range(len(self.best['history'][0]))]

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

        self.files = sorted(self.files, key=lambda dic: dic["loss"])
        for idx, file in enumerate(self.files):
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

        def plotter(axe, num_loss):
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
            axe.legend()
            axe.grid()
        fig, axs = plt.subplots(len(self.best['loss']), 1)
        try:
            for idx_loss, ax in enumerate(axs):
                plotter(ax, idx_loss)
        except TypeError:
            plotter(axs, 0)

        plt.tight_layout()
        plt.show(block=False)
   

    def turn_dic_in_graph_state(self, graph_dict: dict, thresholds_amplitudes=np.inf):
        """
        returns a GRaph object and a state_analyzer object for given state

        Parameters
        ----------
        graph_dict : dict
            dict representing a graph.
        thresholds_amplitudes : TYPE, optional
            DESCRIPTION. The default is np.inf.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.only_pm:
            for ket, ampl in graph_dict.items():
                if self.summary['imaginary'] is False:
                    graph_dict[ket] = 1 if ampl > 0 else -1
                else:
                    graph_dict[ket] =  [1,ampl[1]] if ampl[0] > 0 else [-1,ampl[1]] 

        graph = Graph(graph_dict, dimensions=self.dim,
                      imaginary=self.imaginary)
        if self.imaginary is not False:
            graph.toCartesian()
        graph.getState()

        return graph, state_analyzer(graph.state, dim=self.dim)

    def all_perfect_matchings_to_pdf(self, nth_file=0, other_weights=[],
                                                   given_ket_only="", show=True, row_len=True,
                                                   dpi=100) -> None:
        """


        Parameters
        ----------
        state_sys : dict
            DESCRIPTION.
        other_weights : TYPE, optional
            DESCRIPTION. The default is [].
        given_ket_only : TYPE, optional
            DESCRIPTION. The default is "".
        show : TYPE, optional
            DESCRIPTION. The default is True.
        row_len : TYPE, optional
            DESCRIPTION. The default is True.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """

        state_sys = self.files[nth_file]
        pt = self.folder
        pt = pt / 'state_pdfs' / 'tmp'  # move data directory
        pt.mkdir(parents=True, exist_ok=True)
        savepath = str(pt)
        for f in os.listdir(savepath):
            os.remove(os.path.join(savepath, f))

        graph, __ = self.turn_dic_in_graph_state(state_sys['graph'])

        graph.getStateCatalog()
        plt.ioff()
        cat = graph.state_catalog
        if len(given_ket_only) != 0:
            if type(given_ket_only) == str:
                given_ket_only = [given_ket_only]
            for ket_i in given_ket_only:
                try:
                    th_state = hf.stringToTerm(ket_i)
                    cat = {th_state: cat[th_state]}
                except KeyError:
                    raise ValueError(
                        "The given Ket must be wrong, Graph does not have it")
        # calculation how many rows
        if row_len:
            row_len = max([len(pm_graphs) for pm_graphs in cat.values()])

        # calculation for coloring background
        colors = []
        for kk, vv in cat.items():
            total_weight = 0
            for ii, cover in enumerate(vv):
                weights_for_cover = [graph[edge] for edge in cover]
                total_weight += np.prod(weights_for_cover)

            if round(total_weight.real, 2) == 0:
                colors.append((1, 0.89, 0.77))
            else:
                colors.append((1, 1, 1))
        num_kets = len(self.dim) - self.dim.count(1)  # eliminate ancillas
        # determinate fontsize:
        fontsize = max(20, 33-num_kets-row_len)
        amount_pm = len(cat.items())
        for idx, (kk, vv) in enumerate(cat.items()):
            ket_string = "".join([str(int(k[1])) for k in kk[:num_kets]])
            total_weight = 0
            figgy, ax = plt.subplots(1, row_len,
                                     figsize=(row_len*800/dpi, 800/dpi),
                                     dpi=dpi)
            
            if row_len == 1:
                ax = [ax]  # make ax subscriptable when lenght one
            for ii, cover in enumerate(vv):
                weights_for_cover = [graph[edge] for edge in cover]
                total_weight += np.prod(weights_for_cover)
                figgy = gp.graphPlot(Graph(cover, weights=weights_for_cover),
                                     show=False, weight_product=True,
                                     show_value_for_each_edge=True,
                                     ax_fig=(figgy, ax[ii]), fontsize=fontsize
                                     )

            title = r'$  {0} \ket{{ {1} }} $'.format(
                    num_in_str(total_weight), ket_string)

            props = dict(boxstyle='round', facecolor='lightgrey')
            figgy.suptitle(title, y=0.99, fontsize=35, bbox=props)
            figgy.patch.set_facecolor(colors[idx])

            for ii in range(len(ax)):
                ax[ii].axis('off')
            plt.tight_layout()
            figgy.savefig(os.path.join(savepath, ket_string + '(' + str(ii) +
                                       ')' + ".jpeg"),
                          dpi=dpi, bbox_inches='tight',
                          facecolor=figgy.get_facecolor(), edgecolor='none')
            x = int(10*idx/amount_pm)
            print("state {}: [{}{}] {}/{}".format(nth_file,"#"*x, "."*(10-x), idx, amount_pm),
                  end='\r', flush=True)
            plt.close(figgy)

        self.convert_img_to_pdf(savepath, 1, self.files[nth_file]['file_name'])
        plt.ion()

    def convert_img_to_pdf(self, savepath, row_len, file_name='pm_pdf', show=False):
        from PIL import Image

        images = [
            Image.open(os.path.join(savepath, f))
            for f in os.listdir(savepath)
        ]

        def pil_grid(images, max_horiz=np.iinfo(int).max):
            def get_grid(func):
                n_images = len(images)
                n_horiz = func(n_images, max_horiz)
                h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)

                for i, im in enumerate(images):
                    h, v = i % n_horiz, i // n_horiz
                    h_sizes[h] = max(h_sizes[h], im.size[0])
                    v_sizes[v] = max(v_sizes[v], im.size[1])
                h_sizes, v_sizes = np.cumsum(
                    [0] + h_sizes), np.cumsum([0] + v_sizes)
                im_grid = Image.new(
                    'RGB', (h_sizes[-1], v_sizes[-1]), color='white')
                for i, im in enumerate(images):
                    im_grid.paste(
                        im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
                return im_grid
            try:
                return get_grid(min)
            except IndexError:
                return get_grid(max)
        pt = self.folder
        pt = pt / 'state_pdfs'
        name = str(file_name)
        if self.only_pm:
            name = 'all_w_1__' + name
        if not name.endswith('.pdf'):
            name += '.pdf'

        pt.mkdir(parents=True, exist_ok=True)
        filepath = pt / "".join(name)

        whole_image = pil_grid(images, row_len)
        whole_image.save(str(filepath))

        if show:
            plt.figure()
            plt.imshow(whole_image.convert('RGB'))
            plt.axis('off')

    def info_statex(self, idx=0, infos=[], filter_zeros=False, figsize=(14, 8)):

        ### plot setup ###

        fig = plt.figure(str(idx), figsize=figsize)
        graph_ax = fig.add_subplot(1, 2, 1)
        state_ax = fig.add_subplot(2, 2, 2)
        loss_ax = fig.add_subplot(2, 2, 4)

        graph, state = self.turn_dic_in_graph_state(self.files[idx]['graph'])
        st_dic = self.files[idx]
        ### Graph ###

        gp.graphPlot(graph, ax_fig=(fig, graph_ax), show=False)
        try:
            self.summary['target_state'] = 1
        except:
            pass

        ### state + infos ###

        st_string = state.state_string(filter_zeros=filter_zeros)
        print(st_string)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fs = max(19 - st_string.count('\n'), 10)

        text = state_ax.annotate(st_string,
                                 xy=(0.5, 1.05), xycoords=("data", 'axes fraction'),
                                 xytext=(0, -20), textcoords='offset points',
                                 ha="center", va="top", fontsize=fs,
                                 bbox=props)

        props = dict(boxstyle='round', facecolor='grey', alpha=0.2)
        state_ax.annotate(state.info_string(*infos, filter_zeros=filter_zeros),
                          xy=(0.5, 0.), xycoords=text,
                          xytext=(0, -20), textcoords='offset points',
                          ha="center", va="top", fontsize=int(0.8*fs),
                          bbox=props)

        state_ax.axis('off')

        ### loss ###
        loss_history = st_dic['history']
        min_edge = len(st_dic['graph'])
        edges = [ii for ii in range(
            min_edge, min_edge+len(loss_history))]
        edges.reverse()
        for ii in range(len(loss_history[0])):
            loss = [ll[ii] for ll in loss_history]
            loss_ax.plot(
                edges, loss, label=self.loss_func_names[ii], alpha=0.4, lw=3)
            loss_ax.plot(edges, loss, alpha=0.6, lw=0.7, color="black")

        fs_loss_plot = int(fs*0.7)  # make fontsize smaller for loss_plot
        loss_ax.set_ylabel("Loss functions", fontsize=fs_loss_plot)
        loss_ax.set_xlabel("Amount of edges left", fontsize=fs_loss_plot)
        loss_ax.tick_params(axis='x', labelsize=fs_loss_plot)
        loss_ax.tick_params(axis='y', labelsize=fs_loss_plot)
        loss_ax.legend(fontsize=fs_loss_plot)
        loss_ax.invert_xaxis()
        loss_ax.grid()
        plt.tight_layout()
        return state


def input_with_check_ints(question: str, restriciton=[]):
    counter = 0
    while counter < 3:
        try:
            user_input = input(question)
            if '-' not in user_input:
                user_input = [int(user_input)]
            else:
                user_input = [int(xx) for xx in user_input.split('-')]

            if len(restriciton) != 0:

                if all(restriciton[0] <= xx and xx <= restriciton[1] for xx in user_input):
                    return user_input
                else:
                    raise ValueError()

            else:
                return user_input
        except:
            counter += 1
        print(f'inproper input!: try again ({3-counter} trys left)')

    import sys
    sys.exit()


def check_which_summary_file(path):
    folder_list = [xx for xx in path.iterdir() if xx.is_dir()]
    # when only one folder exists we take this folder oc
    if len(folder_list) == 1:
        return folder_list[0]
    for idx, ff in enumerate(folder_list):
        aa = convert_file_path_to_dic(ff / 'summary.json')
        print('\n' + ff.name + f'  ({idx=})' + '\n')
        for key, val in aa.items():
            print(f'  - {key} : {val}')
            last_key = key
        print(len(f'  - {last_key} : {aa[last_key]}')*'-')

    folder_idx = input_with_check_ints(f'Choose one folder index ( 0 - {idx} ):',
                                       restriciton=[0, idx]
                                       )[0]
    return folder_list[folder_idx]


def walk(path):
    for p in Path(path).iterdir():
        if p.is_dir():
            yield p
            yield from walk(p)
            continue


def convert_input_path(path: str):

    # first check if no path given then we have to choose system we want to analyze
    current_dir = os.getcwd()
    if not str(current_dir).endswith('output'):
        output_dir = Path(current_dir) / 'output'
    else:
        output_dir = Path(current_dir)

    if path is None or len(path) == 0:
        folder_list = list(output_dir.iterdir())
        for idx, folder in enumerate(folder_list):

            print(f'( {idx} ) : {folder}'.replace(str(output_dir), ''))
        idx_folder = input_with_check_ints(f'Choose one folder to analyse ( index: 0 - {idx} )\n',
                                           restriciton=[0, idx])[0]
        path = Path(output_dir) / str(folder_list[idx_folder])

    path = Path(path)
    # first check if given path is an absolut path and if exists then return
    for folder in walk(output_dir):
        if str(path) in str(folder):
            path = folder
            break

    if path.is_absolute() and any(pp.suffix == '.json' for pp in path.iterdir()):
        if path.exists():
            return path
        else:
            raise ValueError('Given absolute Path does not exist')

    # second one check if name corresponds to one of the folder in output

    else:
        return check_which_summary_file(path)


def get_analyse(which_directory, all_weights_plus_minus_one=False,
                which_infos=['norm'], create_perfect_machting_pdf=False):

    path = convert_input_path(which_directory)
    a = analyser(path, only_pm=all_weights_plus_minus_one)
    #a.plot_losses()
    indexs = input_with_check_ints(
        f'which state? (int from 0 - {len(a.files)-1} or idx_1 - idx_2): ',
        [0, len(a.files)-1])
    if len(indexs) == 1:
        indexs.append(indexs[0])
    for idx in range(indexs[0],indexs[1]+1):
        a.info_statex(idx, infos=which_infos)
        if create_perfect_machting_pdf:
            a.all_perfect_matchings_to_pdf(idx)
    plt.show()
    input('press arbitary key to exit')
    return 0



def save_graph(file_loc, pm=False, fontsize = 12):
    
    
    def get_figsize(columnwidth):
        """Parameters:
          - wf [float]:  width fraction in columnwidth units
          - hf [float]:  height fraction in columnwidth units.
                         Set by default to golden ratio.
          - columnwidth [float]: width of the column in latex. Get this from LaTeX 
                                 using \showthe\columnwidth
        Returns:  [fig_width,fig_height]: that should be given to matplotlib
        """
        fig_width_pt = columnwidth  # Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27               # Convert pt to inch
        golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = fig_width*golden_mean      # height in inches
        return [fig_width, fig_height]
    
    def turn_dic_in_graph(graph_dict: dict,only_pm= False, thresholds_amplitudes=np.inf):
        """
        returns a GRaph object and a state_analyzer object for given state

        Parameters
        ----------
        graph_dict : dict
            dict representing a graph.
        thresholds_amplitudes : TYPE, optional
            DESCRIPTION. The default is np.inf.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        imaginary = False
        if only_pm:
            for ket, ampl in graph_dict.items():
                if isinstance(ampl,list):
                    imaginary = 'polar'
                    graph_dict[ket] = [1,0] if ampl[0] > 0 else [-1,0]
                else:
                    graph_dict[ket] = 1 if ampl > 0 else -1
        for ket, ampl in graph_dict.items():
            if isinstance(graph_dict[ket],list):
                imaginary = 'polar'
        graph = Graph(graph_dict,imaginary=imaginary)
        if imaginary is not False:
            graph.toCartesian()
        graph.getState()
        return graph
    
    params = {'backend': 'ps',
          'axes.labelsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,}
    
    plt.rcParams.update(params)

    for folder in [pp for pp in os.listdir(file_loc) if '.' not in pp ]:

        cur_file_loc = Path(file_loc) / folder
        plots = 0
        for xx in os.listdir(cur_file_loc):
            if xx.endswith('.json'):
                plots += 1
        figgy, ax = plt.subplots(1,plots,figsize=get_figsize(418.2555))
    
        for idx,ii in enumerate([xx for xx in os.listdir(cur_file_loc) if xx.endswith('.json')]):

            graph = turn_dic_in_graph(convert_file_path_to_dic(Path(cur_file_loc ) / ii)['graph'],pm)
            try:
                fig = gp.graphPlot(graph, show=False,ax_fig=(figgy, ax[idx] ), fontsize=fontsize,
                               max_thickness= 5, zorder= 20,markersize=10)
            except TypeError:
                fig = gp.graphPlot(graph, show=False,ax_fig=(figgy, ax ), fontsize=fontsize,
                               max_thickness= 5, zorder= 7,markersize=12)
        plt.tight_layout()
        save_path = Path(file_loc ) / (folder + '.pdf')
        fig.savefig(str(save_path), format='pdf', bbox_inches='tight')
            
if __name__ == '__main__':
    save_graph(r'C:/Users/janpe/Downloads/test_th/paper',pm=True)
  
