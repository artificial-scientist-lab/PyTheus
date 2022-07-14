from cmath import polar
import numpy as np
import theseus as th
import help_functions as hf
from termcolor import colored
from scipy.linalg import logm
import re


def find_idx_of_given_char(string: str, char: str) -> str:
    return [i for i, ltr in enumerate(string) if ltr == char]


def get_next_sign(string: str, sign=['-', '+', '−'], reverse=True):
    """
    get_next_sign( '000 + 123 - 122' ) =  '-' when reverse = True
                                       =  '+' when reverse = False
    """
    if reverse:
        rev_string = string[::-1]
        idxes = [rev_string.find(char) for char in sign]
        idxes = list(filter(lambda idx: idx != -1, idxes))
        try:
            return rev_string[min(idxes)]
        except ValueError:
            return "+"
    else:
        idxes = [string.find(char) for char in sign]
        try:
            return string[min(idxes)]
        except ValueError:
            return "+"


def multiply_brackets(state: str) -> str:
    """
    given a string state retuns string state but with no bracktes
    e.g.: |000> - ( |111> + |010>) --> |000> - |111> - |010>)

    Parameters
    ----------
    state : str
        a string of a given state

    Returns
    -------
    str
        corrected string state

    """
    bracket_idxs_left = find_idx_of_given_char(state, '(')
    bracket_idxs_right = find_idx_of_given_char(state, ')')

    if len(bracket_idxs_left) != len(bracket_idxs_right):
        raise ValueError(f"unmatched brackets: {len(bracket_idxs_left)}"
                         f" x ( and {len(bracket_idxs_right)} x ) ")
    # check if no bracket in bracket exists
    for ii in range(len(bracket_idxs_left) - 1):
        if bracket_idxs_right[ii] >= bracket_idxs_left[ii+1]:
            raise ValueError('No bracket in bracket is allowed')
    state_ls = list(state)  # for item assignment
    for left, right in zip(bracket_idxs_left, bracket_idxs_right):
        sign_for_bracket = get_next_sign(state[:left])
        if sign_for_bracket == '-':  # when sign is + no change needed
            for idx, char in enumerate(state[left:right]):
                if char == '-' or char == '−':
                    state_ls[left+idx] = '+'
                elif char == '+':
                    state_ls[left+idx] = '-'
    return "".join(state_ls)


def get_ampl_plus_ket(char: str, weight=None):
    pat = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(pat, re.VERBOSE)
    if weight is None:
        return rx.findall(char)
    else:
        ampl_ket = rx.findall(char)
        return [weight, ampl_ket[-1]]


def make_num(num):
    """
    turns a string to float, when num is of type complex leave unchanged
    """
    if isinstance(num, str):
        return float(num)
    else:
        return num


def update_dic(dic: dict, sign_ket: str, value_str: str) -> dict:

    try:
        if sign_ket == '+':
            dic[value_str[1]] = dic.setdefault(
                value_str[1], 0) + make_num(value_str[0])

        else:

            dic[value_str[1]] = dic.setdefault(
                value_str[1], 0) - make_num(value_str[0])

    except IndexError:
        if sign_ket == '+':
            dic[value_str[0]] = dic.setdefault(value_str[0], 0) + 1

        else:
            dic[value_str[0]] = dic.setdefault(value_str[0], 0) - 1
    return dic


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


def state_string_to_state_dic(state: str, weights=[]):
    state = multiply_brackets(state)

    split = re.split(r"\-|\+|\−", state.replace('\n', ''))
    state_dic = state_dic_class()
    count_dic = dict()
    # remove no digits things
    split = [ ket for ket in split if any(char.isdigit() for char in ket)]
    for idx, ket in enumerate(split):
        ket = ket.replace(' ', '')
        
        try:
            ampl_ket = get_ampl_plus_ket(ket, weights[idx])
        except IndexError:
            ampl_ket = get_ampl_plus_ket(ket)
        # find all indexes where the current ket appears (for double kets)
        idx_of_ket = [m.start() for m in re.finditer(ampl_ket[-1], state)]
        # save num of kets in a count dic
        try:
            count_dic[ampl_ket[-1]] += 1
        except KeyError:
            count_dic[ampl_ket[-1]] = 0
        # get correct idx for corrosponding ket and then get corrosponing sign
        sign_of_ket = get_next_sign(
            state[:idx_of_ket[count_dic[ampl_ket[-1]]]])
        state_dic = update_dic(state_dic, sign_of_ket, ampl_ket)
    return state_dic

    def get_rounded_weights(self, polar_form=True,  multiple_pi=True, dec=2):

        if polar_form:
            a = list(self.round_weight(polar_form=polar_form, dec=dec))
            if multiple_pi:
                a = [[num[0], np.round(num[1]/np.pi, dec)] for num in a]
                return [f'{num[0]}*exp({num[1]}*i*π)' for num in a]
            else:
                return [f'{num[0]}*exp({num[1]}*i)' for num in a]
        else:
            return list(self.round_weight(polar_form=polar_form))


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


# %%
class state1():

    def __init__(self, state_string,  weights=[], dim=[], precision=1e-3,
                 measure_kind='con'):
        if type(state_string) is dict:
            self.dic = state_dic_class(state_string)
        else:
            self.dic = state_string_to_state_dic(state_string, weights=weights)
        self.pre = precision
        self.len = len(self.dic)  # number of summands
        if len(weights) != 0:
            assert self.len == len(weights), "lenght of weight should match"

        if len(dim) == 0:
            ni = len(list(self.dic.keys())[0])
            self.dim =ni * [1 + max([int(x) for ket in
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
            temp_dic = state_dic_class()
            for key, val in self.dic.items():
                temp_dic[key[:-num_ancillas]
                         ] = temp_dic.setdefault(key[:-num_ancillas], 0) + val
            self.dic = temp_dic
            self.dim = self.dim[:-num_ancillas]

    def state_vec(self, normalized = False):
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
            red = th.ptrace(mat, par, self.dim)
            return self.func(red)
        self.con_vec = np.array([calc_ent(qstate, par[0]) for par in
                                 self.allbipar])
        self.c = sum(self.con_vec)
        self.con_vec_norm = 1/(self.max.max_values(full_vec=True))*self.con_vec
        return self.c

    def get_reduced_density(self, bipar, normalized = False):
        return th.ptrace(self.state_vec(normalized), bipar, self.dim)

    def print_red_densitys(self,k):
        self.calc_k_uniform() # for getting  k mask
        for bipar in np.array(self.allbipar)[self.k_mask[k-1]]:
            print(f'{bipar} :')
            print(self.get_reduced_density(bipar[0],True) )
        
    def spin_flip(self, *args):
        def flip(value: int):
            if value == 0:
                return 1
            if value == 1:
                return 0
            else:
                raise TypeError('spin flip only allowed for qubits!')
        for num in args:
            flipped_dic = state_dic_class()
            for kets, val in self.dic.items(False):
                flipped_ket = [tuple(x if x[0] != num-1 else (x[0], flip(x[1]))
                               for ket in kets for x in ket)]
                flipped_dic[flipped_ket] = val
            self.dic = flipped_dic

    def state_string(self, dec=2, filter_zeros=False):
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
                    col = 'white'
                if abs(ampl) != 0 or not filter_zeros:
                    strs_plus.append(colored(f' + {num_in_str(ampl)}', col))
                    strs_plus.append(colored(f'|{ket}>', col))
                    cet_counts += 1
            else:
                ampl = np.round(1/most * ampl, dec)
                if abs(ampl) < float('10e-{}'.format(dec)):
                    col = 'red'
                else:
                    col = 'white'
                if abs(ampl) != 0 or not filter_zeros:
                    strs_min.append(
                        colored(f' + {num_in_str(ampl,change_sign=True)}', col))
                    strs_min.append(colored(f'|{ket}>', col))
                    cet_counts += 1

        strs_plus.append(colored(' - (', 'white'))
        strs_plus += strs_min
        strs_plus.append(colored(') ', 'white'))
        if filter_zeros:
            strs_plus.append(
                f'\n -- filtered {len(self.dic)-cet_counts} kets with amplitudes zero')
        return ("".join(strs_plus)).replace('- ()', '').replace('(+', '(')

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

    def info(self, filter_zeros=False, ret=False, dec=3):
        self.ent()
        k, k_level = self.calc_k_uniform()
        info_str = "-----------------info-----------------\n"
        info_str += self.state_string(dec, filter_zeros=filter_zeros)
        info_str += '\n'
        info_str += "--------------------------------------\n"
        info_str += f'normalized by: 1/{round(self.norm,dec)}\n'
        info_str += f'total concurrence: {round(self.c,dec)}\n'
        info_str += 'concurrence vector: \n'
        for kk, mask in enumerate(self.k_mask):
            info_str += f'k = {kk+1} : {[round(ii,dec) for ii in self.con_vec_norm[mask]]}\n'
        info_str += "--------------------------------------\n"
        info_str += f'k-uniform: {k}\n'
        for ix, klev in enumerate(zip(k_level, self.max.len)):
            if np.isclose(klev[0], 1, rtol=self.pre):
                col = 'green'
            else:
                col = 'red'
            info_str += colored(
                f'k={ix + 1}: mean = {klev[0]:.3f} ({klev[1]}) \n', col)
        if ret:
            return info_str
        else:
            print(info_str)


# %%


class state():
    """
    given a state as a string and corrosponding weights for each ket as
    a string, one get several informaitons:
    atributes:
        - state.density is the corrosponding density matrix
        - state.con is the total concurrence
        - state.svd is the corrosponding Schmidt rank vector
        - state.H is the total entropy (Von Neumann)
    funcitons:
        - state.info(): prints all neccecarry information of the given state
        - info_string same as state.info() but instead of printing
          it returns infos as string

    e.g. ST = state("000+111",[0.5,-0.5]) to get 0.5 |000> -0.5 |111>

    """

    def __init__(self, state_string,  weights=[], dim=[]):
        state_string = state_string.replace('|', '').replace('>', '')\
            .replace(' ', '').replace('\n', '')
        state_splitted = state_string.split('+')
        self.len = len(state_splitted)  # number of summands
        self.num_par = len(state_splitted[0])  # number particles
        if len(dim) == 0:
            self.dim = self.num_par * \
                [1+max([int(i) for ket in state_splitted for i in ket])]
        else:
            self.dim = dim
        if len(weights) == 0:
            weights = np.ones(self.len)
        # sort weights and kets
        assert self.len == len(weights), "lenght of weight should match"
        ket_int = [int(ket) for ket in state_splitted]
        __, self.weights, self.state_string = zip(
            *sorted(zip(ket_int, weights, state_splitted)))
        self.state = top.makeState('+'.join(self.state_string))

        self.bipar = list(hf.get_all_bi_partions(self.num_par))

        self.q_state = self.state_2_qutip()
        self.density = self.q_state * self.q_state.dag()
        self.con = self.compute_concurrence()
        self.svd = self.getSVD()
        self.H = self.get_entropy()

    def state_2_qutip(self):
        """
        returns state given in state_def in topopt as a qutip object

        """
        state = self.state
        # eg: |000> + |111> : state_in_np = [[0, 0, 0], [1, 1, 1]]
        whole_state = [[x[i][1] for i in range(len(state[0]))] for x in state]
        state_list = [qt.tensor([qt.basis(self.dim[idx], quibit_i)
                                 for idx, quibit_i in enumerate(adden)])
                      for adden in whole_state]
        statet = state_list[0] * self.weights[0]
        for i in range(1, len(state_list)):
            statet += state_list[i] * self.weights[i]
        self.amount_of_basis_kets = np.product(self.dim)
        self.norm = statet.norm()
        self.weights = [weight * 1 / self.norm for weight in self.weights]
        return statet * 1 / self.norm

    def compute_concurrence(self, dec=4, p=0):
        def compute_concurrence_biparation(state, partion):
            reduced_density = (state.ptrace(partion))**2
            return np.sqrt(2 * (1 - min(reduced_density.tr(), 1)))
        self.c_list = [np.round(compute_concurrence_biparation(
            self.density, par[p]), dec)for par in self.bipar]
        return sum([compute_concurrence_biparation(self.density, par[p])
                    for par in self.bipar])

    def compute_concurrence_given_bipar(self, bipar, dec=2, p=1):
        def compute_concurrence_biparation(state, partion):
            reduced_density = (state.ptrace(partion))**2
            return np.sqrt(2 * (1 - min(reduced_density.tr(), 1)))
        return compute_concurrence_biparation(self.density, bipar)

    def get_entropy(self, dec=2):
        def von_neuman(state, partion):
            reduced_density = state.ptrace(partion)
            return qt.entropy_vn(reduced_density)
        self.entropy = [np.round(von_neuman(self.density, par[0]), dec)
                        for par in self.bipar]
        return sum([von_neuman(self.density, par[0]) for par in self.bipar])

    def getSVD(self):
        svd = []
        for idx, partion in enumerate(self.bipar):
            reduced_density = self.density.ptrace(partion[0])
            svd.append(np.linalg.matrix_rank(np.array(reduced_density.full())))
        return svd  # sorted(svd,reverse=True )

    def round_weight(self, dec=2, polar_form=False):
        if polar_form:
            for ww in self.weights:
                yield np.round(polar(ww), dec)
        else:
            for ww in self.weights:
                yield np.round(ww, dec)

    def get_rounded_weights(self, polar_form=True,  multiple_pi=True, dec=2):

        if polar_form:
            a = list(self.round_weight(polar_form=polar_form, dec=dec))
            if multiple_pi:
                a = [[num[0], np.round(num[1]/np.pi, dec)] for num in a]
                return [f'{num[0]}*exp({num[1]}*i*π)' for num in a]
            else:
                return [f'{num[0]}*exp({num[1]}*i)' for num in a]
        else:
            return list(self.round_weight(polar_form=polar_form))

    def get_reduced_density(self, partion):
        return self.density.ptrace(partion)

    def info(self, kets_weight_sep=False, polar_form=True, dec=2):
        print("---------------------------------")
        if kets_weight_sep:
            print(str([" |"+st+">" for st in self.state_string]).replace("[",
                  "").replace("]", "").replace("'", "").replace(",", " +"))
            print(self.get_rounded_weights(polar_form, dec=dec))
        else:
            print(str([str(w) + " |"+st+">" for st, w in
                       zip(self.state_string, self.get_rounded_weights(polar_form, dec=dec))])
                  .replace("[", "").replace("]", "").replace("'", "").replace(",", " +"))
        print("")
        print("total concurrence {} ".format(self.compute_concurrence()))
        print(f"SVD: {self.svd}")
        print(f"Concurrence: {self.c_list}")
        print(f"Was normalized by: 1/{self.norm}")
        print(
            f"amount of sums: {self.len} (theo: {self.amount_of_basis_kets})")
        print("---------------------------------")

    def info_string(self, kets_weight_sep=False, polar_form=True, dec=2):
        string = []

        string.append("---------------------------------")
        if kets_weight_sep:
            string.append("|"+self.state_string.replace("+", "> + |") + ">  ")
            string.append(self.get_rounded_weights(polar_form, dec=dec))
        else:
            string.append(str([str(w) + " |"+st+">" for st, w in zip(self.state_string,
                                                                     self.get_rounded_weights(polar_form, dec=dec))])
                          .replace("[", "").replace("]", "").replace("'", "").replace(",", " +"))
        string.append("")
        string.append("total concurrence {} ".format(
            self.compute_concurrence()))
        string.append(f"SVD: {self.svd}")
        string.append(f"Concurrence: {self.c_list}")
        string.append(f"Was normalized by: 1/{self.norm}")
        string.append(
            f"amount of sums: {self.len} (theo: {self.amount_of_basis_kets})")
        string.append("---------------------------------")
        return str("\n".join(string))


def start_info(edge_list, real):
    """
        plot and print infos for a given edge list
    """
    print('starting graph:')
    print('    #edges =', len(edge_list))
    gp.graphPlot(edge_list, scaled_weights=True, show=True, max_thickness=10)
    end, weigt_end = hf.graph_in_state_string(
        edge_list, np.ones(len(edge_list)), real=real)
    state(end, weigt_end).info(polar_form=not real)


if __name__ == '__main__':
    ss = state('|0000> + |0010> + |0011> + |0100>', [1,2,1,1.3 ])
    print(ss.get_reduced_density([2]) * 2/0.26007802)