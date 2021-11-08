#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 07:02:11 2021

@author: alejomonbar
"""
from typing import List, Union, Optional

import numpy as np
import itertools
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import sympy as sp


class Graph:
    """The algorithm THESEUS [1] automates the design of quantum optical experiments,
    which is based on an abstract physics-inspired representation. We use it to
    discover several previously unknown experimental configurations to realize
    quantum states and transformations in the challenging high-dimensional and
    multi-photonic regime, such as the generation of high-dimensional GHZ states.

    References:
        [1]: 'Theseus',
        `Conceptual understanding through efficient automated design of quantum
        optical experiments <https://arxiv.org/abs/2005.06443>`
    """
    def __init__(self, dimensions: List[int]) -> None:
        """
        Args:
            dimensions : dimensions of the graph

        Returns
        -------
        None
            DESCRIPTION.

        """
        self._num_vertices = len(dimensions)
        self._max_modes = np.array(dimensions) 
        self._num_vars = self.num_init_vars()
        self._triggerableState = self.whole_state()
        self._combinations = self.iterables()
        
    def tensor_weights(self):
        x = sp.symbols(f"x(:{self._num_vars})")
        n = self._num_vertices
        localDim = self._max_modes 
        max_mode = max(localDim)
        weights = sp.MutableDenseNDimArray(np.zeros((n, n, max_mode, max_mode)))
        ii = 0 # counting variables
        for i in range(n - 1): # node i
            for j in range(i + 1, n): # node j
                for c1 in range(localDim[i]): # c1 edge color for node ith
                    for c2 in range(localDim[j]): # c2 edge color for node jth
                        weights[i, j, c1, c2] = x[ii]
                        ii += 1
        return weights        

    def num_init_vars(self):
        n = self._num_vertices
        localDim = self._max_modes 
        num_vars = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                num_vars += localDim[i] * localDim[j]
        return num_vars
    
    def whole_state(self):
        n = self._num_vertices
        state = [[x] for x in range(self._max_modes[0])]
        for i in range(1, n):
            state = [x + [y] for x in state for y in range(self._max_modes[i])]
        return np.array(state)

    def fidelity(self, desired_state: List[List[int]]) -> sp.core.mul:
        tensor_vars = self.tensor_weights()
        NormalisationConstant = []
        for state in self._triggerableState:# create whole state
            sum_of_w = 0   
            for comb in self._combinations:
                mult = 1
                for ii in np.array(comb).reshape(self._num_vertices//2, -1):
                     mult *= tensor_vars[ii[0], ii[1], state[ii[0]], state[ii[1]]] 
                sum_of_w += mult
            NormalisationConstant.append(sum_of_w)
        
        Normalisation = np.sum(np.array(NormalisationConstant) ** 2)  
        TargetEquation = []
        for state in desired_state:# Create the target state
            sum_of_w = 0   
            for comb in self._combinations:
                mult = 1
                for ii in np.array(comb).reshape(self._num_vertices//2, -1):
                     mult *= tensor_vars[ii[0], ii[1], state[ii[0]],state[ii[1]]] 
                sum_of_w += mult
            TargetEquation.append(sum_of_w)
        Fidelity = np.sum(np.array(TargetEquation)) ** 2 / (len(TargetEquation) * Normalisation)
        return Fidelity
    
    def minimize(self, desired_state, not_to_include=[], initial_weights=[], alpha=0):
        """
        

        Parameters
        ----------
        alpha : float
            Regularization constant.

        Returns
        -------
        None.

        """
        num_vars = self.num_vars - len(not_to_include)
        Fidelity = lambda x: self.fidelity(x, desired_state, not_to_include)
        if len(initial_weights) == 0:
            initial_weights = 2 * np.random.rand(num_vars) - 1
        loss = lambda x: (1 - Fidelity(x)) + alpha * np.sum(np.abs(x))
        sol = optimize.minimize(loss, initial_weights, tol=1e-2)
        return sol, sol.x
        
    def ampltiudes(self, vars_):
        """
        Returns the w's of the problem: for example the GHZ of 4 qubits. This function
        return's: w_|0000> and w_|1111>

        Parameters
        ----------
        vars_ : List of tuples
            value of the w's variables based on the minimization problem.

        Returns
        -------
        list
            As in the example above [w_|0000>, w_|1111>] 

        """
        return [i.subs(vars_) for i in self.TargetEquation]
        
    def topological_optimization(self, desired_state, initial_weights=[], alpha=0.5, loss_min=1e-2,
                                 max_counts=100, w_limit=1):
        sol, vars_ = self.minimize(desired_state, initial_weights, alpha=0)
        weight_last = sol.x
        count = 0
        not_to_include = []
        vars_list = list(range(self.num_vars))
        while count < max_counts:
            ith_rid = np.random.choice(vars_list)
            not_to_include.append(ith_rid)
            initial_weights = [i for n, i in enumerate(weight_last) if vars_list[n] != ith_rid]
            sol, vars_ = self.minimize(desired_state, not_to_include, initial_weights, alpha=alpha)
            w_sum = np.abs(sol.x).sum()
            if (sol.fun < loss_min) and (w_sum < w_limit):
                count = 0
                weight_last = sol.x
                vars_list.remove(ith_rid)
            else:
                not_to_include.remove(ith_rid)
                count += 1
            print(f"The solution:{sol.fun} - count:{count} - w_sum:{w_sum}")
            print(vars_list)
        return sol, self.tensor_weights(sol.x, not_to_include)
        
    def iterables(self):
        n = self._num_vertices
        combinations = list(itertools.combinations(range(n),2))
        Comb = []
        for comb in itertools.combinations(combinations, n // 2):
            new = []
            for ii in comb:
                flag = any(item in new for item in ii)
                if not flag:
                    new += list(ii)
                else:
                    break
            if len(new) == n:
                Comb.append(new)
        return Comb
    
    def plot(self, optimization=False, filename=None):
        # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if optimization:
            if "weights" not in dir(self):
                print("--------------------------------------------------------------")
                print("Error: first, you should execute 'topological_optimization'")
                print("--------------------------------------------------------------")
                return
        n = self.num_vertices
        dim_max = max(self.max_modes)
        colorsfun = plt.cm.get_cmap("Set1",lut=dim_max)
        colors = [colorsfun(i) for i in range(dim_max)]
        angle = np.linspace(0,2*np.pi*(n-1)/n,n)
        x = np.cos(angle)
        y = np.sin(angle)
        fig, ax = plt.subplots(figsize=(5,5))
        circle = plt.Circle((0, 0), 1.2, color='sienna', alpha=0.1,edgecolor="black")
        ax.add_patch(circle)
        ref = 0.3 # Separation between weights
        edge_pos = np.linspace(-ref,ref,dim_max)
        r = 0
        for i in range(dim_max-1):
            for j in range(i+1,dim_max):
                if i != j:
                    r += 1
                    self.edge(ax, x, y, ref + 0.2*r, [colors[i], colors[j]], i, j, optimization)
                    self.edge(ax, x, y, -ref - 0.2*r, [colors[j], colors[i]], j, i, optimization)
        for i in range(dim_max):
            self.edge(ax, x, y, edge_pos[i], colors[i], i, i, optimization)

        for i in range(n):
            ax.plot(x[i], y[i], "o", markersize=20, markeredgecolor="black", markerfacecolor="white")
            ax.text(x[i] - 0.025, y[i] - 0.01, str(i))
        ax.axis('off')
        if isinstance(filename, str):
            fig.savefig(filename)
        
    def edge(self, ax, x, y, a, color, state1, state2, optimization):
        n = len(x)
        dims = self.max_modes
        for i, j in itertools.combinations(range(n), 2):
            h = np.sqrt((x[j] - x[i])**2 + (y[j] - y[i])**2)
            r = h * np.linspace(0, 1, 50)
            fr = a * ((1 - r/h)**2 - (1 - r/h))
            if  y[j] - y[i] > 1e-6:
                theta = -np.arccos((x[j]-x[i])/h)
            else:
                theta = np.arccos((x[j]-x[i])/h)
            xp = r * np.cos(theta) + fr * np.sin(theta) + x[i]
            yp = -r * np.sin(theta) + fr * np.cos(theta) + y[i]
            w_name = f"w_{i}{j}_{state1}{state2}"
            if (optimization and w_name in self.weights.keys()) or (
                    state1 < dims[i] and state2 < dims[j] and not optimization):
                if optimization:
                    norm = np.sqrt(np.sum(np.array(list(self.weights.values()))**2))
                    args = {"linewidth":3*abs(self.weights[w_name])/norm}
                else:
                    args = {}
                if isinstance(color, list):
                    ax.plot(xp[:len(xp)//2+1], yp[:len(xp)//2+1], color=color[1], **args)
                    ax.plot(xp[len(xp)//2:], yp[len(xp)//2:], color=color[0], **args)
                else:
                    ax.plot(xp, yp, color=color, **args)
            


