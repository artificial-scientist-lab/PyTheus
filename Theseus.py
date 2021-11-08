#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 07:02:11 2021

@author: alejomonbar
"""
import numpy as np
from sympy import symbols
import itertools
import scipy.optimize as optimize
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, Dimensions):
        self.num_vertices = len(Dimensions)
        self.max_modes = np.array(Dimensions) 
        self.vertices = self.node_paths()
        self.combinations = self.AllCombinations()
        self.TriggerableState = self.paths()
        
    def AllCombinations(self):
        """
        Creating all possible paths
        Parameters
        ----------
        abcd : List of sympy symbols
    
        Returns
        -------
        prod : np.array of sympy symbols
            
    
        """
        prod = 1
        for i in self.vertices:
            prod = np.kron(prod, i)
        return prod
    
    def node_paths(self):
        nodes = []
        for n in range(self.num_vertices):
            nodes.append(symbols(f"n{n}_:{self.max_modes[n]}"))
        return nodes
    
    def state_symbol(self, state):
        symb = 1
        for n, i in enumerate(state):
            symb *= self.vertices[n][i]
        return symb
    
    def paths(self):
        """
        Create the second order state, i.e. two excited edges
    
        Parameters
        ----------
        vertices : List of sympy symbol
        localDim : list of integers
    
        Returns
        -------
        FullState : List of sympy equations
            Create a full state based on the edges combinations
    
        """
        n = len(self.vertices)
        localDim = self.max_modes
        FullState = 0
        for state in self.iterables():
            whole_list = 1
            for i in range(0,n,2):
                ij = [state[i], state[i+1]]
                lis = []
                for l in itertools.product(range(localDim[ij[0]]), range(localDim[ij[1]])):
                    lis.append(symbols(f"w_{ij[0]}{ij[1]}_{l[0]}{l[1]}")*self.vertices[ij[0]][l[0]]*self.vertices[ij[1]][l[1]])
                whole_list = np.kron(whole_list, np.array(lis))
            FullState += whole_list.sum()
        return FullState


    def fidelity(self, desired_state):
        AllEquations = []
        TargetEquations = []
        for comb in self.combinations:
            newEq = np.sum([i for i in self.TriggerableState.args if str(comb) in str(i)]).subs([(comb,1)])
            for state in desired_state:
                if comb == self.state_symbol(state):
                    # This term is part of the quantum state we want
                    TargetEquations.append(newEq)
            AllEquations.append(newEq)
        # Run the Optimization 
        self.TargetEquation = TargetEquations
        NormalisationConstant2 = np.sum(np.array(AllEquations)**2)
        self.NormalisationConstant = NormalisationConstant2
        Fidelity = np.sum(np.array(TargetEquations))**2/(len(TargetEquations)*NormalisationConstant2)
        return Fidelity

    def LossFun(self, variables, Loss2fun):
        return Loss2fun(*variables)
    
    def minimize(self, Fidelity, initial_weights=[], alpha=0):
        """
        

        Parameters
        ----------
        alpha : float
            Regularization constant.

        Returns
        -------
        None.

        """
        variables = list(Fidelity.free_symbols)
        if len(initial_weights) == 0:
            initial_weights = 2 * np.random.rand(len(variables)) - 1
        loss = (1 - Fidelity) + alpha * np.sum(np.abs(variables))
        loss2fun = lambdify(variables, loss, modules="numpy")
        sol = optimize.minimize(self.LossFun, initial_weights, args=(loss2fun))
        vars_ = [(key,value) for key, value in zip(variables,sol.x)]
        return sol, vars_
        
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
        
    def topological_optimization(self, Fidelity, initial_weights=[], alpha=0.5, loss_min=5e-2,
                                 max_counts=100, w_limit=1):
        sol, vars_ = self.minimize(Fidelity, initial_weights, alpha=0)
        weight_last = sol.x
        count = 0
        while count < max_counts:
            self.Fidelity_last = Fidelity
            variables = list(Fidelity.free_symbols)
            ith_rid = np.random.choice(np.arange(len(variables)))
            Fidelity = Fidelity.subs([(variables[ith_rid], 0)])
            new_vars = list(Fidelity.free_symbols)
            initial_weights = [i for n, i in enumerate(weight_last) if variables[n] in new_vars]
            if len(initial_weights) == 0:
                Fidelity = self.Fidelity_last
                count += 1
            else:
                sol, vars_ = self.minimize(Fidelity, initial_weights, alpha=alpha)
                w_sum = np.abs(sol.x).sum()
                if (sol.fun < loss_min) and (w_sum < w_limit):
                    count = 0
                    weight_last = sol.x
                else:
                    Fidelity = self.Fidelity_last
                    count += 1
            print(f"The solution:{sol.fun} - count:{count} - w_sum:{w_sum}")
        self.weights = {str(i[0]):i[1] for i in vars_}
        return sol, vars_
        
    # def iterables(self):
    #     n = self.num_vertices
    #     combinations = list(itertools.combinations(range(n),2))
    #     Comb = []
    #     for num, c0 in enumerate(combinations):
    #         if c0[0] == 0:
    #             cT = list(c0)
    #             for c1 in combinations:
    #                 c1 = list(c1)
    #                 flag = any(item in c1 for item in cT)
    #                 if not flag:
    #                     cT += c1
    #             Comb.append(cT)
    #     return Comb
    
    def iterables(self):
        n = self.num_vertices
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
            


