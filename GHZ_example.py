#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 12:39:33 2021

@author: alejomonbar
"""
import Theseus

Dimensions = 4 * [2]  # Dimensions of all states
State = [[0, 0, 0, 0], [1, 1, 1, 1]]
Graph = Theseus.Graph(Dimensions)
Fidelity = Graph.fidelity(State)
sol, vars_ = Graph.topological_optimization(Fidelity)
print("w_ab_00:", Graph.TargetEquation)
print("Weights of the solution: ", vars_)
Graph.plot(filename="GHZ_prior_optimization.png")
Graph.plot(optimization=True, filename="GHZ_after_optimization.png")
