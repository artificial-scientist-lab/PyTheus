import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as collections
import pytheus.theseus as th
# import pytheus.analyzer as anal
import matplotlib.patheffects as pe
# from pytheus.fancy_classes import Graph
import json
import os
import pytheus.leiwand

from matplotlib.patches import Rectangle, Wedge, Circle
import matplotlib
matplotlib.rcParams['figure.dpi']=300
import itertools
import random
from matplotlib.markers import MarkerStyle
from collections import Counter
import matplotlib as mpl
from collections.abc import Iterable
from collections import Counter
from ast import literal_eval


def drawEdge(edge, verts, ind, mult, ax, scale_max=None, max_thickness=10,
             show_val=False, fs=15, markersize=25):
    colors = ['dodgerblue', 'firebrick', 'limegreen', 'darkorange', 'purple', 'yellow', 'cyan']
    col1 = colors[int(edge[2])]
    col2 = colors[int(edge[3])]

    vert1 = np.array(verts[int(edge[0])])
    vert2 = np.array(verts[int(edge[1])])
    if not np.array_equal(vert1, vert2):
        diff = vert1 - vert2
        rect = [diff[1], -diff[0]]
        rect /= np.linalg.norm(rect)
        hp = (vert1 + vert2) / 2 + (2 * ind - mult + 1) * 0.05 * rect
    else:
        hp = vert1 * 1.2

    if scale_max is None:
        lw = max_thickness

    else:
        lw = np.max([abs(max_thickness * edge[4]) / scale_max, 0.5])

    try:
        transparency = 0.2 + abs(edge[4]) * 0.8
        transparency = min(transparency, 1)
    except IndexError:
        transparency = 1
    except TypeError:
        transparency = 1

    ax.plot([vert1[0], hp[0]], [vert1[1], hp[1]], color=col1, linewidth=lw, alpha=transparency)
    ax.plot([hp[0], vert2[0]], [hp[1], vert2[1]], col2, linewidth=lw, alpha=transparency)

#     if show_val:

#         if transparency > 0.5 and col1 == "blue":
#             font_col = 'white'
#         else:
#             font_col = 'black'
#         latex_weight = '${}$'.format(anal.num_in_str(edge[4]))
#         if latex_weight == '$$':
#             latex_weight = str(edge[4])
#         ax.text(np.mean([0.9 * vert1[0], hp[0]]), np.mean([0.9 * vert1[1], hp[1]]),
#                 latex_weight,
#                 bbox={'facecolor': col1, 'alpha': transparency, 'edgecolor': col2, 'pad': 1}, c=font_col,
#                 ha='center', va='center', rotation=0, fontweight='heavy', fontsize=fs)
    try:
        if edge[4] < 0:
            ax.plot(hp[0], hp[1], marker="d", markersize=markersize, markeredgewidth="3", markeredgecolor="black",
                    color="white")
    except:
        pass


def graphPlot(graph, scaled_weights=False, show=True, max_thickness=10,
              weight_product=False, ax_fig=(), add_title='',
              show_value_for_each_edge=False, fontsize=30, zorder=11,
              markersize=25, number_nodes=True, filename='',figsize=10):
    '''
    Introducing a list/tuple of edges or a dictionary {edge:weight}, 
    this function plots the corresponding graph.
    
    Parameters
    ----------
    graph : list, tuple or dictionary
        List/tuple of all colored edges: [(node1, node2, color1, color2), ...]
        or dictionary with weights: {(node1, node2, color1, color2):weight1, ...}
    
    TODO 
    '''
    if type(graph) != dict:
        graph = {edge:1 for edge in graph}
        edge_list = list(graph.keys())
        weight_list = list(graph.values())
    edge_list = list(graph.keys())
    weight_list = list(graph.values())
        
    edge_dict = th.edgeBleach(edge_list)

    num_vertices = len(np.unique(np.array(edge_list)[:, :2]))

    angles = np.linspace(0, 2 * np.pi * (num_vertices - 1) / num_vertices, num_vertices)

    rad = 0.9
    vertcoords = []
    for angle in angles:
        x = rad * np.cos(angle)
        y = rad * np.sin(angle)
        vertcoords.append(tuple([x, y]))

    vertnums = list(range(num_vertices))
    verts = dict(zip(vertnums, vertcoords))

    if scaled_weights:
        try: # I think this doesn't work anymore
            scale_max = np.max(np.abs(np.array(edge_list)[:, 4]))
        except:
            scale_max = None
    else:
        scale_max = None

    if len(ax_fig) == 0:
        fig, ax = plt.subplots(figsize=(figsize,)*2)
    else:
        fig, ax = ax_fig

    for uc_edge in edge_dict.keys():
        mult = len(edge_dict[uc_edge])
        for ii, coloring in enumerate(edge_dict[uc_edge]):
            drawEdge(uc_edge + coloring + tuple([graph[tuple(uc_edge + coloring)]]), verts, ii, mult, ax,
                     scale_max=scale_max, max_thickness=max_thickness,
                     show_val=show_value_for_each_edge, fs=0.8 * fontsize, markersize=markersize)

    circ = []
    if number_nodes:
        node_labels = verts.keys()
    else:
        node_labels = list(map(chr, range(97, 123)))
    for vert, coords in zip(node_labels, verts.values()):
        circ.append(plt.Circle(coords, 0.1, alpha=0.5))
        ax.text(coords[0], coords[1], str(vert), zorder=zorder,
                ha='center', va='center', size=fontsize)

    circ = collections.PatchCollection(circ, zorder=zorder - 1)
    circ.set(facecolor='lightgrey', edgecolor='dimgray', linewidth=3)
    ax.add_collection(circ)

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.axis('off')

#     if weight_product:
#         total_weight = np.product(weight_list)

#         wp = '${}$'.format(anal.num_in_str(total_weight))
#         if wp == '$$':
#             wp = str(total_weight)
#         ax.set_title(wp + str(add_title), fontsize=fontsize)

    if add_title != '' and weight_product is False:
        ax.set_title(str(add_title), fontsize=fontsize)

    if show:
        plt.show()
        plt.pause(0.01)
    else:
        pass
    if filename:
        fig.savefig(filename + ".pdf")

    return fig


def leiwandPlot(graph, name='graph'):
    data = []
    edge_dict = th.edgeBleach(graph.edges)
    for uc_edge in edge_dict.keys():
        mult = len(edge_dict[uc_edge])
        loop = (uc_edge[0] == uc_edge[1])
        for ii, coloring in enumerate(edge_dict[uc_edge]):
            edge = tuple(uc_edge + coloring)
            weight = graph[edge]
            if loop:
                loose = 10 + 5 * ii
                data.append([weight, str(edge[0]), edge[2], str(edge[1]), edge[3], loose])
            else:
                bend = -22.5 + (ii + 0.5) * 45 / mult
                data.append([weight, str(edge[0]), edge[2], str(edge[1]), edge[3], bend])
    pytheus.leiwand.leiwand(data, name)


def leiwandPlotBulk(graph, cnfg, root, name = 'graph'):
    # if graph is imaginary, just take absolute value as weight for now
    if graph.imaginary:
        graph.absolute()
    data = []
    edge_dict = th.edgeBleach(graph.edges)
    for uc_edge in edge_dict.keys():
        mult = len(edge_dict[uc_edge])
        loop = (uc_edge[0] == uc_edge[1])
        for ii, coloring in enumerate(edge_dict[uc_edge]):
            edge = tuple(uc_edge + coloring)
            weight = graph[edge]
            if loop:
                loose = 10 + 5 * ii
                data.append([weight, str(edge[0]), edge[2], str(edge[1]), edge[3], loose])
            else:
                bend = -22.5 + (ii + 0.5) * 45 / mult
                data.append([weight, str(edge[0]), edge[2], str(edge[1]), edge[3], bend])
    pytheus.leiwand.leiwandBulk(data, cnfg, root=root, name=name)


def plotFromFile(filename, number_nodes=True, outfile=""):
    if not os.path.exists(filename) or os.path.isdir(filename):
        raise IOError(f'File does not exist: {filename}')
    with open(filename) as input_file:
        sol_dict = json.load(input_file)
    # graph = Graph(sol_dict['graph'])
    # graphPlot(graph.graph, scaled_weights=True, number_nodes=number_nodes, filename=outfile)
    graphPlot(sol_dict['graph'], scaled_weights=True, number_nodes=number_nodes, filename=outfile)
    
##############################################################################################################################

colors = ['dodgerblue', 'firebrick', 'limegreen', 'darkorange', 'purple', 'yellow', 'cyan']
Paths = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

#plot path and optical elements
def Plot_BS(ax, X, Y, width, height, color):
    ax.add_patch(Rectangle((X, Y), width, height,fc = 'lavender', \
                           ec = 'navy', angle = 45, zorder =18))
    d0 = np.sqrt(width**2+height**2)/4
    ax.plot([X+d0, X-d0],[Y+d0, Y+3*d0 ],zorder = 20 , color = color)
    ax.plot([X-d0, X+d0],[Y+d0, Y+3*d0 ],zorder = 20 , color = color)
    ax.vlines(X, ymin = Y, ymax =Y+4*d0, colors ='navy',zorder = 19  )
     
def Plot_PBS(ax, X, Y, width,height, color1, color2 ):
    ax.add_patch(Rectangle((X, Y), width, height,fc = 'thistle', \
                           ec = 'indigo', angle = 45, zorder =18) )
    d0 = np.sqrt(width**2+ height**2)/4
    ax.plot([X+d0, X-d0],[Y+d0, Y+3*d0 ],zorder = 20 , color = color1)
    ax.plot([X-d0, X],[Y+d0, Y+2*d0 ],zorder = 20 , color = color2)
    ax.plot([X, X-d0],[Y+2*d0, Y+3*d0 ],zorder = 20 , color = color2,linestyle =':')
    ax.vlines(X, ymin = Y, ymax =Y+4*d0, colors ='indigo',zorder = 19  )
    
def Plot_SPDC(ax, X, Y, width, height, color1, color2, W, wmax = 1 ):
    ax.add_patch(Rectangle((X, Y), width/2, height, fc = color1, ec = 'none',alpha= abs(W)/wmax))
    ax.add_patch(Rectangle((X+width/2,Y), width/2, height, fc = color2, ec ='none' , alpha= abs(W)/wmax))
    ax.add_patch(Rectangle((X, Y),width, height, fc = \
                           'none', ec ='black',zorder = 10 ))
    d0 = width/10
    d1 = Y+height
    ax.vlines(X+d0, ymin = d1, ymax = d1+height, colors = color1)
    ax.vlines(X+width-d0, ymin = d1, ymax = d1+height, colors = color2)  
    
def Plot_Absorber(ax , X , Y,  width, height) :  
    ax.add_patch(Rectangle((X, Y), width, height,fc = 'k', ec = 'r',zorder=10, joinstyle= 'bevel', lw =2))
    
def Plot_Hline(ax , XMIN, XMAX, Y , color): 
    ax.hlines(Y, xmin=XMIN, xmax=XMAX, colors=color, zorder = 9)
    
def Plot_Vline(ax , YMIN, YMAX, X , color ): 
    ax.vlines(X, ymin=YMIN, ymax=YMAX, colors=color\
              ,zorder = 8 )   

def Plot_Connection_Line(ax,X,Y):
    t=np.linspace(0,1,1000)
    ax.plot(X[0]+(3*t**2-2*t**3)*(X[1]-X[0]),Y[0]+t*(Y[1]-Y[0]),
           color='k')    

def Plot_Detector(ax , X, Y, leng, step, radius ):
    pos = Pos_Element(X,step,leng)
    for ii in range(len(pos)):
        ax.add_patch(Wedge((pos[ii], Y), radius,0, 180,fc = 'k', ec = 'k', zorder = 10))
        ax.add_patch(Rectangle((pos[ii]-1.2*radius, Y-radius/2), 2.4*radius, radius/2, fc = 'k', ec = 'k', zorder = 12))
        Plot_Connection_Line(ax, [pos[ii], pos[ii]-radius], [Y+radius,Y+2.5*radius] )

def Plot_Crystal (X, Y, color, width, height, W, wmax =1): #for path identity
    ax.add_patch(Rectangle((X, Y), width, height,fc = 'none', ec ='black' ,zorder=6))
    row = len(color)
    column = 2
    y_crystal = Pos_Element(Y,height/row,row)
    x_crystal = Pos_Element(X,width/column, column)
    if len (y_crystal) == 1:
        height1 = height
    else:    
        height1 = y_crystal[1]-y_crystal[0]
    width1 = x_crystal[1]-x_crystal[0]
    for y in range(len(y_crystal)):
        posy = y_crystal[y]
        ax.hlines(posy, xmin=X ,xmax=X+width, colors='k', zorder = 6)
        for x in range(len(x_crystal)):
            posx= x_crystal[x]
            colors = color[y][x]
            ax.add_patch(Rectangle((posx, posy), width1, height1,\
                                   fc = colors, ec ='none' ,zorder=5, alpha =abs(W[y])/wmax ))
            
def Plot_Sorter(ax , X, Y, leng, step, width, height, color):
    pos = Pos_Element(X,step,leng)
    xmin = min(pos)
    xmax = max(pos)
    Plot_Hline(ax , xmin, xmax+width, Y+ height/10 , 'k')
    Plot_Hline(ax , xmin, xmax+width, Y+9*height/10 , 'k')
    Plot_Hline(ax , xmin, xmax+width, Y+height/2, 'k')
    for p in range(len(pos)):
        ax.add_patch(Circle((pos[p]+ width/2, Y+height/2), width/2,fc = color[p], ec = 'k', zorder = 15))
        ax.add_patch(Rectangle((pos[p], Y), width, height,fc = 'lightgray', ec = 'k', zorder = 12))

def Plot_Multi_Color_Line(X, Y, height, color, leng, radius):
    step = height/float(leng)
    y = Pos_Element(Y,step,leng)
    Plot_Detector(ax , X, y[-1]+radius/2, 1, 1, radius )
    loc = generate_N_grams (y, ngram = 2)
    for pp in range(len(loc)):
        Plot_Vline(ax , loc[pp][0],loc[pp][1], X, color[pp] ) 
