import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as collections
import pytheus.theseus as th
import matplotlib.patheffects as pe
import json, os
import pytheus.leiwand

import pytheus
from pytheus.fancy_classes import Graph
from matplotlib.patches import Rectangle, Wedge, Circle
import matplotlib
matplotlib.rcParams['figure.dpi']=300
from matplotlib.markers import MarkerStyle
from collections import Counter
from collections.abc import Iterable
from ast import literal_eval
from collections import OrderedDict
import itertools, random, string

colors = ['dodgerblue', 'firebrick', 'limegreen', 'darkorange', 'purple', 'yellow', 'cyan']
Paths = list(string.ascii_lowercase)

# # Graph plotting tools

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
    # if weight_product:
    #     total_weight = np.product(weight_list)
    #     wp = '${}$'.format(anal.num_in_str(total_weight))
    #     if wp == '$$':
    #         wp = str(total_weight)
    #     ax.set_title(wp + str(add_title), fontsize=fontsize)
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
    graphPlotOld(sol_dict['graph'], scaled_weights=True, number_nodes=number_nodes, filename=outfile)

# # Experiment plotting tools

def transparency(W):
    try:
        transparency = 0.2 + abs(W) * 0.8
        transparency = min(transparency, 1)
    except IndexError:
        transparency = 1
    except TypeError:
        transparency = 1  
    return transparency

def PosOfVertices (num_nodes, side_length):
    vertices = []
    for nv in range(num_nodes):
        angle = 2 * nv * np.pi / num_nodes
        x = side_length * np.cos(angle)
        y = side_length * np.sin(angle)
        vertices.append((x, y))
    return(vertices)

def correct(tpl):
    corrected_item = []
    for i, num in enumerate(tpl):
        if  9 < num < 100:
            num = num % 10 + 10     
        corrected_item.append(num)
    return tuple(corrected_item)

def convert_bools_to_ints(input_list):
    converted_list = [int(item) if isinstance(item, bool) else item for item in input_list]
    return converted_list

def convert_to_fancy_graph(graph):
    if isinstance(graph, list) or isinstance(graph, dict):
        fancy_graph = Graph(graph)
        return fancy_graph
    elif isinstance(graph, pytheus.fancy_classes.Graph):
        return graph
    else:
        raise ValueError("Input should be a list or dictionary or fancy graph")

def calculate_b(nb, x):
    b = (np.array(range(nb))-0.5*(nb-1)) * x / (np.max(np.arange(1, nb+1))-0.5*(nb-1))
    return b

def openfile(filename):
    if not os.path.exists(filename) or os.path.isdir(filename):
        raise IOError(f'File does not exist: {filename}')
    with open(filename) as input_file:
        file_dict = json.load(input_file)    
    return file_dict 

def plot_diamond(ax ,center_x, center_y, diamond_width, diamond_height, zorder  ): 
    
    diamond_x = [center_x, center_x + diamond_width / 2, center_x, center_x - diamond_width / 2, center_x]
    diamond_y = [center_y + diamond_height / 2, center_y, center_y - diamond_height / 2,\
                 center_y, center_y + diamond_height / 2]
    ax.plot(diamond_x, diamond_y, color='black', zorder = zorder+1)
    ax.fill(diamond_x, diamond_y, color='w', zorder = zorder)

def plot_triangle(ax ,center, side_length, zorder, linewidth):
    center_x = center[0]
    center_y = center[1]
    x = [center_x - side_length/2, center_x + side_length/2, center_x]
    y = [center_y - (3**0.5)*side_length/6, center_y - (3**0.5)*side_length/6, center_y + (2*(3**0.5)*side_length)/6]
    ax.plot(x + [x[0]], y + [y[0]], zorder = zorder+1, lw =linewidth, color ='k' )
    plt.fill(x, y, color='w', zorder = zorder)

def Plot_Vertices(ax, num_nodes, side_length, type_photons = None, font_size =12, linewidth=2):
    r = side_length/10
    vertices = PosOfVertices(num_nodes, side_length)
    if type_photons is None:
        for i, vertex in enumerate(vertices):
            ax.add_patch(Circle(vertex, radius= r, facecolor='w',\
                            edgecolor='black', zorder = 10, linewidth=linewidth))
            ax.text(vertex[0], vertex[1], str(i), ha='center', \
                va='center', fontsize=font_size,zorder = 12 )
        
    elif isinstance(type_photons, tuple) or isinstance(type_photons, list):
        if len(type_photons)==3:
            ancilla, single_emitters, in_nodes = type_photons
            for i, vertex in enumerate(vertices):
                if i in ancilla:
                    if i in single_emitters:
                        plot_triangle(ax, vertex ,2.5*r, 10, linewidth=linewidth  )
                    else:
                        x =vertex[0]-r
                        y = vertex[1]-r
                        ax.add_patch(Rectangle((x,y),2*r, 2*r,fc = 'w', \
                                       ec = 'k',zorder = 10, linewidth=linewidth))
                elif i in single_emitters or i in in_nodes:
                    plot_triangle(ax, vertex ,2.5*r, 10, linewidth=linewidth  )
                else:
                    ax.add_patch(Circle(vertex, radius= r, facecolor='w',\
                                    edgecolor='black', zorder = 10, linewidth=linewidth))
                ax.text(vertex[0], vertex[1], str(i), ha='center', \
                        va='center', fontsize=font_size,zorder = 12 )
        else:
               raise ValueError("The type_photons is not valid.")

def Plot_Edges(ax, V1, V2, colors, side_length, max_len, max_thickness = 5,\
               min_thickness = 2, thickness=10, de = 5 ):
    line_width = max(min(thickness/ max_len, max_thickness), min_thickness)
    r =side_length/6#12
    if V1 == V2:
        w =colors[0][1]
        x = V1[0] + side_length / 10 if V1[0] > 0 else V1[0] - side_length / 10
        y = V1[1] + side_length / 10 if V1[1] > 0 else V1[1] - side_length / 10
        x1 = x + r if V1[0] > 0 else x-r
        y1 = y+r if V1[1] > 0 else y-r
        ax.add_patch(Circle((x,y), radius=r, facecolor='w', edgecolor=colors[0][0][0],\
                        zorder = 7,linewidth=line_width,alpha = transparency(w) ))
        if w< 0:
             plot_diamond(ax, x1, y1, r/2, r, zorder = 10 )            
    else:
        h = (V1[0] + V2[0]) / 2
        k = (V1[1] + V2[1]) / 2
        theta = np.arctan2(V1[1] - V2[1], V1[0]-V2[0])
        a = np.sqrt((V1[0]-V2[0]) ** 2 + (V1[1] - V2[1]) ** 2) / 2
        nb = len(colors)
        b = calculate_b(nb, side_length/de )
        t1 = np.linspace(0, np.pi / 2, 1000)
        t2 = np.linspace(np.pi, np.pi / 2, 1000)
        for ed in range(len(b)):
            xi = h + a * np.cos(theta) * np.cos(t1) - b[ed] * np.sin(theta) * np.sin(t1)
            yi = k + a * np.sin(theta) * np.cos(t1) + b[ed] * np.cos(theta) * np.sin(t1)
            xj = h + a * np.cos(theta) * np.cos(t2) - b[ed] * np.sin(theta) * np.sin(t2)
            yj = k + a * np.sin(theta) * np.cos(t2) + b[ed] * np.cos(theta) * np.sin(t2)
            w = colors[ed][1]
            ax.plot(xi, yi, color=colors[ed][0][0], linewidth=line_width, alpha = transparency(w), zorder = 9 )
            ax.plot(xj, yj, color= colors[ed][0][1], linewidth=line_width, alpha = transparency(w), zorder = 9 )
            if w< 0:
                 plot_diamond(ax, xi[-1],yi[-1], r/2, r, zorder = 9 )

def Type_Photons(config_file_name, sol_file_name):
    sol_file = openfile(sol_file_name)
    config_file = openfile(config_file_name)
    graph = sol_file['graph']
    edge_list = list(graph.keys())
    for ii in range (len(edge_list)):
        edge_list[ii] = literal_eval(edge_list[ii])
    num_vertices = len(np.unique(np.array(edge_list)[:, :2]))
    vertex_list = np.arange(0, num_vertices, 1).tolist()
    if 'num_anc' in config_file:
        num_anc = config_file['num_anc']
        if num_anc ==0:
            ancilla = []
        else:
            ancilla = vertex_list[-num_anc:]
    else :
        ancilla = []
   
    if 'single_emitters' in  config_file:
        single_emitters = config_file['single_emitters']
    else:
        single_emitters = []
        
    if 'in_nodes' in  config_file:
        in_nodes = config_file['in_nodes']
    else:
        in_nodes = []   
    return ancilla, single_emitters, in_nodes

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

def Plot_SPDC(ax, X, Y, width, height, color1, color2, W ):
    alpha = transparency(W)
    ax.add_patch(Rectangle((X, Y), width/2, height, fc = color1, ec = 'none',alpha= alpha))
    ax.add_patch(Rectangle((X+width/2,Y), width/2, height, fc = color2, ec ='none' , alpha=alpha))
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
    t=np.linspace(0,1,20)
    ax.plot(X[0]+(3*t**2-2*t**3)*(X[1]-X[0]),Y[0]+t*(Y[1]-Y[0]),
           color='k',zorder = 8)  

def Plot_Detector(ax , X, Y, leng, step, radius ):
    pos = Pos_Element(X,step,leng)
    for ii in range(len(pos)):
        ax.add_patch(Wedge((pos[ii], Y), radius,0, 180,fc = 'k', ec = 'k', zorder = 10))
        ax.add_patch(Rectangle((pos[ii]-1.2*radius, Y-radius/2), 2.4*radius, radius/2, fc = 'k', ec = 'k', zorder = 12))
        Plot_Connection_Line(ax, [pos[ii], pos[ii]-radius], [Y+radius,Y+2.5*radius] )

def Plot_Crystal (ax, X, Y, color, width, height, W): #for path identity
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
                                   fc = colors, ec ='none', zorder=5, alpha =transparency(W[y])))

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

def Plot_Multi_Color_Line(ax, X, Y, height, color, leng, radius):
    step = height/float(leng)
    y = Pos_Element(Y,step,leng)
    Plot_Detector(ax , X, y[-1]+radius/2, 1, 1, radius )
    loc = generate_N_grams (y, ngram = 2)
    for pp in range(len(loc)):
        Plot_Vline(ax , loc[pp][0],loc[pp][1], X, color[pp] ) 

def Write_Label(ax, X, Y, text, fontsize ):
   ax.text(X,Y, s= text, fontsize = fontsize)

def get_num_label(labels):
    num_to_label = dict((num, label) for num, label in enumerate(labels))
    return num_to_label

def encoded_label(nums,labels ):# for transform num to alphabet
    encoded_labels =[labels[num] for num in nums]
    return encoded_labels

def grouper(n, iterable):
    args = [iter(iterable)] * n
    return list(zip(*args))

def Combine(x):
    y = (list(itertools.combinations(x,2)))
    return y

def Pos_Element(low,step,leng):
    Pos = []
    if leng == 0:
        Pos = Pos
    elif leng>0:
        up = step*float(leng)+low
        for i in range(leng):
            Pos.append(low)
            low = low + step
    return Pos

def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

def DuplicateList(lst):
    uniqueList = []
    duplicateList = []
    for i in lst:
        if i not in uniqueList:
            uniqueList.append(i)
        elif i not in duplicateList:
            duplicateList.append(i)
    pos_list = []       
    for jj in duplicateList:
        x = list_duplicates_of(lst,jj)
        pos_list.append([jj, x])    
    return( pos_list)

def REMOVE_BS(lst1 , lst2 ):
    lst3 = []
    lst1 = list(itertools.permutations(lst1, 2))
    for ii in lst2:
        for jj in lst1:
            if ii == jj:
                lst3.append(ii)
    lst2 = [x for x in lst2 if  x not in lst3]
    return lst2

def flatten(X):
    if isinstance(X, Iterable):
        return [A for I in X for A in flatten(I)]
    else:
        return [X]

def LengDuplicate(lst):
    result = dict((i, lst.count(i)) for i in lst)
    x = result.values()
    count = 0
    for item in x:
        if item>1:
            count += item
    return(count)

def gen_list_of_lists(original_list, new_structure):
    assert len(original_list) == sum(new_structure)  
    list_of_lists = [[original_list[i + sum(new_structure[:j])] for i in range(new_structure[j])] \
                     for j in range(len(new_structure))]  
    return list_of_lists

def uniqueList(lst):
    uniqueList = []
    for i in lst:
        if i not in uniqueList:
            uniqueList.append(i)
    pos_list = []       
    for jj in uniqueList:
        x = list_duplicates_of(lst,jj)
        pos_list.append([jj, x])    
    return( pos_list)

def generate_N_grams (position,ngram = 1):
    positions=[pos for pos in position]
    grams=zip(*[positions[i:] for i in range(0,ngram)])
    return list(grams)

#index 0 :path, 1: color
def GetGraphColorEdge(graph, index, PC):
    GraphED = [grouper(2,i)[index] for i in graph.edges]
    GraphEC = [encoded_label(ED,get_num_label(PC))for ED in GraphED ]
    return GraphEC

def Pos0fpath(lst, x):
    Pospath = []
    d0 = x/10
    for pos in lst:
        x1 = pos+d0
        x2 = pos+x-d0
        Pospath.extend([x1, x2]) 
    return(Pospath)

def get_index_color(colors,lst_col):
    num_to_color = dict((num, color) for num, color in enumerate(colors))
    color_to_num = {color: num for num, color in num_to_color.items()}
    index_col = encoded_label(lst_col,color_to_num )
    return  index_col

def find_index_duplicate(lists, item):
    index = []
    for idx in range(len(lists)):
        for ele in lists[idx]:
            if ele == item :
                index.append(idx)
    return index

def union(lst):
    for i in range(len(lst)):
            for j in range(i+1,len(lst)):
                for k in lst[j]:
                    if k[0] not in list(itertools.chain(*lst[i]))\
                    and  k[1] not in list(itertools.chain(*lst[i])):
                        lst[i].append(k)
                        lst[j].remove(k)
    lst =  list(filter(None, lst))
    return (lst)

def layer0fcrystal (crystal_lst, Numphoton):
    res = th.findPerfectMatchings(crystal_lst)
    ll = int(Numphoton/2)
    layer0= []
    other_crystal = []
    while len(res)>0:
        r = res[0]
        layer0.append(r)
        res = [[ele for j,ele in enumerate(sub) if ele not in r] for i,sub in enumerate(res)]
        for item  in res:
            if 0<len(item)<ll:
                other_crystal.append(item)
        res = [item for item in res if len(item)>=ll]  
    layer1 = [[ele for j,ele in enumerate(sub) if ele not in  list(itertools.chain(*layer0))]
                      for i,sub in enumerate(other_crystal)]
    flatten = []
    for nl in  layer1:
        for i in range(len(nl)-1, -1, -1):
            if nl[i] not in flatten:
                flatten.append(nl[i])
            else:
                nl.pop(i)
    layer1= sorted(union(sorted(list(filter(None, layer1)))),\
                   key=lambda l: (len(l), l),reverse = True)
    layer = layer0+layer1
    return( layer)

def Get_Color_Weight_Crystals(gea, Numphoton, gcw, Layers):
    colwei =  uniqueList(gea)
    for ii in range(len(colwei)):
        x =colwei[ii][1]
        for jj in range(len(x)):
            x[jj]=gcw[x[jj]]     
    Remove_Duplicate = list(gea for gea,_ in itertools.groupby(gea))
    cw_spdc = []
    for ii in Layers :
        cw = [Remove_Duplicate.index(list(jj)) for jj in ii]
        cw_spdc.append(cw)
    wc =[]
    for ii in range(len(colwei)):
        wc.append(colwei[ii][1])

    for ii in range(len(cw_spdc)):
        wcspdc = cw_spdc[ii]
        for jj in range(len(wcspdc)):
            wcspdc[jj] =wc[wcspdc[jj]]
    return(cw_spdc)

def PlotPathIdentity(graph,  filename= "", width=0.1, figsize= (8, 8) , 
                       fontsize = 16 , colors = colors , Paths= Paths):
    
    fig, ax = plt.subplots(ncols=1, nrows=1,figsize= figsize, facecolor='w')
    
    graph = convert_to_fancy_graph(graph)
    GraphEdgesAlphabet = GetGraphColorEdge(graph, 0, Paths )
    count = 0
    for v in GraphEdgesAlphabet:
        if v[0]==v[1]:
            count +=1
    if count >0:
        raise ValueError("The graph has self-loops")  
        
    else:
        
        GraphEdgesColor  =  GetGraphColorEdge(graph, 1, colors )
        Graphweight = convert_bools_to_ints(graph.weights)
        Dimension = len(np.unique(list(itertools.chain(*GraphEdgesColor))))
        Numphoton =  len(np.unique(list(itertools.chain(*GraphEdgesAlphabet))))

        Remove_Multiedge = list(tuple(GraphEdgesAlphabet) for GraphEdgesAlphabet
                                ,_ in itertools.groupby(GraphEdgesAlphabet))
      
        height = width/2
       

        if  len(th.findPerfectMatchings(Remove_Multiedge))==0:

            raise ValueError("It appears that there is no perfect matching in the graph.")  

        else:
            Layers1 = layer0fcrystal(Remove_Multiedge, Numphoton)

        NotInPM =[edges for edges in Remove_Multiedge if edges
                  not in list(itertools.chain(*Layers1))]
    
        if len( NotInPM) > 0:
            Layers2 = union([NotInPM [i:i+1] for i in range(0, len(NotInPM ), 1)])
            Layers =  Layers1 + Layers2
        else:
            Layers = Layers1
        Layers = union( Layers)
        
        color_spdc = Get_Color_Weight_Crystals(GraphEdgesAlphabet, Numphoton,
                                                GraphEdgesColor, Layers)

        w_spdc = Get_Color_Weight_Crystals(GraphEdgesAlphabet, Numphoton, Graphweight, Layers)

        Detector =list(itertools.chain(*Layers[0]))
        PosxSpdc = []
        ys = []
        width = 0.1
        height = width/2
        for ii in range(len(Layers)):
            numx = len(Layers[ii])
            ys.append(numx)
            px = Pos_Element(0, 3/2*width , numx)
            PosxSpdc.append(px)

        numy = len(PosxSpdc)
        PY = Pos_Element(0, 2*height , numy)
        PY.sort(reverse=True)
        P0 = PosxSpdc[0]
        PosySpdc=  gen_list_of_lists(np.repeat(PY, ys).tolist(), ys)

        pospathx = []
        pospathy = []

        for ii in range(len(PosxSpdc)):
            x = Pos0fpath(PosxSpdc[ii], width)
            pospathx.append(x)
            for jj in range(len(PosxSpdc[ii])):
                Plot_Crystal (ax, PosxSpdc[ii][jj], PosySpdc[ii][jj], color_spdc[ii][jj], width\
                              , height, w_spdc[ii][jj]) 

                y1 = PosySpdc[ii][jj]+height-height/10
                y2 =PosySpdc[ii][jj]+height/10
                pospathy.extend([[y1, y2]])
                Plot_Vline(ax , y1, y2 , x , 'k')

        YDR = max(PY)+2*height 
        XDR = Pos0fpath(P0, width)
        for pos in range (len(XDR)):
            Plot_Detector(ax , XDR[pos], YDR, 1,1, height/4 )
            Plot_Vline(ax ,  pospathy[0][0], YDR, XDR[pos], 'k' )
            Write_Label(ax,  XDR[pos], YDR+width/4, Detector[pos] , fontsize )

        lrs = [list(itertools.chain(*pp)) for pp in Layers]
        ps = sorted(list(itertools.chain(*Layers[0])))
        duplicate0fps= [find_index_duplicate(lrs, pp) for pp in ps]

        virtual = []
        flr=  pospathx[0]
        for lst in range (len(duplicate0fps )):
            for idx in range(duplicate0fps [lst][0], duplicate0fps [lst][-1]):
                if idx not in duplicate0fps [lst]:
                    duplicate0fps[lst].append(idx)
                    Layers[idx].append(ps[lst])
                    virtual.append(idx)

        connecty =[[PY[idx] for idx in sorted(duplicate0fps[lst])]\
               for lst in range(len(duplicate0fps))]

        connecty= [ grouper(2, sorted(Pos0fpath(lst, height ),\
                       reverse = True)[1:-1]) for lst in connecty]
        connecty= [y for y in connecty if y != []]


        y = [PY[idx] for idx in sorted(virtual) ]
        y =  grouper(2, Pos0fpath(y, height ))


        count =  dict(Counter(sorted(virtual)))
        ele = list(count.keys())
        num = list(count.values())

        fl =  pospathx[0]
        x = []
        for ii in range(len(ele)):
            leng = len(pospathx[ele[ii]])
            for i in range(num[ii]):
                item = fl[i+leng]
                x.append(item)
                pospathx[ele[ii]].append(item)    
        Pathconnect = list(itertools.chain(*Layers))
        Pathconnect = list(itertools.chain(*Pathconnect))
        Connection_Line = DuplicateList(Pathconnect)
        connect = flatten( pospathx)


        for ii in range(len(Connection_Line)):
            cl = Connection_Line[ii][1]
            for jj in range(len(cl)):
                cl[jj] = connect[cl[jj]]

        Connection_Line = dict(Connection_Line)
        CL = dict(OrderedDict(sorted(Connection_Line.items())))

        connectx =[generate_N_grams(lst, 2) for lst in list(CL.values())]

        for ii in range(len(connectx)):
            for jj in range(len(connectx[ii])):
                Plot_Connection_Line(ax,connectx[ii][jj],connecty[ii][jj])

        for ii in range(len(x)):
            Plot_Vline(ax , y[ii][0], y[ii][1], x[ii] ,'k' )
        ax.set_aspect( 1 )        
        ax.axis('off') 
        experiment = fig.savefig(filename + ".pdf", bbox_inches='tight')   
    return experiment


def PlotBulkOpticsPathEncoding(graph, task = 'PathEncoding'  , filename='', width =0.1\
                               , figsize =(16, 16) , fontsize= 16 , colors= colors , Paths=Paths):
    graph = convert_to_fancy_graph(graph)
    GraphEdgesAlphabet = GetGraphColorEdge(graph, 0, Paths )
    GraphEdgesColor  =  GetGraphColorEdge(graph, 1, colors )
    Graphweight = convert_bools_to_ints(graph.weights)
    Num0fCrystal = len(graph)
    Dimension = len(np.unique(list(itertools.chain(*GraphEdgesColor))))
    Numphoton =  len(np.unique(list(itertools.chain(*GraphEdgesAlphabet))))
    height =width/2
    PosX0fcrystal = Pos_Element(0, 3*width/2 , Num0fCrystal)
    PosY0fcrystal = np.full(Num0fCrystal, 0) 
    
    fig,ax=plt.subplots(ncols=1,nrows=1,figsize= figsize, facecolor='w') 
  
    for num in range(len(graph)):
        Plot_SPDC(ax, PosX0fcrystal[num], PosY0fcrystal[num],\
                  width, height, GraphEdgesColor[num][0],\
                  GraphEdgesColor[num][1], Graphweight[num])
  
    PosX0fpath = Pos0fpath(PosX0fcrystal, width)
    Y = 2*height
    PosY0fpath = [ Y for pos in range(len(PosX0fpath))]

    AllPath= [] 
    for pp in range(len(graph)):
            AllPath.extend([str(GraphEdgesAlphabet[pp][0])+str(pp)\
                          ,str(GraphEdgesAlphabet[pp][1])+str(pp)])

    AllColor = list(itertools.chain(*GraphEdgesColor))
    PossiblePath = Combine(AllPath)
    PossibleColor= Combine(AllColor) 
    PossibleposX = Combine(PosX0fpath)
    

    PosX_L  = []
    PosX_IN_BS = []
    Path_L = []


    for ii in range(len(PossiblePath)):
        path1 = PossiblePath[ii][0]
        path2 = PossiblePath[ii][1]
        color1 = PossibleColor[ii][0]
        color2 = PossibleColor[ii][1]
        posx1 = PossibleposX[ii][0]
        posx2 = PossibleposX[ii][1]

        if path1[0]==path2[0] and color1== color2:
            PosX_IN_BS.append(posx2)
            PosX_L.extend([posx1,posx2] )
            Path_L.extend([path1, path2])

    PosX_IN_BS  = list(set(PosX_IN_BS ))
    PosX_L = grouper(2, PosX_L)  
    Pos =  REMOVE_BS(PosX_IN_BS, PosX_L )
    if len (Pos)>0:
        P1 , P2 =   list(zip(*Pos )) 
        GET_DUBL = DuplicateList(P1)
        GET_DUBL2 = DuplicateList(P1)
        PosX1 =[]
        
        for ii in range(len(GET_DUBL)):
            x = GET_DUBL[ii][1]
            for jj in x:
                PosX1.append(((P1[jj], P2[jj])))
     

        PosX2 = [pos for pos in Pos if pos not in PosX1] 
        PosY0 = Pos_Element(Y, height/3 ,len(GET_DUBL))
        if (len(PosY0))>0:
            PosY1 = Pos_Element(max(PosY0)+height/2,2.5*height , len(PosX1))
        if (len(PosX1))>0:
            PosY2 = Pos_Element(max(PosY1)+4*height,1.5*height,len(PosX2))
        elif (len(PosX1))==0:
            PosY2 = Pos_Element(Y+height, 2*height, len(PosX2))
        d0 = np.sqrt((width/2)**2+ height**2)/4
        for ii in range(len(PosX1)):
            x = PosX1[ii][1]
            y = PosY1[ii]
            c = AllColor[PosX0fpath.index(x)]
            Plot_BS(ax , x-d0 , y, width/2, height, c)
            Plot_Vline(ax , Y, y+d0 ,x , c,  )
            Plot_Vline(ax , y+3*d0, y+2.5*height, x-2*d0, c) 
            Plot_Absorber(ax ,  x-2*d0-height/4 , y+2.5*height, height/2, height/2) 

        ynab1 = [] 
        xnab1 = []
        colnab1 = []
        pathnab1 = []

        for ii in range(len(PosX2)):
            x= PosX2[ii][1]
            x1 =PosX2[ii][0]
            y = PosY2[ii]
            ynab1.append(y+3.5*height)
            xnab1.append(x)
            c = AllColor[PosX0fpath.index(x)]
            p = AllPath[PosX0fpath.index(x)]
            colnab1.append(c)
            pathnab1.append(p)
            Plot_BS(ax , x-d0 , y, width/2, height, c)
            Plot_Vline(ax , Y, y+d0, x , c )
            Plot_Vline(ax , Y, y+d0, x1 , c,) 
            Plot_Hline(ax , x1, x-2*d0, y+d0 , c)
            Plot_Vline(ax , y+3*d0, y+3.5*height, x, c,) 
            Plot_Vline(ax , y+3*d0, y+2.5*height, x-2*d0, c) 
            Plot_Absorber(ax , x-2*d0-height/4 ,y+2.5*height, height/2, height/2)
 
        PosX0 = []
        for ii in range(len(GET_DUBL)):
            x = GET_DUBL[ii][1]
            x1 = GET_DUBL[ii][0]
            PosX0.append(x1)
            for jj in range(len(x)):
                x[jj] = P2[x[jj]]
        ColX0 = [AllColor[PosX0fpath.index(pos)] for pos in PosX0fpath if pos in PosX0]
        turn_leng = [len(GET_DUBL2[ii][1]) for ii in range(len(GET_DUBL2))]
    
        if len(turn_leng)>0:
            PosY1 = gen_list_of_lists(PosY1, turn_leng)

            for ii in range(len(GET_DUBL)):
                GET_DUBL2[ii][1] = PosY1[ii]
                GET_DUBL2[ii][0]= PosY0[ii]
    
        for ii in range(len( PosX0)):
                Plot_Vline(ax ,Y, PosY0[ii]+2*d0, PosX0[ii],  ColX0[ii]) 
        xh1 = []
        yh1 = []
        xh2 = []
        yh2 = []
        col1 = []

        for ii in range(len(GET_DUBL)):
            x = GET_DUBL[ii]
            y = GET_DUBL2[ii]
            xh1.append(flatten(x))
            yh1.append(flatten(y))
            xh2.append(x[1])
            yh2.append(y[1])
            color = [AllColor[PosX0fpath.index(pos)] for pos in PosX0fpath if pos in x[1]] 
            col1.append(color) 

        for ii in range(len(yh1)):
            x = yh1[ii]
            for jj in range(len(x)):
                if jj ==0:
                     x[jj] = x[jj]
                elif jj >0:
                     x[jj]=x[jj]+d0

        for ii in range(len(xh1)):
            for jj in range(len(xh1[ii])-1):
                Plot_Hline(ax , xh1[ii][jj], xh2[ii][jj]-2*d0, yh1[ii][jj]+2*d0, col1[ii][jj])
                Plot_Vline(ax , yh1[ii][jj]+2*d0, yh2[ii][jj]+d0, xh2[ii][jj]-2*d0 ,col1[ii][jj] )

        ynab2 = []
        xnab2 = []  
        for ii in range(len(xh1)):
            x = xh1[ii]
            y = yh1[ii]
            xnab2.append(x[-1])
            ynab2.append(y[-1]+3*height)

        for pos in range(len(ynab2)):
            Plot_Vline(ax , ynab2[pos]-3*height+2*d0, ynab2[pos], xnab2[pos] ,ColX0[pos])
            
        PosX_NAB =  xnab1+ xnab2
        PosY_NAB  = ynab1+ ynab2
        Path1nab = pathnab1
        Path2nab = [AllPath[PosX0fpath.index(pos)] for pos in PosX0fpath if pos in PosX0]
        Path_NAB =  Path1nab + Path2nab
        colornab1 = colnab1
        colornab2 = ColX0
        Color_NAB = colornab1+colornab2
        PoSX_R = [pos for pos in PosX0fpath  if pos not in P2+P1]
        Color_R = [AllColor[PosX0fpath.index(pos)] for pos in PosX0fpath if pos in PoSX_R]
        Path_R = [AllPath[PosX0fpath.index(pos)] for pos in PosX0fpath if pos in PoSX_R]
        PosY_R = [max(PosY0fpath) for pos in range(len(PoSX_R))] 

        PosX_Concat =  PoSX_R + PosX_NAB
        PosY_Concat = PosY_R+ PosY_NAB   
        Path_Concat = Path_R + Path_NAB
        Color_Concat = Color_R + Color_NAB

    elif len(Pos) == 0:
        PosX_Concat = PosX0fpath 
        PosY_Concat = PosY0fpath   
        Path_Concat = AllPath
        Color_Concat = AllColor

    Path_alphabet = []   
    Path_Number =[]
    
    for path in Path_Concat:
        Path_alphabet.append(path[0])
        Path_Number.append(path[1])

    counts = Counter(Path_alphabet)    
    single_path = [[PosX_Concat[Path_alphabet.index(item)],\
                    PosY_Concat[Path_alphabet.index(item)],\
                    Color_Concat[Path_alphabet.index(item)],
                    Path_Concat[Path_alphabet.index(item)]]\
                    for item in Path_alphabet if counts[item] <= 1]         

    for sp in range(len(single_path)):
        Plot_Detector(ax , single_path[sp][0], \
                      single_path[sp][1], 1, height/4,height/3)
        Write_Label(ax, single_path[sp][0], single_path[sp][1]+height/2, single_path[sp][3][0], fontsize )

    get_to_posX = sorted(DuplicateList(Path_alphabet))
    get_to_posY = sorted(DuplicateList(Path_alphabet))
    get_to_Color = sorted(DuplicateList(Path_alphabet))
    XtoD = sorted(DuplicateList(Path_alphabet))
    yy = max(PosY_Concat)

    try:
        if len(get_to_posX ) > 0:
            XD = Pos_Element(min(PosX_Concat)+height/3,(max(PosX_Concat)\
                 -min(PosX_Concat)-1.75*height)/len(get_to_posX), len(get_to_posX))
            FF = []
            for ii in range(len(XtoD)):
                Y = len(XtoD[ii][1])
                FF.append(Y)
            YY = Pos_Element(yy,height/3,sum(FF))
            YtoD = gen_list_of_lists(YY, FF) 
            YD = Pos_Element(max(YY)+height,height,len(get_to_posX))

            col_index = []
            for index in range(len(get_to_posX)):
                X = get_to_posX[index][1]
                Y = get_to_posY[index][1]
                C = get_to_Color[index][1]
                col_index.append(C)
                x = XD[index]
                y = YD[index]
                XtoD[index][1] = Pos_Element(XD[index],height,len(XtoD[index][1]))
                xd =  XtoD[index][1]
                yd = YtoD[index]
                
                for jj in range(len(X)):
                    X[jj]= PosX_Concat[X[jj]]
                    Y[jj]= PosY_Concat[Y[jj]]
                    C[jj]= Color_Concat[C[jj]]
                    Plot_Vline(ax ,Y[jj], yy, X[jj], C[jj] )
                    Y[jj]= yy
                    Plot_Vline(ax , Y[jj], yd[jj], X[jj] , C[jj] )
                    Plot_Vline(ax , y, yd[jj], xd[jj], C[jj] )
                    Plot_Hline(ax , X[jj], xd[jj], yd[jj] , C[jj])

            pathx_sort = []
            for col in range(len(col_index)):
                x = XtoD[col][1]
                col_index[col] = get_index_color(colors,col_index[col])
                min_index  = col_index[col].index(min(col_index[col]))
                pathx_sort.append(x[min_index ] )
            for px in range(len(get_to_posX)):
                c = get_to_Color[px][1]
                tt = get_to_Color[px][0]
                if task == 'BulkOptics':
                    Plot_Sorter(ax , XD[px]-height/4 ,YD[px], len(get_to_posX[px][1]), height, height/2, height,c)
                    Plot_Multi_Color_Line(ax,pathx_sort[px], YD[px]+height, 2*height, c, len(c)+1, height/3)
                    Write_Label(ax, pathx_sort[px], YD[px]+3.5*height, tt,  fontsize )
                elif task == 'PathEncoding':
                    Plot_Detector(ax , XD[px] ,YD[px], len(get_to_posX[px][1]), height, height/3)
                    Write_Label(ax, XD[px], YD[px]+height, tt,  fontsize )
    except:
         pass        
    ax.axis('off')     
    ax.set_aspect(1 )
    experiment = fig.savefig(filename + ".pdf", bbox_inches='tight') 
    return experiment


def graphPlotNew(graph, type_photons = None, DistanceOfVertices=0.1,filename='',
               show=True,max_thickness = 5, min_thickness = 2, thickness=10, font_size =12,
              linewidth= 2, de = 5, colors = colors,figsize=10):

    graph = convert_to_fancy_graph(graph)
    GraphEdge = [grouper(2,i)[0] for i in list(sorted(graph.edges))]
    GEC = [grouper(2,i)[1] for i in list(sorted(graph.edges))]
    GraphEdgeColor  =[encoded_label(ED,get_num_label(colors))for ED in GEC  ]
    Num_Vertices = len(np.unique(list(itertools.chain(*GraphEdge))))
    Graphweight = convert_bools_to_ints(graph.weights)
    g = uniqueList(GraphEdge)
    posv = PosOfVertices (Num_Vertices, DistanceOfVertices)
    
    
    for n, eg in enumerate(g):
        posxy = eg[0]
        posxy= correct(posxy)
        c = eg [1]
        updated_posxy = tuple(posv[xy] for xy in posxy)
        updated_color = [[GraphEdgeColor[cc],Graphweight[cc]]  for cc in c]
        g[n] = [updated_posxy, updated_color]
        
    fig, ax = plt.subplots(figsize=(figsize,)*2)
    
    Plot_Vertices(ax, Num_Vertices, DistanceOfVertices,type_photons, font_size = font_size,\
                  linewidth=linewidth)
    max_length = max(g, key=lambda x: len(x[1]))
    max_len = len(max_length[1])
    for eg in g:
        Plot_Edges(ax, eg[0][0], eg[0][1], eg[1], DistanceOfVertices, max_len = max_len,\
                  max_thickness = max_thickness ,\
               min_thickness = min_thickness , thickness=thickness, de = de )   
    ax.set_aspect(1 )
    ax.axis('off') 

    if show:
        plt.show()
        plt.pause(0.01)
    else:
        pass
    if filename:
        fig =fig.savefig(filename + ".pdf", bbox_inches='tight')
    return fig

cols = ['#66c2a5', '#fc8d62', '#8da0cb']


