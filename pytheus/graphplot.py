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

def Plot_Crystal (ax, X, Y, color, width, height, W, wmax =1): #for path identity
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
        
def Write_Label(ax, X, Y, text, fontsize ):
   ax.text(X,Y, s= text, fontsize = fontsize)
    
######################################################################
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

def StringT0Tuple(Graph):
    key = list(Graph.keys())
    for ii in range (len(key)):
        key[ii] = literal_eval(key[ii])
    graph = dict(zip(key,list(Graph.values()))) 
    return graph

def generate_N_grams (position,ngram = 1):
    positions=[pos for pos in position]
    grams=zip(*[positions[i:] for i in range(0,ngram)])
    return list(grams)

#index 0 :path, 1: color
def GetGraphColorEdge(Graph, index, PC):
    GraphED = [grouper(2,i)[index] for i in list(sorted(Graph.keys()))]
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
  ################################################################################################################
def PerfectMatching (GraphEdgesAlphabet, Numphoton):
    Remove_Duplicate = list(GraphEdgesAlphabet for GraphEdgesAlphabet\
                            ,_ in itertools.groupby(GraphEdgesAlphabet))
    Com = list(itertools.combinations(Remove_Duplicate,int(Numphoton/2)))
    perfect_matching= []
    for ii in range(len(Com)):
        a  = list(itertools.chain(*Com[ii]))
        count = LengDuplicate(a)
        if count ==0:
            perfect_matching.append(list(Com[ii]))
    return(perfect_matching)

def layer0fcrystal (lst, Numphoton):
    res = lst
    ll = int(Numphoton/2)
    remian= []
    while len(res)>0:
        r = res[0]
        remian.append(r)
        res = [[ele for j,ele in enumerate(sub) if ele not in r] for i,sub in enumerate(res)]
        res = [item for item in res if len(item)>=ll]
    return(remian)

def Get_Color_Weight_Crystals(gea, Numphoton, gcw, Layers):
    colwei =  uniqueList(gea)
    for ii in range(len(colwei)):
        x =colwei[ii][1]
        for jj in range(len(x)):
            x[jj]=gcw[x[jj]]     
    Remove_Duplicate = list(gea for gea,_ in itertools.groupby(gea))
    cw_spdc = []
    for ii in Layers :
        cw = [Remove_Duplicate.index(jj) for jj in ii]
        cw_spdc.append(cw)
    wc =[]
    for ii in range(len(colwei)):
        wc.append(colwei[ii][1])

    for ii in range(len(cw_spdc)):
        wcspdc = cw_spdc[ii]
        for jj in range(len(wcspdc)):
            wcspdc[jj] =wc[wcspdc[jj]]
    return(cw_spdc)
  
def Plot_Path_Identity(graph,  filename, width, figsize , fontsize, colors, Paths ):
    fig,ax=plt.subplots(ncols=1,nrows=1,figsize= figsize, facecolor='w')
    Graph =StringT0Tuple(graph)
    Edge= list(Graph.keys())
    Edge.sort()
    Graph = {i: Graph[i] for i in Edge}
    GraphEdgesAlphabet = GetGraphColorEdge(Graph, 0, Paths )
    GraphEdgesColor  =  GetGraphColorEdge(Graph, 1, colors )
    Graphweight= list(Graph.values())
    Num0fCrystal = len(Graph)
    Dimension = len(np.unique(list(itertools.chain(*GraphEdgesColor))))
    Numphoton =  len(np.unique(list(itertools.chain(*GraphEdgesAlphabet))))
    #width = 0.1
    height = width/2
    wmax=0.0 # for finding the maximum weight
    for w in range(len(Graphweight)):
        wmax=np.maximum(wmax,np.max(np.abs(Graphweight[w])))
        
    Layers = layer0fcrystal(PerfectMatching (GraphEdgesAlphabet, Numphoton), Numphoton)
    color_spdc = Get_Color_Weight_Crystals(GraphEdgesAlphabet, Numphoton, GraphEdgesColor, Layers)
    w_spdc =Get_Color_Weight_Crystals(GraphEdgesAlphabet, Numphoton, Graphweight, Layers)
    Detector =list(itertools.chain(*Layers[0]))
    numX = int(Numphoton/2)
    PX = Pos_Element(0, 3/2*width , numX)
    PosxSpdc = list(itertools.repeat(PX,len(Layers)))
    numY = len(PosxSpdc)
    PY = Pos_Element(0, 2*height , numY)
    PY.sort(reverse=True)
    PosySpdc= [a for a in PY for i in range(numX )]
    PosySpdc = grouper(numX,  PosySpdc )
    YDR = max(PY)+2*height 
    XDR = Pos0fpath(PX, width)
    connectx =  grouper(2,  XDR)
    connectx = list(itertools.repeat(connectx ,len(Layers)))
    
    CY = []
    for ii in range(len(PY)):
        y1 = PY[ii]+height+height/10
        y2 =PY[ii]-height/10
        CY.extend([[y1, y2]])
    CY[-1][1]= CY[-1][1]+height/5
    connecty = flatten(CY )
    del(connecty[0])
    del (connecty[-1])
    connecty = grouper(2, connecty )

    Pathconnect = list(itertools.chain(*Layers))
    Pathconnect = list(itertools.chain(*Pathconnect))
    Connection_Line = DuplicateList(Pathconnect)
    connectx = flatten(connectx)

    for ii in range(len(Connection_Line)):
        cl = Connection_Line[ii][1]
        for jj in range(len(cl)):
            cl[jj] = connectx[cl[jj]]

    for jj in range(len(Connection_Line)):
        Connection_Line[jj][1] = generate_N_grams(Connection_Line[jj][1],ngram = 2)
        CL = Connection_Line[jj][1]
        for jj in range(len(CL)):
            Plot_Connection_Line(ax,CL[jj],connecty[jj])

    for ii in range(len(PosxSpdc)):
        for jj in range(len(PosxSpdc[ii])):
            Plot_Crystal (ax, PosxSpdc[ii][jj], PosySpdc[ii][jj], color_spdc[ii][jj], width, height, w_spdc[ii][jj], wmax = wmax) 

    for pos in range (len(XDR)):
        Plot_Detector(ax , XDR[pos], YDR, 1,1, height/4 )
        Plot_Vline(ax , CY[0][1], YDR, XDR[pos], 'k' )
        Write_Label(ax,  XDR[pos], YDR+width/4, Detector[pos] , fontsize )

    for posx in range(len(XDR)):
        for posy in range(len(CY)):
            Plot_Vline(ax , CY[posy][0], CY[posy][1], XDR[posx] , 'k')
            
    ax.set_aspect( 1 )        
    ax.axis('off') 
    experiment = fig.savefig(filename + ".pdf", bbox_inches='tight') 
 
    return experiment
   
