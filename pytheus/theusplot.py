import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge, Circle, Ellipse
import matplotlib
matplotlib.rcParams['figure.dpi']=300
import  string, itertools
from pytheus.fancy_classes import Graph, invalidInput
import pytheus.theseus as th
from collections import OrderedDict, Counter
from collections.abc import Iterable

Colors = ['dodgerblue', 'firebrick', 'limegreen', 'darkorange', 'purple', 'yellow', 'cyan']
Paths = list(string.ascii_lowercase)

def transparency(W):
    try:
        transparency = 0.2 + abs(W) * 0.8
        transparency = min(transparency, 1)
    except IndexError:
        transparency = 1
    except TypeError:
        transparency = 1  
    return transparency

def get_num_label(labels): 
    num_to_label = dict((num, label) for num, label in enumerate(labels))
    return num_to_label

def encoded_label(nums,labels):# for transform num to alphabet and colors
    encoded_labels =[labels[num] for num in nums]
    return encoded_labels

def gen_list_of_lists(original_list, new_structure): # Create a list according to a new structure
    assert len(original_list) == sum(new_structure)  
    list_of_lists = [[original_list[i + sum(new_structure[:j])]
                      for i in range(new_structure[j])]
                     for j in range(len(new_structure))]  
    return list_of_lists

def union(lst):
    for i in range(len(lst)):
            for j in range(i+1,len(lst)):
                for kk in lst[j]:
                    if kk[0] not in list(itertools.chain(*lst[i]))\
                    and kk[1] not in list(itertools.chain(*lst[i])):
                        lst[i].append(kk)
                        lst[j].remove(kk)
    lst =  list(filter(None, lst))
    return (lst)

def generate_N_grams (position, ngram = 1):
    positions=[pos for pos in position]
    grams=zip(*[positions[i:] for i in range(0,ngram)])
    return list(grams)

def Combine(x): 
    y = (list(itertools.combinations(x,2)))
    return y

def updated_edgeBleach(edges, colors, other_label):#other_label: coord_vertex , paths
        dictted = th.edgeBleach(edges)
        for key in dictted:
            dictted[key] = [encoded_label(num, get_num_label(colors))
                           for num in dictted[key]] 
        keys= list(dictted.keys())
        new_keys =  [tuple(encoded_label(num, get_num_label(other_label)))
                     for num in keys]
        updated_list = list(zip(new_keys, list(dictted.values())))
        return updated_list
    
def new_structure_weights(edges, weights):#according to values of edgeBleach
        dictted = th.edgeBleach(edges)
        leng_ws = []
        for w in list(dictted.values()):
            leng_ws.append(len(w))
        updated_structure = gen_list_of_lists(weights, leng_ws)
        return updated_structure
    
def DuplicateList(lst):#items and index of duplicate items
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
    return(pos_list)

def list_duplicates_of(seq,item):#index of duplicate items
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

def remove_unused(lst1, lst2):
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
def grouper(n, iterable):
    args = [iter(iterable)] * n
    return list(zip(*args))
    
def Pos0fpath(lst, x):#to get the position of the path
    Pospath = []# x is the length of crystal
    d0 = x/10 
    for pos in lst:
        x1 = pos+d0 #  Location of one of the pair photons. For example (idler)
        x2 = pos+x-d0 # location of the signal photon
        Pospath.extend([x1, x2]) 
    return(Pospath)
    
def find_index_duplicate(lists, item):
    index = []
    for idx in range(len(lists)):
        for ele in lists[idx]:
            if ele == item :
                index.append(idx)
    return index
    
def get_index_color(colors, lst_col):
    num_to_color = dict((num, color) for num, color in enumerate(colors))
    color_to_num = {color: num for num, color in num_to_color.items()}
    index_col = encoded_label(lst_col,color_to_num )
    return  index_col

## The  graph plotting class
class GraphPlotter(Graph):
    
    """Graphplotter class draws graphs that can be equivalent to a quantum optics experiment. """
    
    def __init__(self,
                 edges, 
                 dimensions = None,
                 weights = [], 
                 imaginary = False,
                 order = 0,
                 loops = False,
                 edgecolor = Colors,
                 fillvertexcolor = 'white',
                 outvertexcolor = 'black',
                 sidelength = 0.3, #Regular polygon side length for drawing the graph
                 detectortype = None, 
                 figsize = 10,
                 rv = 10, # To determine the radius of the vertex
                 ld = 8, # To determine the length of the diamond
                 vlinewidth = 1, # vertex linewidth
                 fontsize = 12, 
                 textcolor = 'black',
                 de = 5, #To specify the distance between multiple edges 
                 maxthickness = 5,
                 minthickness = 2,
                 thickness = 10,
                 rows = 1,
                 cols = 1, 
                 showPM = False,
                 inherited = False):
        
        super().__init__(edges, dimensions, weights, imaginary, order, loops)
        self.sidelength = sidelength
        self.edgecolor = edgecolor
        self.figsize = figsize
        self.fig, self.ax = None, None
        self.rows = rows
        self.cols = cols
        self.showPM = showPM
        self.inherited = inherited
        if self.showPM:
            self.numPM = (len(self.perfect_matchings))
            if  self.numPM > 1:
                self.rows = round(np.sqrt(self.numPM))
                self.cols = self.rows if self.rows**2 >= self.numPM else self.rows+1
            elif  self.numPM ==1:
                self.rows = rows
                self.cols = cols
            else:
                raise ValueError(invalidInput("It appears that there is"\
                                               " no perfect matching in the graph"))
        else :
            self.rows = rows
            self.cols = cols
        self.circleradius = self.sidelength/rv #If the shape of the vertex is a circle.
        self.squareside = 2*self.circleradius #If the shape of the vertex is a square
        self.triangleside = 2.5*self.circleradius #If the shape of the vertex is a triangle
        self.sizediamond = self.sidelength/ld
        self.detectortype = detectortype
        self.fillvertexcolor =  fillvertexcolor 
        self.outvertexcolor = outvertexcolor 
        self.vlinewidth = vlinewidth
        self.fontsize = fontsize
        self.textcolor = textcolor
        self.diamond_width  = self.sizediamond/2
        self.diamond_height = self.sizediamond
        self.dist =  self.sidelength/de
        self.maxthickness = maxthickness
        self.minthickness = minthickness
        self.thickness = thickness
        self.updated_edgeBleach = updated_edgeBleach(self.edges, self.edgecolor, self.CoordOfVertices)
        self.new_structure_weights = new_structure_weights(self.edges, self.weights)
        
    @property    
    def CoordOfVertices (self):
        self.coordinate = []
        for nv in range(self.num_nodes):
            angle = 2 * nv * np.pi / self.num_nodes
            x = self.sidelength * np.cos(angle)
            y = self.sidelength * np.sin(angle)
            self.coordinate.append((x, y))
        return(self.coordinate)
    
    def __repr__(self):
        return " "
    
    def get_vertex_color(self, lst):
        grouped_list = []
        for inner_list in lst:
            for num in range(0, 2):
                 grouped_list.append([inner_list[num], inner_list[num+2]])
        v, c = list(zip(*sorted( grouped_list)))
        return encoded_label(c, self.edgecolor)
    
    def plot_triangle(self,ax, center, color):
        center_x = center[0]
        center_y = center[1]
        x = [center_x - self.triangleside/2,
             center_x + self.triangleside/2,
             center_x]
        y = [center_y - (3**0.5)*self.triangleside/6,
             center_y - (3**0.5)*self.triangleside/6,
             center_y + (2*(3**0.5)*self.triangleside)/6]
        ax.fill(x, y, color = color, zorder = 3)
        ax.plot(x + [x[0]], y + [y[0]], linewidth = self.vlinewidth,
                     color = self.outvertexcolor, zorder = 3)
        
    def plot_environment (self, ax, center, color, alpha):
        angle_step = 45  # degrees
        angles = np.arange(0, 180, angle_step)
        
        for angle in angles:
             ax.add_patch(Ellipse(center,
                                  2.25*self.circleradius,
                                  self.circleradius,
                                  angle = angle,
                                  alpha = alpha,
                                  ec = color[0],
                                  fc =  color[1],
                                  linewidth = self.vlinewidth,
                                  zorder =3))
        
    def calculate_b(self, ne): #To get the diameter of the ellipse to draw the edge
        #ne : Number of edges in multiple edges (len(colors))
        b = (np.array(range(ne))-0.5*(ne-1)) * self.dist /(np.max(np.arange(1, ne+1))-0.5*(ne-1))
        return b 
               
    def plot_diamond(self, ax, center): 
        center_x = center[0]
        center_y = center[1]
        diamond_x = [center_x,
                     center_x + self.diamond_width / 2,
                     center_x,
                     center_x - self.diamond_width / 2,
                     center_x]
        diamond_y = [center_y + self.diamond_height / 2,
                     center_y,
                     center_y - self.diamond_height / 2,
                     center_y, 
                     center_y + self.diamond_height / 2]
        ax.fill(diamond_x, diamond_y, color ='white', zorder = 3)
        ax.plot(diamond_x, diamond_y, color ='black', linewidth = 1, zorder = 3)
        
    def fun_plot_edges(self,ax, V, edgecol, max_len, w):
        edgewidth = max(min(self.thickness/ max_len, self.maxthickness), self.minthickness)
        V1 = V[0]
        V2 = V[1]
        
        if V1 == V2:
            x = V1[0] + self.circleradius if V1[0] > 0 else V1[0] - self.circleradius
            y = V1[1] + self.circleradius if V1[1] > 0 else V1[1] - self.circleradius
            x1 = x + self.sizediamond/2 if V1[0] > 0 else x - self.sizediamond/2
            y1 = y + self.sizediamond/2 if V1[1] > 0 else y - self.sizediamond/2
            ax.add_patch(Circle((x,y),
                                radius = self.sizediamond/2,
                                facecolor ='white',
                                edgecolor = edgecol[0][0],
                                linewidth = edgewidth,
                                alpha = transparency(w[0])))
            if w[0]< 0:
                 self.plot_diamond(ax, (x1, y1))
            else:
                pass
        else:
            h = (V1[0] + V2[0]) / 2
            k = (V1[1] + V2[1]) / 2
            theta = np.arctan2(V1[1] - V2[1], V1[0]-V2[0])
            a = np.sqrt((V1[0]-V2[0]) ** 2 + (V1[1] - V2[1]) ** 2) / 2
            ne = len(edgecol)
            b = self.calculate_b(ne)
            t1 = np.linspace(0, np.pi / 2, 1000)
            t2 = np.linspace(np.pi, np.pi / 2, 1000)
            
            for ed in range(len(b)):
                xi = h + a * np.cos(theta) * np.cos(t1) - b[ed] * np.sin(theta) * np.sin(t1)
                yi = k + a * np.sin(theta) * np.cos(t1) + b[ed] * np.cos(theta) * np.sin(t1)
                xj = h + a * np.cos(theta) * np.cos(t2) - b[ed] * np.sin(theta) * np.sin(t2)
                yj = k + a * np.sin(theta) * np.cos(t2) + b[ed] * np.cos(theta) * np.sin(t2)
                ax.plot(xi,
                        yi,
                        color = edgecol[ed][0],
                        linewidth = edgewidth,
                        alpha = transparency(w[ed]))
                ax.plot(xj,
                        yj,
                        color = edgecol[ed][1],
                        linewidth = edgewidth,
                        alpha = transparency(w[ed]))
                if w[ed] < 0:
                     self.plot_diamond(ax, (xi[-1], yi[-1]))
                else:
                    pass
                                   
    def plot_edges(self, ax, coordcoloredge, w):
        max_length = max(coordcoloredge, key = lambda x: len(x[1]))
        max_len = len(max_length[1])
        for ii, cce in enumerate(coordcoloredge):
            self.fun_plot_edges(ax, cce[0], cce[1], max_len, w[ii])
                
    def plot_vertices(self, ax, color):
        vertices = self.CoordOfVertices
        if self.detectortype is None:
            
            for i, vertex in enumerate(vertices):
                ax.add_patch(Circle(vertex, 
                                    radius = self.circleradius,
                                    facecolor = color[i],
                                    edgecolor = self.outvertexcolor,
                                    linewidth = self.vlinewidth,
                                    zorder = 3))
                if not self.showPM:
                    ax.text(vertex[0], vertex[1],
                                 str(i), ha ='center',
                                 va ='center', fontsize = self.fontsize,
                                 color = self.textcolor)
                else:
                    pass
                
        elif isinstance(self.detectortype, tuple) or isinstance(self.detectortype, list):
            
            if len(self.detectortype) == 4:
                ancilla, single_emitters, in_nodes, env = self.detectortype
                
                for i, vertex in enumerate(vertices):
                    if i in ancilla:
                        if i in single_emitters:
                            self.plot_triangle(vertex)
                        else:
                            x = vertex[0] - self.circleradius
                            y = vertex[1] - self.circleradius
                            ax.add_patch(Rectangle((x, y), 
                                                    self.squareside, 
                                                    self.squareside,
                                                    facecolor = color[i], 
                                                    edgecolor = self.outvertexcolor, 
                                                    linewidth = self.vlinewidth,
                                                    zorder = 3))                                       
                    elif i in single_emitters or i in in_nodes:
                        self.plot_triangle(ax, vertex, color[i])
                        
                    elif i in env:
                        self.plot_environment(ax, vertex, ('w','w'), 1)
                        self.plot_environment(ax, vertex, (self.outvertexcolor, color[i]), 0.5)
                        
                    else:
                        ax.add_patch(Circle(vertex, 
                                            radius =  self.circleradius,
                                            facecolor = color[i],
                                            edgecolor = self.outvertexcolor,
                                            linewidth = self.vlinewidth,
                                            zorder = 3))
                    if not self.showPM:
                        ax.text(vertex[0], vertex[1],
                                     str(i), ha ='center',
                                     va ='center', fontsize = self.fontsize, 
                                     color = self.textcolor)
                    else:
                        pass
            else:
                raise ValueError(invalidInput('detectortype'))  
        
    def graphPlot(self):
        if not self.showPM:
            self.fig, self.ax = plt.subplots(figsize=(self.figsize,)*2,
                                             nrows = self.rows, ncols = self.cols)  
            
            self.plot_edges(self.ax, self.updated_edgeBleach, self.new_structure_weights)
            self.plot_vertices(self.ax, [self.fillvertexcolor]*self.num_nodes)
            self.ax.set_aspect(1 )
            self.ax.axis('off') 
        else:
            PM_counter = 0
            lmin = self.sidelength + self.sidelength/5
            self.fig, self.ax = plt.subplots(figsize=(self.figsize,)*2,
                                             nrows = self.rows, ncols = self.cols) 
            PM = self.perfect_matchings
            graph = self.graph 
            updated_PM = []
            for p in PM:
                  updated_PM.append(updated_edgeBleach(p, self.edgecolor,
                                                       self.CoordOfVertices))
            weight = []  
            for item in PM:
                w = [[graph[key]] for key in item]
                weight.append(w)
            
            if self.numPM > 1:
                for row in range(self.rows):
                    for col in range(self.cols):
                        if  PM_counter < self.numPM:
                            self.ax[row, col].axis('off')
                            #self.ax[row, col].set_title(f"PM {PM_counter+ 1}", fontsize = self.fontsize)
                            if self.inherited:
                                self.plot_vertices(self.ax[row, col], self.get_vertex_color(PM[PM_counter]))
                            else:
                                self.plot_vertices(self.ax[row, col], [self.fillvertexcolor]*self.num_nodes)
                            self.plot_edges(self.ax[row,col], updated_PM[PM_counter], weight[PM_counter])
                            self.ax[row,col].set_xlim(xmin = -lmin, xmax = lmin)
                            self.ax[row,col].set_ylim(ymin = -lmin, ymax = lmin)
                            self.ax[row,col].set_aspect(1)
                            PM_counter += 1
                for num in range(PM_counter, self.cols*self.rows):
                    self.fig.delaxes(self.ax.flatten()[num])
            else:
                if self.inherited:
                    self.plot_vertices(self.ax, self.get_vertex_color(PM[0]))
                else:
                    self.plot_vertices(self.ax, [self.fillvertexcolor]*self.num_nodes)
                self.plot_edges(self.ax, updated_PM[0], weight[0])
                self.ax.set_aspect(1)
                self.ax.axis('off') 

    def showgraph(self):
        plt.axis('equal')
        plt.axis('off')
        plt.show()
        
    def savegraph(self, filename):
        self.showgraph()
        graph = self.fig.savefig(filename + ".pdf", bbox_inches='tight')
        return graph   

##  The experiment plotting class 
class ExperimentPlotter(Graph):   
    """ExperimentPlotter can translate graphs to experiments based on three approaches:
    path identity, bulk optics, and path encoding."""
    def __init__(self,
                 edges, 
                 dimensions = None,
                 weights = [], 
                 imaginary = False,
                 order = 0,
                 loops = False,
                 width = 0.1,
                 colors = Colors,
                 paths = Paths, 
                 figsize = 10, 
                 fontsize = 12,
                 task = 'BulkOptics'): # else 'PathEncoding'
        
            super().__init__(edges, dimensions, weights, imaginary, order, loops)
            self.fig, self.ax = None, None
            self.width = width
            self.height = self.width/2
            self.colors = colors
            self.paths  = paths      
            self.updated_edgeBleach = updated_edgeBleach(self.edges, self.colors, self.paths)
            self.new_structure_weights = new_structure_weights(self.edges, self.weights)
            self.Remove_Multiedge = list(zip(*self.updated_edgeBleach))[0]
            self.new_structure_colors = list(zip(*self.updated_edgeBleach))[1]
            self.fontsize = fontsize
            self.figsize = figsize
            self.task = task
                 
    def Plot_BS(self, X, Y, color):#The color is used for the color of photon paths
        self.ax.add_patch(Rectangle((X, Y), self.height,
                                    self.height, fc = 'lavender', 
                                    ec = 'navy', angle = 45))
        d0 = np.sqrt(2*self.height**2)/4
        self.ax.vlines(X, ymin = Y, ymax =Y+4*d0, colors ='navy'  )
        self.ax.plot([X+d0, X-d0],[Y+d0, Y+3*d0 ], color = color)
        self.ax.plot([X-d0, X+d0],[Y+d0, Y+3*d0 ], color = color)
        
    def Plot_PBS(self, X, Y, color1, color2): 
        self.ax.add_patch(Rectangle((X, Y), self.height,
                                    self.height, fc = 'thistle',
                                    ec = 'indigo', angle = 45 ))
        d0 = np.sqrt(2*self.height**2)/4
        self.ax.vlines(X, ymin = Y, ymax =Y+4*d0, colors ='indigo')
        self.ax.plot([X+d0, X-d0],[Y+d0, Y+3*d0 ], color = color1)
        self.ax.plot([X-d0, X],[Y+d0, Y+2*d0 ], color = color2)
        self.ax.plot([X, X-d0],[Y+2*d0, Y+3*d0 ], color = color2, linestyle =':')
    
    def Plot_SPDC(self, X, Y, C, W):
        alpha = transparency(W)
        self.ax.add_patch(Rectangle((X, Y),
                                    self.height,self.height,
                                    fc = C[0] , ec = 'none',
                                    alpha = alpha))
        self.ax.add_patch(Rectangle((X+self.height,Y),
                               self.height, self.height, 
                               fc = C[1], ec ='none',
                               alpha=alpha))
        self.ax.add_patch(Rectangle((X, Y),
                                    self.width, self.height,
                                    fc ='none', ec ='black'))

    def Plot_Hline(self, XMIN, XMAX, Y, color): 
        self.ax.hlines(Y, xmin = XMIN, xmax = XMAX, colors = color, zorder = 1)
                          
    def Plot_Vline(self, YMIN, YMAX, X, color): 
        self.ax.vlines(X, ymin = YMIN, ymax = YMAX, colors = color, zorder = 1)  
        
    def Plot_Absorber(self, X, Y) :  
        self.ax.add_patch(Rectangle((X, Y), self.height/2,
                                    self.height/2, fc = 'black',
                                    ec = 'r', joinstyle = 'bevel',
                                    linewidth =2))        
    def Plot_Connection_Line(self, X, Y):
        t = np.linspace(0,1,20)
        self.ax.plot(X[0]+(3*t**2-2*t**3)*(X[1]-X[0]),
                     Y[0]+t*(Y[1]-Y[0]), color='black')  
    
    def Plot_Detector(self, X, Y, leng, step, radius):
        pos = self.Pos_Element(X, step, leng)
        for p in pos:
            self.Plot_Connection_Line([p, p-radius], [Y+radius,Y+2.5*radius] )
            self.ax.add_patch(Wedge((p, Y), radius, 0, 180, fc = 'black', ec = 'black'))
            self.ax.add_patch(Rectangle((p-1.2*radius, Y-radius/2),
                                        2.4*radius, radius/2,
                                        fc = 'black', ec = 'black'))
                          
    def Plot_Crystal(self, X, Y, color, W): #for path identity
        row = len(color)
        column = 2
        y_crystal = self.Pos_Element(Y, self.height/row, row)
        x_crystal = self.Pos_Element(X, self.width/column, column)
        if len (y_crystal) == 1:
            height1 = self.height
        else:    
            height1 = y_crystal[1]-y_crystal[0]
        width1 = x_crystal[1]-x_crystal[0]
        for y, posy in enumerate(y_crystal) :             
            self.ax.hlines(posy, xmin = X, xmax = X+self.width, colors = 'k', linewidth = 0.5)
            for x, posx in enumerate(x_crystal):
                colors = color[y][x]
                self.ax.add_patch(Rectangle((posx, posy),
                                       width1, height1,
                                       fc = colors, ec ='none',
                                       alpha =  transparency(W[y])))                        
        self.ax.add_patch(Rectangle((X, Y), self.width, self.height, 
                                    fc = 'none', ec ='black'))
    
    def Plot_Sorter(self, X, Y, leng, step, color):#for bulk optics
        Pos = self.Pos_Element(X, step, leng)
        xmin = min(Pos)
        xmax = max(Pos)
        self.Plot_Hline(xmin, xmax, Y + self.height/10 , 'k')
        self.Plot_Hline(xmin, xmax, Y + 9*self.height/10 , 'k')
        self.Plot_Hline(xmin, xmax, Y + self.height/2, 'k')
        for p, pos in enumerate(Pos) :
            self.ax.add_patch(Circle((pos + self.height/4, Y+self.height/2),
                                     self.height/4, fc = color[p],
                                     ec = 'k', zorder = 15))
            self.ax.add_patch(Rectangle((pos, Y),
                                        self.height/2, self.height,
                                        fc = 'lightgray', ec = 'k'))
            
    def Plot_Multi_Color_Line(self, X, Y, color, leng, radius):# for bulk optics
        step = 2*self.height/float(leng)
        y = self.Pos_Element(Y, step, leng)
        self.Plot_Detector(X, y[-1]+radius/2, 1, 1, radius)
        loc = generate_N_grams (y, ngram = 2)
        for p, pos in enumerate(loc):
            self.Plot_Vline(pos[0], pos[1], X, color[p])
            
    def Write_Label(self, X, Y, text): # for labelling
        self.ax.text(X, Y, s = text, fontsize = self.fontsize)

    def Pos_Element(self, low, step, leng):
        Pos = []
        if leng == 0:
            Pos = Pos
        elif leng > 0:
            for i in range(leng):
                Pos.append(low)
                low = low + step
        return Pos
    
    def get_color_weight_crystals(self, color_weight):
        self.Layers = self.layers0fcrystal
        cw_spdc =[]
        for layer in self.Layers :
            cw_spdc.append([self.Remove_Multiedge.index(jj)for jj in layer])
        for ii in range(len(cw_spdc)):
            for jj in range(len(cw_spdc[ii])):
                cw_spdc[ii][jj] = color_weight[cw_spdc[ii][jj]]
        return(cw_spdc)
    
    @property    
    def layers0fcrystal(self):
        res = th.findPerfectMatchings(self.Remove_Multiedge)
        if len(res) == 0:
             raise ValueError(invalidInput("It appears that there is"\
                                           " no perfect matching in the graph"))
        else:    
            ll = len(res[0])
            layer0 = []
            other_crystals = []
            while len(res)> 0:
                r = res[0]
                layer0.append(r)
                res = [[ele for j,ele in enumerate(sub) if ele not in r]
                       for i,sub in enumerate(res)]
                
                for item  in res:
                    if 0 < len(item) < ll:
                        other_crystals.append(item)
                res = [item for item in res if len(item)>=ll]
                
            layer1 = [[ele for j, ele in enumerate(sub) if ele 
                       not in list(itertools.chain(*layer0))]
                      for i,sub in enumerate(other_crystals)]
            
            flatten = []
            for nl in layer1:
                for i in range(len(nl)-1, -1, -1):
                    if nl[i] not in flatten:
                        flatten.append(nl[i])
                    else:
                        nl.pop(i)
                        
            layer1= sorted(union(sorted(list(filter(None, layer1)))),
                           key= lambda l: (len(l), l), reverse = True)
            layers = layer0 + layer1
            NotInPM = [edges for edges in self.Remove_Multiedge if 
                          edges not in list(itertools.chain(*layers))]
            
            if len( NotInPM) > 0:
                layers2 = union([NotInPM [i:i+1] for i in range(0, len(NotInPM ),1)])
                Layers =  layers + layers2
            else:
                Layers = layers
                
            Layers = union(Layers)
            return(Layers)
                  
    def PathIdentityPlot(self):
        if self.loops:
             raise ValueError(invalidInput("The graph has self-loops"))         
        else:
            if len(self.perfect_matchings) == 0:
                 raise ValueError(invalidInput("It appears that there is"\
                                               " no perfect matching in the graph"))
            else:
                self.fig, self.ax = plt.subplots(figsize=(self.figsize,)*2)
                self.Layers = self.layers0fcrystal
                self.c_spdc = self.get_color_weight_crystals(self.new_structure_colors)
                self.w_spdc = self.get_color_weight_crystals(self.new_structure_weights)
                Detector =list(itertools.chain(*self.Layers[0]))
                
                PosxSpdc = []
                ys = []
                for layer in self.Layers:
                    numx = len(layer)
                    ys.append(numx)
                    px = self.Pos_Element(0, 3/2*self.width, numx)
                    PosxSpdc.append(px)
                    
                numy = len(PosxSpdc)
                PY = self.Pos_Element(0, 2*self.height, numy)
                PY.sort(reverse=True)
                P0 = PosxSpdc[0]
                PosySpdc=  gen_list_of_lists(np.repeat(PY, ys).tolist(), ys)
                
                pospathx = []
                pospathy = []
                for ii in range(len(PosxSpdc)):
                    x = Pos0fpath(PosxSpdc[ii], self.width)
                    pospathx.append(x)
                    for jj in range(len(PosxSpdc[ii])):
                        self.Plot_Crystal (PosxSpdc[ii][jj],
                                      PosySpdc[ii][jj],
                                      self.c_spdc[ii][jj],
                                      self.w_spdc[ii][jj]) 
                        y1 = PosySpdc[ii][jj] + 0.9*self.height
                        y2 =PosySpdc[ii][jj] + 0.1*self.height
                        pospathy.extend([[y1, y2]])
                        self.Plot_Vline(y1, y2 , x , 'black')
                        
                YDR = max(PY)+2*self.height 
                XDR = Pos0fpath(P0, self.width)
                for pos, xd in enumerate(XDR):
                    self.Plot_Detector(xd, YDR, 1, 1, self.height/4)
                    self.Plot_Vline(pospathy[0][0], YDR, xd, 'black' )
                    self.Write_Label(xd, YDR+self.width/4, Detector[pos])
                    
                lrs = [list(itertools.chain(*pp)) for pp in self.Layers]
                ps = sorted(list(itertools.chain(*self.Layers[0])))
                duplicate0fps= [find_index_duplicate(lrs, pp) for pp in ps]
                
                virtual = []
                for ii, lst in enumerate(duplicate0fps):
                    for idx in range(lst[0], lst[-1]):
                        if idx not in lst:
                            lst.append(idx)
                            self.Layers[idx].append(ps[ii])
                            virtual.append(idx)
                            
                connecty = [[PY[idx] for idx in sorted(duplicate0fps[lst])]
                           for lst in range(len(duplicate0fps))]
                connecty = [grouper(2, sorted(Pos0fpath(lst, self.height ),reverse = True)[1:-1])
                           for lst in connecty]
                connecty = [y for y in connecty if y != []]
                
                y = [PY[idx] for idx in sorted(virtual)]
                y =  grouper(2, Pos0fpath(y, self.height))
                count = dict(Counter(sorted(virtual)))
                ele = list(count.keys())
                num = list(count.values())
                fl =  pospathx[0]
                x = []
                
                for ii, item in enumerate(ele):
                    leng = len(pospathx[item])
                    for jj in range(num[ii]):
                        item = fl[jj+leng]
                        x.append(item)
                        pospathx[ele[ii]].append(item) 
                        
                Pathconnect = list(itertools.chain(*self.Layers))
                Pathconnect = list(itertools.chain(*Pathconnect))
                Connection_Line = DuplicateList(Pathconnect)
                connect = flatten(pospathx)
                
                for ii, line in enumerate(Connection_Line):
                    cl = line[1]
                    for jj in range(len(cl)):
                        cl[jj] = connect[cl[jj]]
                Connection_Line = dict(Connection_Line)
                CL = dict(OrderedDict(sorted(Connection_Line.items())))
                connectx =[generate_N_grams(lst, 2) for lst in list(CL.values())]
                
                for ii in range(len(connectx)):
                    for jj in range(len(connectx[ii])):
                        self.Plot_Connection_Line(connectx[ii][jj],
                                                  connecty[ii][jj]) 
                        
                for ii in range(len(x)):
                    self.Plot_Vline(y[ii][0], y[ii][1], x[ii] ,'k')
                    
                self.ax.set_aspect(1)
                self.ax.axis('off')
                
    def BulkOpticsPathEncodingPlot(self):
        self.fig, self.ax = plt.subplots(figsize=(self.figsize,)*2)
        numofcrystals = len(self.edges)
        coordxcrystals = self.Pos_Element(0, 1.5*self.width , numofcrystals)
        coordycrystals = np.full(numofcrystals, 0) 
        
        colorcrystals = [encoded_label(num,get_num_label(self.colors))for num in
                         [grouper(2,i)[1] for i in self.edges]]
        
        pathcrystals = [encoded_label(num, get_num_label(self.paths)) for num in
                         [grouper(2,i)[0] for i in self.edges]]
        
        coordxpath = Pos0fpath(coordxcrystals, self.width)
        coordypath = np.full(len(coordxpath), self.height) 
        allcolor = list(itertools.chain(*colorcrystals))
        allpath= [] 
        
        for pp in range(numofcrystals):
            allpath.extend([str(pathcrystals[pp][0])+str(pp)
                            ,str(pathcrystals[pp][1])+str(pp)])
        possiblepath  = Combine(allpath)
        possiblecolor = Combine(allcolor) 
        possiblecoordx = Combine(coordxpath)
        #plot crystals
        for coordx, coordy, color, weight \
        in zip(coordxcrystals, coordycrystals, colorcrystals, self.weights):
            self.Plot_SPDC(coordx, coordy, color, weight)
            
        for coordx, coordy, color in zip(coordxpath, coordypath, allcolor):
            self.Plot_Vline(coordy, coordy + 0.75*self.height, coordx, color)
            
        #To draw beam splitters and absorbers
        path_bs = []
        coordx_bs = []
        coordx_a = []
        for (p1, p2), (c1, c2), (cx1, cx2) in\
        zip(possiblepath, possiblecolor, possiblecoordx):
            if p1[0] == p2[0] and c1 == c2:
                coordx_a.append(cx2)
                coordx_bs.append((cx1, cx2))
                path_bs.append((p1, p2))
                
        coordx_ab = list(set(coordx_a))
        coordx =  remove_unused(coordx_ab, coordx_bs)
        
        if len(coordx) > 0:
            P1 , P2 =  list(zip(*coordx)) 
            get_dupl0 = DuplicateList(P1)
            get_dupl = DuplicateList(P1)  
            coordx1 = []
            
            for num, dupl in enumerate(get_dupl):
                x = dupl[1]
                for jj in x:
                    coordx1.append(((P1[jj], P2[jj])))
                    
            coordx2 = [c for c in  coordx if c not in coordx1] 
            coordy0 =  self.Pos_Element(1.75*self.height, self.height/3, len(get_dupl0))

            if len(coordy0)>0:
                coordy1 = self.Pos_Element(max(coordy0)+1.75*self.height,
                                           2.5*self.height, len(coordx1))
            
            if len(coordx1)>0:
                coordy2 = self.Pos_Element(max(coordy1)+3*self.height,
                                           2.5*self.height, len(coordx2))
            elif len(coordx1)==0:
                coordy2 = self.Pos_Element(2.75*self.height, 2.5*self.height, len(coordx2))
                
            d0 = np.sqrt(2*self.height**2)/4
            for ii in range(len(coordx1)):
                    x = coordx1[ii][1]
                    y = coordy1[ii]
                    c = allcolor[coordxpath.index(x)]
                    self.Plot_BS(x-d0, y, c)
                    self.Plot_Vline(1.75*self.height, y+d0, x, c)
                    self.Plot_Vline(y+3*d0, y+1.75*self.height, x-2*d0, c) 
                    self.Plot_Absorber(x-2*d0-self.height/4, y+1.75*self.height) 
     
            xnab2, ynab2, colnab2, pathnab2 = [], [], [], []
            for x, y in zip(coordx2, coordy2):
                ynab2.append(y+3*self.height)
                xnab2.append(x[1])
                c = allcolor[coordxpath.index(x[0])]
                p = allpath[coordxpath.index(x[0])]
                colnab2.append(c)
                pathnab2.append(p)
                self.Plot_Vline(1.75*self.height, y+d0, x[1], c)
                self.Plot_Vline(1.75*self.height, y+d0, x[0], c) 
                self.Plot_Hline(x[1], x[0], y+d0, c)
                self.Plot_Vline(y+3*d0, y+3*self.height, x[1], c,) 
                self.Plot_Vline(y+3*d0, y+1.75*self.height, x[1]-2*d0, c) 
                self.Plot_Absorber(x[1]-2*d0-self.height/4, y+1.75*self.height)
                self.Plot_BS(x[1]-d0, y, c)
                
            coordx0 = [] 
            for ii in range(len(get_dupl)):
                x = get_dupl[ii][1]
                x1 = get_dupl[ii][0]
                coordx0.append(x1)
                for jj in range(len(x)):
                    x[jj] = P2[x[jj]]
                    
            colx0 = [allcolor[coordxpath.index(pos)] for pos in coordxpath if pos in coordx0]
            turn_leng = [len(get_dupl0[ii][1]) for ii in range(len(get_dupl0))]
            
            if len(turn_leng)>0:
                coordy1 = gen_list_of_lists(coordy1, turn_leng)
            else:
                pass
            
            for ii in range(len(get_dupl0)):
                get_dupl0[ii][1] = coordy1[ii]
                get_dupl0[ii][0]= coordy0[ii]  
                
            for x, y, c  in zip (coordx0, coordy0, colx0):
                    self.Plot_Vline(1.75*self.height, y+2*d0, x, c) 
                    
            xh1, yh1, xh2, yh2, ch1 = [], [], [], [], []
            for x, y in zip(get_dupl, get_dupl0):
                xh1.append(flatten(x))
                yh1.append(flatten(y))
                xh2.append(x[1])
                yh2.append(y[1])
                ch1.append([allcolor[coordxpath.index(pos)] for pos in
                         coordxpath if pos in x[1]]) 
                
            for ii, y in enumerate(yh1):
                for jj in range(len(y)):
                    if jj == 0:
                         y[jj] = y[jj]
                    elif jj > 0:
                         y[jj]=y[jj]+d0 
                            
            xnab1, ynab1 = [], []
            for ii in range(len(xh1)):
                xnab1.append(xh1[ii][-1])
                ynab1.append(yh1[ii][-1]+2.5*self.height)
                for jj in range(len(xh1[ii])-1):
                    self.Plot_Hline(xh1[ii][jj], xh2[ii][jj]-2*d0,
                                    yh1[ii][jj]+2*d0, ch1[ii][jj])
                    self.Plot_Vline(yh1[ii][jj]+2*d0, yh2[ii][jj]+d0,
                                    xh2[ii][jj]-2*d0, ch1[ii][jj])
                    
            for pos in range(len(ynab1)):
                self.Plot_Vline(ynab1[pos]-2.5*self.height+2*d0,
                                ynab1[pos], xnab1[pos], colx0[pos])
        #to draw polarizing beam splitters, and detectors
            coordx_nab =  xnab2 + xnab1
            coordy_nab  = ynab2 + ynab1
            
            pathnab1 = [allpath[coordxpath.index(pos)]
                        for pos in coordxpath if pos in coordx0]
            
            path_nab =  pathnab2 + pathnab1
            color_nab = colnab2 + colx0
            
            coordx_r = [pos for pos in coordxpath if pos not in P2+P1]
            
            color_r = [allcolor[coordxpath.index(pos)] for
                       pos in coordxpath if pos in coordx_r ]
            
            path_r = [allpath[coordxpath.index(pos)] for
                      pos in coordxpath if pos in coordx_r]
            
            coordy_r = [1.75*self.height for pos in range(len(coordx_r))] 
            
            CoordX_Concat =  coordx_r + coordx_nab
            CoordY_Concat = coordy_r + coordy_nab   
            Path_Concat = path_r + path_nab
            Color_Concat = color_r + color_nab
            
        elif len(coordx) == 0:
            CoordX_Concat = coordxpath 
            CoordY_Concat = [1.75*self.height for pos in range(len(coordxpath))]   
            Path_Concat = allpath
            Color_Concat = allcolor
            
        Path_alphabet, Path_Number = [], []   
        for path in Path_Concat:
            Path_alphabet.append(path[0])
            Path_Number.append(path[1])
            
        counts = Counter(Path_alphabet)    
        single_path = [[CoordX_Concat[Path_alphabet.index(item)],
                        CoordY_Concat[Path_alphabet.index(item)],
                        Color_Concat[Path_alphabet.index(item)],
                        Path_Concat[Path_alphabet.index(item)]]
                        for item in Path_alphabet if counts[item] <= 1]
        
        for sp in range(len(single_path)):
            self.Plot_Detector(single_path[sp][0], single_path[sp][1], 1,
                          self.height/4, self.height/3)
            self.Write_Label(single_path[sp][0],
                             single_path[sp][1]+self.height/2,
                             single_path[sp][3][0])
            
        get_to_labels = sorted(DuplicateList(Path_alphabet))
        
        if len(get_to_labels) > 0:
            labels = list(zip(*get_to_labels))[1]
            XtoD = sorted(DuplicateList(Path_alphabet))
            get_to_coordx =  [encoded_label(num, get_num_label(CoordX_Concat))
                              for num in labels]
            get_to_coordy = [encoded_label(num, get_num_label(CoordY_Concat))
                              for num in labels]
            get_to_colors = [encoded_label(num, get_num_label(Color_Concat))
                              for num in labels]
            ymax =  max(CoordY_Concat)
            XD = self.Pos_Element(min(CoordX_Concat)+self.height/3,
                                  (max(CoordX_Concat)-min(CoordX_Concat)
                                  -1.75*self.height)/len(get_to_coordx),
                                  len(get_to_coordx))   
            leng = len(flatten(labels))
            new_st = [len(x) for x in labels]
            y = self.Pos_Element(ymax,self.height/3, leng)
            YtoD = gen_list_of_lists(y, new_st) 
            YD = self.Pos_Element(max(y)+self.height, self.height, len( get_to_coordx))
            
            col_index = []
            for ii in range(len(get_to_coordx)):
                XtoD[ii][1] = self.Pos_Element(XD[ii],self.height,len(XtoD[ii][1]))
                col_index.append(get_to_colors[ii])
                for jj in range(len(get_to_coordx[ii])):
                    self.Plot_Vline(get_to_coordy[ii][jj],
                                    ymax,
                                    get_to_coordx[ii][jj],
                                    get_to_colors[ii][jj])
                    self.Plot_Vline(ymax,
                                    YtoD[ii][jj],
                                    get_to_coordx[ii][jj],
                                    get_to_colors[ii][jj])
                    self.Plot_Vline(YD[ii],
                                    YtoD[ii][jj],
                                    XtoD[ii][1][jj],
                                    get_to_colors[ii][jj])
                    self.Plot_Hline(get_to_coordx[ii][jj],
                                    XtoD[ii][1][jj],
                                    YtoD[ii][jj],
                                    get_to_colors[ii][jj])
            pathx_sort = []  
            for col in range(len(col_index)):
                x = XtoD[col][1]
                col_index[col] = get_index_color(self.colors, col_index[col])
                min_index  = col_index[col].index(min(col_index[col]))
                pathx_sort.append(x[min_index])
           
            for px in range(len(get_to_coordx)):
                if self.task:
                    self.Plot_Sorter(XD[px]-self.height/4,
                                     YD[px],
                                     len(get_to_coordx[px]),
                                     self.height,
                                     get_to_colors[px])
                    self.Plot_Multi_Color_Line(pathx_sort[px],
                                               YD[px]+self.height,
                                               get_to_colors[px],
                                               len(get_to_coordx[px])+1,
                                               self.height/3)
                    self.Write_Label(pathx_sort[px],
                                     YD[px]+3.5*self.height,
                                     XtoD[px][0])
                else:
                    self.Plot_Detector(XD[px],
                                       YD[px],
                                       len(get_to_coordx[px]),
                                       self.height,
                                       self.height/3)
                    self.Write_Label(XD[px],
                                     YD[px]+self.height,
                                      XtoD[px][0])
        else:
            pass 
        self.ax.set_aspect(1)
        self.ax.axis('off')
          
    def showexperiment(self):
        plt.axis('equal')
        plt.axis('off')
        plt.show()
        
    def saveexperiment(self, filename):
        self.showexperiment()
        experiment = self.fig.savefig(filename + ".pdf", bbox_inches='tight')
        return experiment  
 
