import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from collections import Counter
import matplotlib.collections as collections

import theseus as th


def drawEdge(edge,verts,ind,mult,scale_max=None,max_thickness = 10):
    colors = ['blue','red','green','darkorange','purple','yellow','cyan']
    col1 = colors[int(edge[2])]
    col2 = colors[int(edge[3])]

    vert1 = np.array(verts[int(edge[0])])
    vert2 = np.array(verts[int(edge[1])])
    if not np.array_equal(vert1, vert2):
        diff = vert1-vert2
        rect = [diff[1],-diff[0]]
        rect /= np.linalg.norm(rect)
        hp = (vert1+vert2)/2 + (2*ind-mult+1)*0.05*rect
    else:
        hp = vert1*1.2
    
    if scale_max is None:
        lw =max_thickness
        
    else:
        lw = np.max([abs(max_thickness*edge[4])/scale_max,0.5])
    
    plt.plot([vert1[0], hp[0]], [vert1[1], hp[1]],color=col1, linewidth=lw)
    plt.plot([hp[0], vert2[0]], [hp[1], vert2[1]],col2, linewidth=lw)
    try:
        if edge[4]<0:
            plt.plot(hp[0],hp[1],marker="d",markersize=25,markeredgewidth = "6",markeredgecolor="black",color="white")
    except:
        pass


def graphPlot(edge_list,scaled_weights=False,show=True,max_thickness = 10,weight_product=False):
    edge_dict = {}
    for edge in edge_list:
        try:
            edge_dict[edge[:2]].append(edge)
        except:
            edge_dict[edge[:2]]=[edge]

    count = 0
    for key in edge_dict.keys():
        count += len(edge_dict[key])

    n = len(np.unique(np.array(edge_list)[:,:2]))

    angles = np.linspace(0,2*np.pi*(n-1)/n,n)

    rad = 0.9
    vertcoords = []
    for angle in angles:
        x = rad*np.cos(angle)
        y = rad*np.sin(angle)
        vertcoords.append(tuple([x,y]))

    vertnums = list(range(n))
    verts = dict(zip(vertnums,vertcoords))
    
    if scaled_weights:
        try:
            scale_max = np.max(np.abs(np.array(edge_list)[:,4]))
        except:
            scale_max = None
    else:
        scale_max = None
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    for uc_edge in edge_dict.keys():
        mult = len(edge_dict[uc_edge])
        for ii, edge in enumerate(edge_dict[uc_edge]):
            drawEdge(edge,verts,ii,mult,scale_max=scale_max,max_thickness = max_thickness)

    circ = []
    for vert,coords in verts.items():    
        circ.append(plt.Circle(coords, 0.1, alpha=0.5))
        plt.text(coords[0],coords[1],str(vert),zorder=11,ha='center', va='center',size = 30)
    circ=collections.PatchCollection(circ, zorder=10)
    ax.add_collection(circ)

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    plt.axis('off')

    if weight_product:
        wp = round(np.product(np.array(edge_list)[:,4]),2)
        plt.title('weight ='+str(wp),fontsize=30)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    return(fig)
