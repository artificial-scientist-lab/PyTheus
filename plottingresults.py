import theseus as th
import graphplot as gp
import matplotlib.pyplot as plt

import numpy as np

# +
# options
# scalable linewidth by weight
# convert different formats [(v1,v2,c1,c2),w] or (v1,v2,c1,c2,w) or (v1,v2,c1,c2)
# show plots or save plots to file
# give edge_list and order -> arrange by state produced

# todo: nice self-loops (distinguish multicolor)
# -

# define graph and plot
# thickness can be scaled by weights
# diamond markers note negative weights
el = [[(0, 1, 2, 2), 0.5], [(0, 3, 0, 0), 1.0], [(0, 4, 1, 1), 0.5], [(1, 4, 0, 0), 1.0], [(1, 6, 1, 0), -1.0],
      [(1, 7, 1, 0), 1.0], [(2, 3, 2, 2), -1.0], [(2, 7, 1, 0), 1], [(3, 5, 1, 1), -1.0], [(5, 6, 2, 0), 1],
      [(5, 7, 2, 0), -1.0], [(6, 7, 0, 0), -1.0]]
el = [tuple(e[0] + tuple([e[1]])) for e in el]
graph = gp.graphPlot(el, scaled_weights=True, show=True, max_thickness=10)

# define covers to plot
cat = th.stateCatalog(th.findEdgeCovers([e[:] for e in el], order=0))

# plot all covers and compute weight product
for kk, vv in cat.items():
    for ii, cover in enumerate(vv):
        gp.graphPlot(cover, show=True, weight_product=True)
    plt.show()


def permuteEdgeList(edge_list, positions):
    # TODO
    pass
