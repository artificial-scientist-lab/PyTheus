Starting from the original graph:
```
original = Graph({ # one must use the Graph class from the PyTheus library
        "(3, 5, 2, 0)": 0.0004883277677380273,
        "(0, 1, 1, 1)": 0.001051425666873128,
        "(1, 2, 3, 3)": 0.001067998490104688,
        "(0, 2, 0, 0)": -0.00450451217873684,
        "(1, 3, 0, 0)": 0.23292250844790044,
        "(4, 5, 0, 0)": -0.6034112926239387,
        "(2, 4, 2, 0)": 0.6549771937248401,
        "(0, 4, 2, 0)": 0.6648696008295439,
        "(1, 5, 2, 0)": 0.8981955391748329,
        "(1, 2, 2, 2)": 0.9749529126983252,
        "(0, 3, 3, 3)": -0.9823837610559265,
        "(0, 1, 2, 2)": 0.9896780534447832,
        "(2, 3, 1, 1)": -0.9978710974509947
    })
```
We rescale all weights with `-1`, which leads to the same state: `original.rescale(-1)`.
We rename the dimensions for the first 4 nodes. Again, it leads to the same GHZ state.
```
for node in range(4):
    original.switchColors(node,0,2)
```
We rotate the first 4 nodes.
```
original.permuteNodes(0,1)
original.permuteNodes(1,2)
original.permuteNodes(2,3)
```
Finally, we flip the signs of the edges connected to the current nodes 1, 2, and 3.
```
for node in range(1,4):
    original.flipNode(node)
```
This leads to the plotted graph:
``` 
{                                           # Analytical solution:
    "(0, 1, 0, 0)": 0.9749529126983252,     # 1
    "(0, 1, 3, 3)": 0.001067998490104688,   # 2*eps
    "(0, 2, 2, 2)": 0.23292250844790044,    # 1
    "(0, 3, 0, 0)": 0.9896780534447832,     # 1
    "(0, 3, 1, 1)": 0.001051425666873128,   # 2*eps
    "(0, 5, 0, 0)": -0.8981955391748329,    # -1
    "(1, 2, 1, 1)": 0.9978710974509947,     # 1
    "(1, 3, 2, 2)": 0.00450451217873684,    # 2*eps
    "(1, 4, 0, 0)": 0.6549771937248401,     # 1
    "(2, 3, 3, 3)": 0.9823837610559265,     # 1
    "(2, 5, 0, 0)": 0.0004883277677380273,  # eps
    "(3, 4, 0, 0)": 0.6648696008295439,     # 1
    "(4, 5, 0, 0)": 0.6034112926239387      # 1
}
```
