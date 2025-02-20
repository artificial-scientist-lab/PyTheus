import os

import sys
import itertools
import numpy as np
import subprocess
from shutil import which

docstring = """
Use this script like this

Or get the data into a file which list the data of the edges like:
weight vertex1 mode1 vertex2 mode2
weight vertex1 mode1 vertex2 mode2

python leiwand.py in=filename

You can call the script with several options like this
python leiwand.py in=filename key1=value1 key2=value2

option keys are the following
"""

variables = {
    "line_width": 4.0,
    "radius": 3,
    "col0": "{RGB}{0,0,204}",
    "col1": "{RGB}{204,0,0}",
    "col2": "{RGB}{0,204,0}",
    "col3": "{RGB}{255, 165, 0}",
    "col4": "{RGB}{128,0,128}",
    "col5": "{RGB}{255, 255, 0}",
    "col6": "{RGB}{102, 0, 102}",
    "vertexcolor": "{RGB}{250,250,250}",
    "fontcolor": "{RGB}{0,0,0}",
    "angle": 0,
    "vertices": None,
    "whitespace": "10pt"
}


def leiwand(data, name='graph'):
    poly = {}
    output = name
    numcolors = 7

    external_vertices = None
    if variables['vertices'] is not None:
        external_vertices = variables["vertices"].split(' ')
        # reverse order (drawing is counter-clockwise)
        external_vertices = list(reversed(external_vertices))
        # print("got vertices: ", external_vertices)

    whitespace = None
    if variables["whitespace"] is not None:
        whitespace = variables["whitespace"]
    with open(output + ".tex", "w") as outf:
        optionmap = {tuple([c1, c2, True]): f"bicolor={{col{c1}}}{{col{c2}}}" for c1, c2 in
                     itertools.permutations(range(numcolors), 2)}
        for ii in range(numcolors):
            optionmap[(ii, ii, True)] = f"bicolor={{col{ii}}}{{col{ii}}}"
        optionmap.update({tuple([c1, c2, False]): f"bicolor_neg={{col{c2}}}{{col{c1}}}" for c1, c2 in
                          itertools.product(range(numcolors), repeat=2)})
        # if use f"bicolor_neg={{col{c1}}}{{col{c2}}}" will inverse the color for the negitive edges, don't know why; 
        # print(optionmap)
        if whitespace is not None:
            print("\documentclass[border={}]{}".format(whitespace, r"{standalone}"), file=outf)
        else:
            print(r"\documentclass{standalone}", file=outf)
        print(r"""
        
         \usepackage{tikz}
         \usetikzlibrary{decorations.markings, shapes.geometric} 
         
        """, file=outf)

        colors = r"\definecolor{vertexcol}" + variables["vertexcolor"]
        for ii in range(numcolors):
            colors += f"\definecolor{{col{ii}}}" + variables[f"col{ii}"]
        colors += r"\definecolor{fontcolor}" + variables["fontcolor"]
        print(colors, file=outf)

        print(r"""
        \begin{document}
      
        \tikzset{
            bicolor/.style n args={2}{
                postaction={draw=#1, decoration={curveto,  post=moveto, post length=0.5*\pgfmetadecoratedpathlength}, decorate},
                postaction={draw=#2, decoration={curveto,  pre=moveto, pre length=0.5*\pgfmetadecoratedpathlength}, decorate},
            },
            bicolor_neg/.style n args={2}{
                postaction={draw=#1, decoration={curveto,  pre=moveto, pre length=0.5*\pgfmetadecoratedpathlength}, decorate},
                postaction={draw=#2, decoration={curveto,  post=moveto, post length=0.5*\pgfmetadecoratedpathlength}, decorate},
                postaction={decoration={markings, mark=at position 0.5 with {\node[diamond, fill=white, aspect=0.7,draw, thin, minimum width=10pt,opacity=1] {};}}, decorate},
            },
            vertex/.style={circle, draw=black, ultra thick ,fill=vertexcol!80,minimum size=15pt},
        }

        \begin{tikzpicture}
        
        """, file=outf)

        vertices = []
        weights = []
        for d in data:
            weights.append(abs(d[0]))
            vertices.append(d[1])
            vertices.append(d[3])
        vertices = list(set(vertices))
        max_weight = max(weights)

        # check if vertices where specified manually
        if external_vertices is not None:
            print("replacing: ", vertices)
            print("with external vertices: ", external_vertices)
            vertices = external_vertices
        else:
            # sort vertices alphabetically
            vertices = list(sorted(vertices))

        if len(poly) < len(vertices):
            # poly = Polygon.regular(len(vertices), radius=variables["radius"], angle=float(180))
            angles = [2 * np.pi * ii / len(vertices) for ii in range(len(vertices))]
            poly = [tuple([variables["radius"] * np.cos(theta - variables["angle"]),
                           variables["radius"] * np.sin(theta - variables["angle"])]) for theta in angles]
        else:
            # sort alphabetically
            poly = reversed(list(dict(sorted(poly.items(), key=lambda x: x[0])).values()))
        for i, coord in enumerate(poly):
            print(r"\node[vertex] ({name}) at ({x},{y}) {xname};".format(name=vertices[i],
                                                                         xname=r"{\color{fontcolor}" + vertices[
                                                                             i] + "}", x=coord[0], y=coord[1]),
                  file=outf)

        # edge_string = r"\path ({v1}) edge[{options}, opacity={opacity}] ({v2});"
        edge_string = r"\path[{options}, opacity={opacity}] ({v1}) to ({v2});"

        for d in data:
            assert (len(d) == 6)
            weight = d[0]
            v1 = d[1]
            t1 = d[2]
            v2 = d[3]
            t2 = d[4]
            b = d[5]
            opacity = max(0.3, abs(weight) / max_weight)
            angles = [360 * ii / len(vertices) for ii in range(len(vertices))]
            weight_pos = (weight > 0)
            if v1 == v2:  # loop
                angle1 = angles[int(v1)] + 30
                angle2 = angles[int(v1)] - 30

                print(edge_string.format(v1=v1, v2=v2,
                                         options="line width={lw},".format(lw=variables["line_width"]) + optionmap[
                                             (t1, t2, weight_pos)] + ", looseness=" + str(
                                             b) + f",right,out={angle1},in={angle2}",
                                         opacity=opacity), file=outf)
            else:
                print(edge_string.format(v1=v1, v2=v2,
                                         options="line width={lw},".format(lw=variables["line_width"]) + optionmap[
                                             (t1, t2, weight_pos)] + ", bend right=" + str(b),
                                         opacity=opacity), file=outf)
        print(r"""
        \end{tikzpicture}

        \end{document}
        """, file=outf)

    # Get the current working directory
    cwd = os.getcwd()

    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    print("created {}.tex".format(output))
    print("trying to compile with pdflatex ... might be caught in endless loop")

    import subprocess
    from shutil import which

    system_has_pdflatex = which("pdflatex") is not None
    if not system_has_pdflatex:
        raise Exception("You need pdflatex in order to export circuits to pdfs")

    with open(output + ".log", "w") as file:
        subprocess.call(["pdflatex", output + ".tex"], stdout=file)

    print("created {}.pdf".format(output))


def leiwandBulk(data, cnfg, name='graph', root=""):
    #go into directory where graph should be saved
    os.chdir(os.path.join(os.getcwd(), root))
    poly = {}
    output = name
    numcolors = 7
    # defining shapes for different kinds of vertices
    shape_dict = {'in': 'inner sep=0.1em, regular polygon,regular polygon sides=3', 'out': "circle", "anc": "rectangle", 'sps': 'inner sep=0.1em,regular polygon,regular polygon sides=3',
                  'mix': 'star,star points=10'}

    external_vertices = None
    if variables['vertices'] is not None:
        external_vertices = variables["vertices"].split(' ')
        # reverse order (drawing is counter-clockwise)
        external_vertices = list(reversed(external_vertices))
        # print("got vertices: ", external_vertices)

    whitespace = None
    if variables["whitespace"] is not None:
        whitespace = variables["whitespace"]
    with open(output + ".tex", "w") as outf:
        optionmap = {tuple([c1, c2, True]): f"bicolor={{col{c1}}}{{col{c2}}}" for c1, c2 in
                     itertools.permutations(range(numcolors), 2)}
        for ii in range(numcolors):
            optionmap[(ii, ii, True)] = f"bicolor={{col{ii}}}{{col{ii}}}"
        optionmap.update({tuple([c1, c2, False]): f"bicolor_neg={{col{c2}}}{{col{c1}}}" for c1, c2 in
                          itertools.product(range(numcolors), repeat=2)})
        # if use f"bicolor_neg={{col{c1}}}{{col{c2}}}" will inverse the color for the negitive edges, don't know why;
        # print(optionmap)
        if whitespace is not None:
            print("\documentclass[border={}]{}".format(whitespace, r"{standalone}"), file=outf)
        else:
            print(r"\documentclass{standalone}", file=outf)
        print(r"""

         \usepackage{tikz}
         \usetikzlibrary{decorations.markings, shapes.geometric} 

        """, file=outf)

        colors = r"\definecolor{vertexcol}" + variables["vertexcolor"]
        for ii in range(numcolors):
            colors += f"\definecolor{{col{ii}}}" + variables[f"col{ii}"]
        colors += r"\definecolor{fontcolor}" + variables["fontcolor"]
        print(colors, file=outf)

        print(r"""
        \begin{document}

        \tikzset{
            bicolor/.style n args={2}{
                postaction={draw=#1, decoration={curveto,  post=moveto, post length=0.5*\pgfmetadecoratedpathlength}, decorate},
                postaction={draw=#2, decoration={curveto,  pre=moveto, pre length=0.5*\pgfmetadecoratedpathlength}, decorate},
            },
            bicolor_neg/.style n args={2}{
                postaction={draw=#1, decoration={curveto,  pre=moveto, pre length=0.5*\pgfmetadecoratedpathlength}, decorate},
                postaction={draw=#2, decoration={curveto,  post=moveto, post length=0.5*\pgfmetadecoratedpathlength}, decorate},
                postaction={decoration={markings, mark=at position 0.5 with {\node[diamond, fill=white, aspect=0.7,draw, thin, minimum width=10pt,opacity=1] {};}}, decorate},
            },
            vertex/.style={circle, draw=black, ultra thick ,fill=vertexcol!80,minimum size=15pt},
        }

        \begin{tikzpicture}

        """, file=outf)

        vertices = []
        weights = []
        for d in data:
            weights.append(abs(d[0]))
            vertices.append(d[1])
            vertices.append(d[3])
        vertices = list(set(vertices))
        max_weight = max(weights)

        # check if vertices where specified manually
        if external_vertices is not None:
            print("replacing: ", vertices)
            print("with external vertices: ", external_vertices)
            vertices = external_vertices
        else:
            # sort vertices alphabetically
            vertices = list(sorted(vertices))

        if len(poly) < len(vertices):
            # poly = Polygon.regular(len(vertices), radius=variables["radius"], angle=float(180))
            angles = [2 * np.pi * ii / len(vertices) for ii in range(len(vertices))]
            poly = [tuple([variables["radius"] * np.cos(theta - variables["angle"]),
                           variables["radius"] * np.sin(theta - variables["angle"])]) for theta in angles]
        else:
            # sort alphabetically
            poly = reversed(list(dict(sorted(poly.items(), key=lambda x: x[0])).values()))
        for i, coord in enumerate(poly):
            print(r"\node[vertex] ({name}) at ({x},{y}) [{shape}] {xname};".format(name=vertices[i], shape=shape_dict[
                cnfg["vert_types"][i]], xname=r"{\color{fontcolor}" + vertices[i] + "}", x=coord[0], y=coord[1]),
                  file=outf)

        # edge_string = r"\path ({v1}) edge[{options}, opacity={opacity}] ({v2});"
        edge_string = r"\path[{options}, opacity={opacity}] ({v1}) to ({v2});"

        for d in data:
            assert (len(d) == 6)
            weight = d[0]
            v1 = d[1]
            t1 = d[2]
            v2 = d[3]
            t2 = d[4]
            b = d[5]
            opacity = max(0.3, abs(weight) / max_weight)
            angles = [360 * ii / len(vertices) for ii in range(len(vertices))]
            weight_pos = (weight > 0)
            if v1 == v2:  # loop
                angle1 = angles[int(v1)] + 30
                angle2 = angles[int(v1)] - 30

                print(edge_string.format(v1=v1, v2=v2,
                                         options="line width={lw},".format(lw=variables["line_width"]) + optionmap[
                                             (t1, t2, weight_pos)] + ", looseness=" + str(
                                             b) + f",right,out={angle1},in={angle2}",
                                         opacity=opacity), file=outf)
            else:
                print(edge_string.format(v1=v1, v2=v2,
                                         options="line width={lw},".format(lw=variables["line_width"]) + optionmap[
                                             (t1, t2, weight_pos)] + ", bend right=" + str(b),
                                         opacity=opacity), file=outf)
        print(r"""
        \end{tikzpicture}

        \end{document}
        """, file=outf)

    # Get the current working directory
    cwd = os.getcwd()

    # Print the current working directory
    # print("Current working directory: {0}".format(cwd))

    # print("created {}.tex".format(output))
    # print("trying to compile with pdflatex ... might be caught in endless loop")

    system_has_pdflatex = which("pdflatex") is not None
    if not system_has_pdflatex:
        raise Exception("You need pdflatex in order to export circuits to pdfs")

    with open(output + ".log", "w") as file:
        subprocess.call(["pdflatex", output + ".tex"], stdout=file)

    print("created {}.pdf".format(output))


if __name__ == "__main__":
    print(docstring)
    for k, v in variables.items():
        print("{}={}".format(k, v))

    print("calling with: ", sys.argv)
    leiwand(sys.argv)
