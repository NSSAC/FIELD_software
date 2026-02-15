DESC='''Given node and edge attributes, create tikz visualization. The
main script is just an example that demos many functionalities. To fully
utilize the features, import this file and use it in a script.

By: AA

Pending:
    - labels outside
    - edge labels
    - pajek style edges
    - multiple networks
    - edge weights helper functions
    - spring layout with edge weights
'''

import argparse
import logging
from math import atan2
import networkx as nx
import numpy as np
from os import devnull
import pandas as pd
from pdb import set_trace
from random import seed
from re import sub
from scipy.spatial import ConvexHull
from subprocess import check_call

GRAPH_COUNT = 1

FORMAT = "[%(filename)s] [%(levelname)s]: %(message)s"
PREAMBLE = r'''
\documentclass[tikz,border=2]{standalone}
\usetikzlibrary{shadows,arrows,shapes,positioning,calc,backgrounds,fit}
\usepackage{amssymb}
\usepackage{array}
\usepackage{colortbl}
\pdfpageattr {/Group << /S /Transparency /I true /CS /DeviceRGB>>}
\newcommand{\lens}[7]{ % ux,uy,vx,vy,in,out,color
\path[fill=#7,out=#5,in=#6] (#1,#2) -- (#3,#4);
}
\newcommand{\lensarrow}[7]{ % ux,uy,vx,vy,in,out,arrowwidth
\draw[-{Latex[width=#7},out=#5,in=#6] (#1,#2) -- (#3,#4);
}

\pgfdeclarelayer{nl}
\pgfdeclarelayer{el}
\pgfdeclarelayer{bg}
\pgfdeclarelayer{fg}
\pgfsetlayers{bg,el,main,nl,fg}
'''

BEGIN_STATEMENT = r'''
\begin{document}
\tikzset{>=latex, shorten >=.5pt, shorten <=.5pt}
\begin{tikzpicture}
'''

END_STATEMENT = r'''
\end{tikzpicture}
\end{document}
'''

DEFAULT_NODE_STYLE = 'draw'
DEFAULT_EDGE_STYLE = ''
DEFAULT_EDGE_DRAW = ''
DEFAULT_NODE_LABEL = ''
DEFAULT_EDGE_LABEL = ''
DEFAULT_EDGE_LABEL_STYLE = ''
DEFAULT_COMMUNITY_STYLE = 'draw'

LENS_THICKNESS_FACTOR = 20
LENS_INNER_FACTOR = 10
LENS_NODE_DISP = .2

# The GraphToDraw class contains properties and methods to
# - generate coordinates for each node
# - set those individual node and edge styles which are difficult to set 
#   directly.
# - set global styles
class GraphToDraw:
    def __init__(self, nodes, edges, **kwargs):
        global GRAPH_COUNT
        if 'name' in kwargs.keys():
            self.name = kwargs['name']
        else:
            self.name = f'{GRAPH_COUNT}'
        GRAPH_COUNT += 1
        self.nodes = nodes

        self.edges = edges
        if 'directed' in kwargs.keys():
            self.directed = kwargs['directed']
        else:
            self.directed = False

        if 'style' not in self.nodes.columns:
            self.nodes['style'] = DEFAULT_NODE_STYLE
        if 'style' not in self.edges.columns:
            self.edges['style'] = DEFAULT_EDGE_STYLE
        if 'draw' not in self.edges.columns:
            self.edges['draw'] = DEFAULT_EDGE_DRAW
        if 'label' not in self.nodes.columns:
            self.nodes['label'] = DEFAULT_NODE_LABEL
        if 'label' not in self.edges.columns:
            self.edges['label'] = DEFAULT_EDGE_LABEL
        if 'label_style' not in self.edges.columns:
            self.edges['label_style'] = DEFAULT_EDGE_LABEL_STYLE
        if 'lens' not in self.edges.columns:
            self.edges['lens'] = False
        if 'weight' not in self.edges.columns:
            self.edges['weight'] = 1
        self.Gnx = None
        self.scope_x = 0
        self.scope_y = 0
        self.communities = None
        self.appendix = ''

    def __convert_to_networkx(self):
        if self.Gnx is None:
            logging.info('Generating networkx graph ...')
            if self.directed:
                constructor = nx.DiGraph
            else:
                constructor = nx.Graph
            if 'weight' in self.edges.columns:
                self.Gnx = nx.from_pandas_edgelist(self.edges, source='source', 
                        target='target', 
                        edge_attr='weight',
                        create_using=constructor)
            else:
                self.Gnx = nx.from_pandas_edgelist(self.edges, source='source', 
                        target='target', create_using=constructor)
            # isolated nodes
            isolated_nodes = list(set(self.nodes.name.tolist()).difference(
                    set(self.Gnx.nodes)))
            if isolated_nodes:
                self.Gnx.add_nodes_from(isolated_nodes)
        return

    def layout(self, layout_type, **kwargs):
        logging.info(f'Layout type: {layout_type}')
        if not 'seed' in kwargs:    # for layouts that require random
            seed = None
        else:
            seed = kwargs['seed']
        if layout_type=='given':
            # expecting x,y in nodes
            if 'x' not in self.nodes.columns or 'y' not in self.nodes.columns:
                logging.error("Expected columns 'x' and 'y'.")
                raise
            # check for overlaps
            coords = pd.Series(list(zip(self.nodes['x'],self.nodes['y'])))
            lc = coords.shape[0]
            if lc > coords.drop_duplicates().shape[0]:
                logging.warning('Found overlapping nodes.')
        elif layout_type=='spring':
            self.__convert_to_networkx()
            pos = pd.DataFrame(nx.spring_layout(self.Gnx,
                seed=seed)).T
            pos = pos.rename(columns={0: 'x', 1: 'y'})
            try:
                logging.info('Removing any existing coordinates ...')
                self.nodes = self.nodes.drop(['x','y'], axis=1)
            except KeyError:
                logging.info('Found none.')
            self.nodes = self.nodes.merge(pos, left_on='name', 
                    right_index=True)
        else:
            logging.error('Unsupported layout type')
            raise

    def layout_scale_round(self, minx=None, maxx=None, 
            miny=None, maxy=None, round=2, padding=0):
        ominx = self.nodes.x.min()
        omaxx = self.nodes.x.max()
        ominy = self.nodes.y.min()
        omaxy = self.nodes.y.max()

        xratio = (maxx-minx-2*padding)/(omaxx-ominx)
        yratio = (maxy-miny-2*padding)/(omaxy-ominy)
        self.nodes.x = ((self.nodes.x - ominx) * xratio + minx + padding).round(round)
        self.nodes.y = ((self.nodes.y - ominy) * yratio + miny + padding).round(round)

    def compute_edge_angles(self):
        # this is for looseness (atan2)
        edges = self.edges[['source', 'target']]
        nodes = self.nodes[['name', 'x', 'y']]
        edges = edges.merge(nodes, left_on='source', right_on='name')
        edges = edges.rename(columns={'x': 'source_x', 'y': 'source_y'})
        edges = edges.reset_index().merge(nodes, left_on='target', right_on='name'
                ).set_index('index').sort_index()
        edges = edges.rename(columns={'x': 'target_x', 'y': 'target_y'})

        coords = edges[['source_x', 'source_y', 'target_x', 'target_y']
                ].to_numpy()
        x1 = coords[:,0]
        y1 = coords[:,1]
        x2 = coords[:,2]
        y2 = coords[:,3]

        self.edges['target_angle'] = np.round(np.arctan2((y1-y2),(x1-x2))*180/np.pi,2)
        self.edges['source_angle'] = np.round(np.arctan2((y2-y1),(x2-x1))*180/np.pi,2)
        
        return

    def displace_angles(self, displacement, mode='random'):
        # displacement for curved lines
        if mode == 'random':
            disp = np.random.choice([-1,1],self.edges.shape[0]) * displacement
        elif mode == 'fixed':
            disp = displacement
        else:
            raise ValueError(f'Invalid mode "{mode}".')
        self.edges['displacement'] = disp
        self.edges['out_angle'] = self.edges.source_angle - disp
        self.edges['in_angle'] = self.edges.target_angle + disp

    def append_node_attribute(self, prefix, values, nodes=None):
        if nodes:
            self.nodes.loc[self.nodes.name.isin(nodes), 'style'] = \
                    self.nodes[self.nodes.name.isin(nodes)]['style'] + ',' + \
                    prefix + values
        else:
            self.nodes['style'] = self.nodes['style'] + ',' + prefix + values
        return

    def reset_node_style(self):
        self.nodes.style = DEFAULT_NODE_STYLE

    def append_edge_attribute(self, prefix, values, mode='edge'):
        if mode == 'edge':
            # Note that "style" is overloaded as it is also used in pandas
            self.edges['style'] = self.edges['style'] + ',' + prefix + values
        elif mode == 'draw':
            self.edges.draw = self.edges.draw + ',' + prefix + values
        else:
            raise ValueError(f'Wrong mode {mode}.')
        return

    def reset_edge_style(self):
        self.edges.style = DEFAULT_EDGE_STYLE

    def append_edge_label_attribute(self, prefix, values):
        self.edges['label_style'] = self.edges['label_style'] + ',' + prefix + \
                values

    # Identifying communities
    def community(self, column=None, label=False, stretch=.4, 
            rounded_corners='1pt', smoothness=10):
        self.communities = self.nodes[~self.nodes[column].isnull()].groupby(
                column).apply(self._find_convex_hull, stretch=stretch,
                        smoothness=smoothness).reset_index().rename(
                        columns={column: 'name'})

        if label:
            self.communities['label'] = self.communities.name
        else:
            self.communities['label'] = ''
        self.communities['style'] = DEFAULT_COMMUNITY_STYLE + \
                ', rounded corners=' + rounded_corners

    # smoothness is an integer that decides number of points in the
    # circumference.
    def _find_convex_hull(self, com, stretch=.5, smoothness=None):
        points = zip(com.x.to_numpy(), com.y.to_numpy())
        augmented_points = []
        for p in points:
            augmented_points += [\
                    np.round((p[0] + np.cos(2*np.pi/smoothness*x)*stretch,
                        p[1] + np.sin(2*np.pi/smoothness*x)*stretch), 2) \
                                for x in range(0,smoothness+1)]
        points = np.array(augmented_points)
        con_points = points[ConvexHull(points).vertices]

        return pd.Series({'community': com['name'].tolist(), 'convex_hull': con_points})

    def append_community_attribute(self, prefix, values):
        try:
            if self.communities == None:
                raise ValueError('No communities detected.')
        except:
            pass

        self.communities['style'] = self.communities['style'] + \
                ',' + prefix + values

    def draw(self):
        if self.nodes.isnull().sum().sum():
            logging.warning('Found some Nans in nodes ...')
        if self.edges.isnull().sum().sum():
            logging.warning('Found some Nans in edges ...')
    
        # set global styles
        out = f'\\begin{{scope}}[xshift={self.scope_x}cm,yshift={self.scope_y}cm]\n'
    
        out += '% Nodes\n\\begin{pgfonlayer}{nl}\n'
    
        for id,node in self.nodes.iterrows():
            out += f"\\node[{node.style}] ({node['name']}) at ({node.x},{node.y}) {{{node.label}}};\n"
    
        out += '\\end{pgfonlayer}\n\n% Edges\n\\begin{pgfonlayer}{el}\n'
    
        directed = ''
        if self.directed:
            directed = ',->'
    
        for id,edge in self.edges.iterrows():
            try:
                if edge.directed == True:   # for mixed edges
                    directed = ',->'
                else:
                    directed = ''
            except:
                pass
    
            if edge.lens:
                out += generate_lens_edge(edge, directed)
            elif edge.label != DEFAULT_EDGE_LABEL:
                edge.label_style = 'inner sep=.5pt' + edge.label_style
                out += f'\\draw[{edge.draw}] ({edge.source}) edge[{edge.style}{directed}] node[{edge.label_style}] {{{edge.label}}} ({edge.target});\n'
            else:
                out += f'\\draw[{edge.draw}] ({edge.source}) edge[{edge.style}{directed}] ({edge.target});\n'
        out += '\\end{pgfonlayer}\n'

        # Communities
        out += '\n% Communities\n\\begin{pgfonlayer}{bg}'
        try:
            for id,com in self.communities.iterrows():
                com_str = '--'.join([f'({x[0]},{x[1]})' for x in com.convex_hull])
                com_str = f'\\draw[{com.style}] {com_str}--cycle; %{com["name"]}\n'
                ## if com.label != '':
                ##     com_str = com_str + f'label={com.label}'
                ## com_str = com_str + com.style + '] {};\n'
                out += com_str
                fp = com.convex_hull[0]
                out += f'\\node at ({fp[0]},{fp[1]}) {{{com.label}}};\n'
        except AttributeError:
            logging.warning('Did not find any communities, ignoring ...')

        out += '\\end{pgfonlayer}\n\n'

        # Appendix
        out += '\n% Appendix\n\\begin{pgfonlayer}{fg}'
        out += self.appendix
        out += '\\end{pgfonlayer}\n\n'

        out += '\n\\end{scope}'
                
        return out

def generate_lens_edge(edge, directed):
    outer_out_angle = np.round(edge.out_angle - 
            LENS_THICKNESS_FACTOR * (1-np.exp(-edge.weight)), 2)
    inner_out_angle = np.round(edge.out_angle + 
            LENS_THICKNESS_FACTOR * (1-np.exp(-edge.weight)), 2)
    outer_in_angle = np.round(edge.in_angle + 
            LENS_THICKNESS_FACTOR * (1-np.exp(-edge.weight)), 2)
    inner_in_angle = np.round(edge.in_angle - 
            LENS_THICKNESS_FACTOR * (1-np.exp(-edge.weight)), 2)
    # remove any out=,in=
    estyle = sub('out=[^,]*', '', edge.style)
    estyle = sub('in=[^,]*', '', estyle)
    estyle = sub(',,*', ',', estyle)

    # source and target locations
    sxd = np.round(LENS_NODE_DISP*np.cos(edge.source_angle),2)
    syd = np.round(LENS_NODE_DISP*np.sin(edge.source_angle),2)
    txd = np.round(LENS_NODE_DISP*np.cos(edge.target_angle),2)
    tyd = np.round(LENS_NODE_DISP*np.sin(edge.target_angle),2)
    source = f'($({edge.source}.center) + ({sxd},{syd})$)'
    target = f'($({edge.target}.center) + ({txd},{tyd})$)'

    # add edges
    estring = f'\\path {source} edge[fill={edge.color},{estyle},out={outer_out_angle},in={outer_in_angle}] {target};\n'
    estring += f'\\path {source} edge[fill={edge.color},{estyle},out={inner_out_angle},in={inner_in_angle}] {target};\n'
    return estring

# for given source node, find an angle that can be used to draw an edge away from existing edges
def gap_angle(node, edges):
    # get all edges connected to this node
    out_angles = edges[edges.source==node].source_angle.tolist()
    in_angles = edges[edges.target==node].target_angle.tolist()
    angles = out_angles + in_angles
    angles.sort()
    max_gap = 0
    gap_start = None
    for i in range(len(angles)):
        if i == len(angles) - 1:
            gap = (angles[0] + 360) - angles[i]
        else:
            gap = angles[i+1] - angles[i]
        if gap > max_gap:
            max_gap = gap
            gap_start = angles[i]
    best_angle = gap_start + max_gap/2
    return best_angle % 360


def define_html_colors(color_dict):
    colors_string = '\n% custom colors\n'
    for name, code in color_dict.items():
        colors_string += f'\\definecolor{{{name}}}{{HTML}}{{{code}}}\n'
    return colors_string

def to_tikz(G_list, colors=None, global_style='', appendix='', 
        mode='segment', outfile='out.tex'):

    # set global styles
    out = ''
    if global_style != '':
        out = '[' + global_style + ']\n\n'

    for G in G_list:
        out += G.draw()
        out += '\n\n'

    colors_string = ''
    if colors:
        colors_string = define_html_colors(colors)

    if mode == 'segment':
        return colors_string + out
    elif mode == 'standalone':
        return PREAMBLE + colors_string + BEGIN_STATEMENT + out + appendix + \
                END_STATEMENT
    elif mode == 'file':
        with open(outfile, 'w') as f:
            f.write(PREAMBLE + colors_string + BEGIN_STATEMENT + out + \
                    appendix + END_STATEMENT)
    else:
        logging.error(f'Wrong mode {mode}')
        raise

def compile(texfile):
    with open(devnull,'w') as fnull:
        check_call(
                ['pdflatex','-interaction','nonstopmode','-halt-on-error',texfile],
                stdout=fnull)

