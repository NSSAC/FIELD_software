'''
Plot HMM models

AA
'''

import kbviz.graph_to_tikz as gtt
import numpy as np
import pandas as pd
import pickle
from pdb import set_trace
import sys

HMM = '../intermediate_data/baseline_conditional_risk.pkl'

def draw_hmm(livestock, models):
    mod = models[0][livestock]

    edges = []
    # extract transition matrix edges from the model
    tm = mod.transmat_
    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            edges.append({'source': f'S{i}', 'target': f'S{j}', 'label': tm[i,j]})

    # extract emission matrix edges from the model
    em = mod.emissionprob_
    for i in range(em.shape[0]):
        for j in range(em.shape[1]):
            edges.append({'source': f'S{i}', 'target': f'{j}', 'label': em[i,j]})

    edf = pd.DataFrame(edges)

    # Global styles
    global_style = '''
    every node/.style={rectangle, inner sep=2pt},
    arr/.style={>=latex, shorten >=1pt, shorten <=1pt},
    state/.style={circle,minimum width=3.5mm,fill=blue!50,draw},
    observation/.style={rectangle,minimum width=3.5mm,fill=red!50,draw},
    '''

    # make node attributes
    ndf = pd.DataFrame({'name': pd.unique(edf[['source', 'target']].values.ravel())})
    ndf['type'] = 'observation'
    ndf.loc[ndf.name.str.startswith('S'), 'type'] = 'state'
    ndf['label'] = ndf.name
    ndf['style'] = 'observation'
    ndf.loc[ndf.type=='state', 'style'] = 'state'

    # make edge attributes
    edf['weight'] = .5
    edf.loc[edf.source.str.startswith('S') & edf.target.str.startswith('S'), 'weight'] = .1
    edf['label'] = edf.label.round(3).astype(str)
    edf['style'] = 'font=\\tiny'

    # initiate graph
    G = gtt.GraphToDraw(ndf, edf, directed=True)

    # create a fixed layout
    ## arrange the states in a circle
    n_states = sum(G.nodes.type=='state')
    angle_step = 360 / n_states
    state_indices = G.nodes[G.nodes.type=='state'].index.tolist()
    radius = 1
    for idx, node_idx in enumerate(state_indices):
        angle = angle_step * idx
        rad = angle * (3.14159 / 180)
        x = np.round(radius * np.cos(rad) + radius, 2)
        y = np.round(radius * np.sin(rad) + radius, 2)
        G.nodes.loc[node_idx, ['x', 'y']] = [x, y]
    ## arrange the observations in a line
    obs_nodes = G.nodes[G.nodes.label.str[0]=='observation'] G.nodes.loc[G.nodes.label=='0',['x', 'y']] = [3, 2]
    G.nodes.loc[G.nodes.label=='1',['x', 'y']] = [3, 0]

    ## G.layout('spring', seed=5)
    ## G.layout_scale_round(minx=0, maxx=5, miny=0, maxy=5, round=2)

    #G.append_edge_attribute('line width=', '1.5mm')
    G.append_edge_attribute('', '>=latex')

    G.append_edge_attribute('looseness=', '1')
    G.edges.loc[G.edges.source==G.edges.target, 'style'] = G.edges[G.edges.source.str.startswith('S') & G.edges.target.str.startswith('S')]['style'] + ',looseness=8'

    G.compute_edge_angles()
    G.edges['displacement'] = 20
    G.edges.loc[G.edges.target=='0', 'displacement'] = -20
    G.displace_angles(G.edges.displacement, mode='fixed')
    G.append_edge_attribute('out=', G.edges.out_angle.astype('str'))
    G.append_edge_attribute('in=', G.edges.in_angle.astype('str'))
    #G.edges.loc[(G.edges.source=='S1') & (G.edges.target=='S1'), 'style'] = G.edges[(G.edges.source=='S1') & (G.edges.target=='S1')]['style'] + ',out=-160,in=160'

    G.append_edge_label_attribute('fill=', 'white')

    # Convert to tikz
    gtt.to_tikz([G], global_style=global_style, 
            mode='file', outfile=f'{livestock}_hmm.tex')
    gtt.compile(f'{livestock}_hmm.tex')

if __name__ == '__main__':
    # Load HMM
    with open(HMM, 'rb') as f:
        models = pickle.load(f)
    
    draw_hmm('milk', models)
    draw_hmm('ckn-layers', models)
    draw_hmm('turkeys', models)
    draw_hmm('backyard', models)