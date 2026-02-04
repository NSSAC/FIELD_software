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
EDGE_THRESHOLD = 0.0001

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

    # for each state, create a separate observation node for each non-zero emission probability
    new_rows = []
    for state in ndf[ndf.type=='state'].name:
        emissions = edf[(edf.source==state) & (edf.target.str[0]!='S')]
        for _, row in emissions.iterrows():
            if row['label'] > EDGE_THRESHOLD:
                new_rows.append({'name': f"{state}_{row['target']}",
                                 'type': 'observation',
                                 'label': row['target'],
                                 'style': 'observation'})
    ndf = pd.concat([ndf, pd.DataFrame(new_rows)], ignore_index=True)
    # remove the old observation nodes
    ndf = ndf[~((ndf.type=='observation') & (~ndf.name.str.contains('_')))]

    # for each state, update the emission edges to point to the new observation nodes
    for state in ndf[ndf.type=='state'].name:
        emissions = edf[(edf.source==state) & (edf.target.str[0]!='S')]
        for _, row in emissions.iterrows():
            if row['label'] > EDGE_THRESHOLD:
                edf.loc[(edf.source==state) & (edf.target==row['target']), \
                         'target'] = f"{state}_{row['target']}"
    
    # remove unnecessary nodes and edges
    edf = edf[edf.label > EDGE_THRESHOLD]

    # make edge attributes
    edf['weight'] = .5
    edf.loc[edf.source.str.startswith('S') & edf.target.str.startswith('S'), 'weight'] = .1
    edf['label'] = edf.label.round(3).astype(str)

    # initiate graph
    G = gtt.GraphToDraw(ndf, edf, directed=True)

    # create a fixed layout
    ## arrange the states in a circle
    n_states = sum(G.nodes.type=='state')
    angle_step = 360 / n_states
    state_indices = G.nodes[G.nodes.type=='state'].index.tolist()
    state_indices.sort()
    radius = 1
    angle_map = {}
    for idx in state_indices:
        angle = angle_step * idx
        rad = angle * (3.14159 / 180)
        x = np.round(radius * np.cos(rad), 1)
        y = np.round(radius * np.sin(rad), 1)
        G.nodes.loc[idx, ['x', 'y']] = [x, y]

        angle_map[idx] = angle

        state_obs = G.nodes[G.nodes.name.str.startswith(f'S{idx}_')].name.tolist()
        state_obs.sort()
        n_obs = len(state_obs)
        # arrange observations in outer circle with respect to their states
        ## S0_0, S0_1 will have center S0 with radius 2
        if n_obs == 0:
            continue
        # at most 60 degrees for observations around each state
        angle_step_obs = 90 / len(state_indices)
        radius_obs = 3
        for i, obs in enumerate(state_obs):
            if n_obs % 2:
                angle_ = angle + (i - n_obs//2) * angle_step_obs
            else:
                angle_ = angle + (i - n_obs/2 + 0.5) * angle_step_obs
            rad = angle_ * (3.14159 / 180)
            x = np.round(radius_obs * np.cos(rad), 1)
            y = np.round(radius_obs * np.sin(rad), 1)
            G.nodes.loc[G.nodes.name==obs, ['x', 'y']] = [x, y]

    #G.append_edge_attribute('line width=', '1.5mm')

    G.append_edge_attribute('looseness=', '1')
    G.edges.loc[G.edges.source==G.edges.target, 'style'] = G.edges[G.edges.source.str.startswith('S') & G.edges.target.str.startswith('S')]['style'] + ',looseness=8'

    # first get edge aesthetics for states
    G.compute_edge_angles()

    for state in angle_map.keys():
        if state:
            prev_state = state - 1
        else:
            prev_state = len(angle_map) - 1
        best_angle = gtt.gap_angle(f'S{state}', G.edges)
        G.edges.loc[(G.edges.source==f'S{state}') & (G.edges.target==f'S{state}'), 'source_angle'] = best_angle
        G.edges.loc[(G.edges.source==f'S{state}') & (G.edges.target==f'S{state}'), 'target_angle'] = best_angle
    G.edges['displacement'] = 20
    G.edges.loc[G.edges.target=='0', 'displacement'] = -20
    G.displace_angles(G.edges.displacement, mode='fixed')

    G.append_edge_attribute('out=', G.edges.out_angle.astype('str'))
    G.append_edge_attribute('in=', G.edges.in_angle.astype('str'))
    G.append_edge_label_attribute('fill=', 'white')

    # create simple edges between states and observations
    G.edges.loc[G.edges.target.str.contains('_'), 'style'] = ''

    # generic style for edges
    G.append_edge_attribute('', '>=latex,font=\\tiny')

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