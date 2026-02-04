DESC = '''
Compare farm assignments with CAFOmaps data.

By: AA
'''

import argparse
from aadata import loader
import geopandas as gpd
import numpy as np
import pandas as pd
from pdb import set_trace

#from aautils import stats, timer

PARSL = False

if PARSL:
    import parsl
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.addresses import address_by_hostname
    from parsl.app.app import python_app, bash_app

    config = Config(
       executors=[
           HighThroughputExecutor(
               label="local_htex",
               cores_per_worker=1,
               max_workers_per_node=4,
               address='127.0.0.1',
           )
       ],
       strategy='none',
       run_dir='ipf'
    )
    parsl.load(config)

def filter_assg(assg, cattle=0, poultry=0, hogs=0):
    return assg[((assg.livestock=='cattle') & (assg.heads>=cattle)) |
                ((assg.livestock=='poultry') & (assg.heads>=poultry)) |
                ((assg.livestock=='hogs') & (assg.heads>=hogs))]

def _cafo_match(cafo, assg):

    from aautils import geometry
    import pandas as pd
    import networkx as nx

    MILES_THRESHOLD = 50

    assg = assg.copy()
    cafo = cafo.copy()
    
    adf = assg[['lat', 'lon', 'fid']].copy()
    cdf = cafo[['lat', 'long']]

    livestock = assg.head(1).livestock.values[0]

    edges = cdf.reset_index().merge(adf.reset_index(), how='cross')
    print(livestock, edges.shape[0])
    dist = geometry.haversine(
            lon1=edges['long'].values,
            lat1=edges['lat_x'].values,
            lon2=edges['lon'].values,
            lat2=edges['lat_y'].values,
            units='miles')
    edges['distance'] = dist
    edges = edges.rename(columns={'index_x': 'cafo', 'index_y': 'assg'})[
            ['cafo', 'fid', 'distance']]

    edges = edges[edges.distance<=MILES_THRESHOLD].copy()

    edges['weight'] = 1/edges.distance
    edges['cafo_'] = 'c' + edges.cafo.astype(str)
    edges['farm_'] = 'f' + edges.fid.astype(str)

    #edges = edges.head(10000)

    # Maximum weighted matching
    G = nx.from_pandas_edgelist(edges, 'cafo_', 'farm_', edge_attr='weight')
    matching = nx.algorithms.matching.max_weight_matching(G, 
                                                          maxcardinality=True)

    if len(matching):
        matched_edges = pd.DataFrame.from_records(
                [(u, v) if u[0]=='c' else (v, u) for u, v in matching])
    else:
        matched_edges = pd.DataFrame(columns=[0, 1])
    
    try:
        for col in matched_edges.columns:
            matched_edges[col] = matched_edges[col].str.slice(
                    start=1).astype(int)
        matched_edges = matched_edges.set_index(0, drop=True)
    except:
        return -1, cafo

    cafo = cafo.join(matched_edges, how='left').rename(
            columns={1: 'agcensus'}).reset_index().rename(columns=
                                                          {'index': 'cafo'})
    cafo = cafo.merge(
            edges[['cafo', 'fid', 'distance']], 
            how='left',
            left_on=['cafo', 'agcensus'], right_on=['cafo', 'fid'])
    cafo = cafo.fillna(-1)
    set_trace()
    
    return cafo

if PARSL:
    cafo_match = python_app(_cafo_match)
else:
    cafo_match = _cafo_match
    
if __name__ == '__main__':
    # parser
    parser=argparse.ArgumentParser(description=DESC, 
    formatter_class=argparse.RawTextHelpFormatter)
    args = parser.parse_args()

    # load data
    print('Loading data ...')
    ## states = loader.load('usa_states')
    ## statefp = states[states.state==args.state].statefp.values[0]

    cafo = pd.read_csv('../../data/cafo/cafomaps.csv.zip')
    cafo.loc[cafo.livestock=='chickens', 'livestock'] = 'poultry'

    assg = pd.read_csv('../../livestock/results/farms_to_cells.csv.zip')
    assg = assg[assg.subtype=='all'].copy()

    glw_cells = gpd.read_file('../../data/glw/glw_cells.shp.zip')
    assg = assg.merge(glw_cells[['x', 'y', 'lat', 'lon']], on=['x', 'y'])

    # Filter assg
    assg = filter_assg(assg, cattle=300, poultry=9000, hogs=750)

    res = []
    for livestock in ['cattle', 'hogs', 'poultry']:
        print(livestock)
        res.append(cafo_match(cafo[(cafo.livestock==livestock)],
                   assg[(assg.livestock==livestock)]))
        ## if i>10:
        ##     break
    tot = 3 #i

    if PARSL:
        rlist = []
        i = 0
        for rr in res:
            i += 1
            if not i % 20:
                print(i, '/', tot)
            tt = rr.result()
            try: 
                if tt[0] == -1:
                    set_trace()
            except:
                pass
            rlist.append(tt)

        res = rlist

    pd.concat(res).to_csv('cafo_match.csv.zip', index=False)

    print('Done')
