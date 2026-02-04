DESC = '''
Helper function for repeated tasks accross scripts.

By: AA
'''

import ast
import click
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pdb import set_trace
from dateutil.relativedelta import relativedelta
from sys import argv

from kbdata import loader

FARM_SIZE = [1, 99, 999, 99999, 1000000000] 
FARM_SIZE_NAMES = ['s', 'm', 'l', 'vl']
CENTRAL_VALLEY = 6999

POULTRY_COMMERCIAL_THRESHOLD = 1000
DAIRY_COMMERCIAL_THRESHOLD = 100

def load_shapes():
    regions = loader.load('usa_county_shapes', contiguous_us=True)
    states = regions[['state_code', 'geometry']].dissolve(by='state_code')
    return regions, states

@click.group()
def cli():
    pass

def load_poultry_farms(aggregate=None, commercial=False, commercial_threshold=POULTRY_COMMERCIAL_THRESHOLD):
    farms = pd.read_csv('../../livestock/results/farms_to_cells.csv.zip')
    farms['county_code'] = farms.state_code*1000 + farms.county_code
    pf = farms[(farms.livestock=='poultry') & 
               (farms.subtype.isin(['ckn-broilers', 'ckn-pullets', 
                                    'ckn-layers', 'turkeys', 'ducks',
                                    'geese', 'poultry-other']))].copy()
    pf.subtype = pf.subtype.map({'geese': 'waterfowl',
                                 'poultry-other': 'waterfowl',
                                 'ckn-broilers': 'ckn-broilers',
                                 'ckn-layers': 'ckn-layers',
                                 'ckn-pullets': 'ckn-pullets',
                                 'turkeys': 'turkeys',
                                 'ducks': 'ducks'})
    pf = pf.drop('livestock', axis=1)

    if commercial:
        pf = pf[pf.heads>commercial_threshold].copy()
    
    if aggregate == 'county':
        pf = pf.groupby(['state_code', 'county_code', 'subtype'], 
                         as_index=False)[['fid', 'heads']].agg({
                             'fid': 'count', 'heads': 'sum'})
        pf = pf.rename(columns={'fid': 'farms'})
    return pf

def load_dairy_farms(aggregate=None, commercial=False, commercial_threshold=DAIRY_COMMERCIAL_THRESHOLD):
    farms = pd.read_csv('../../livestock/results/farms_to_cells.csv.zip')
    farms['county_code'] = farms.state_code*1000 + farms.county_code
    pf = farms[farms.subtype=='milk'].copy()

    if commercial:
        pf = pf[pf.heads>commercial_threshold].copy()
    
    if aggregate == 'county':
        pf = pf.groupby(['state_code', 'county_code', 'subtype'], 
                         as_index=False)[['fid', 'heads']].agg({
                             'fid': 'count', 'heads': 'sum'})
        pf = pf.rename(columns={'fid': 'farms'})
    return pf

@cli.command()
@click.option('--agg_by', default='month')
def bird_features(agg_by):
    nmdf = pd.read_parquet('../intermediate_data/glw_moore_10.parquet') # obtained from neighborhood_graph.py

    # load
    print('Load and aggregate bird data ...')
    df = pd.read_csv('../../../../data/ldt/avian_layer.csv.zip')
    df['date'] = pd.to_datetime(df.week_end)
    if agg_by == 'quarter':
        tdf = df.groupby(['x', 'y', pd.Grouper(key='date', freq='Q')], 
                         as_index=False)['abundance'].sum()
        tdf.date = (tdf.date.dt.month/3).astype('int')
        times = range(1,5)
    elif agg_by == 'month':
        tdf = df.groupby(['x', 'y', pd.Grouper(key='date', freq='M')], 
                         as_index=False)['abundance'].sum()
        tdf.date = tdf.date.dt.month.astype('int')
        times = range(1,13)
    tdf.date = 'birds' + tdf.date.astype('str')
    bdf = tdf.pivot(columns='date', index=['x', 'y'], values='abundance'
                    ).reset_index().fillna(0)

    print('Weighted sum')
    for q in times:
        print(q)
        tbdf = bdf[['x', 'y', f'birds{q}']].copy().rename(columns={
            f'birds{q}': 'val'})
        ndf = neighborhood_weighted(tbdf, nmdf)
        bdf.loc[:, f'birds{q}_W1'] = ndf['nval1']
        bdf.loc[:, f'birds{q}_W2'] = ndf['nval2']

    # save as parquet
    bdf.to_parquet('bird_features.parquet')
    return

def neighborhood_weighted(df, nmdf):

    # join relevant columns from df
    tdf = df.merge(nmdf, on=['x', 'y'], how='left')
    tdf = tdf.merge(df, left_on=['x_','y_'], right_on=['x','y'], how='left').drop(
            ['x_y', 'y_y'], axis=1).rename(columns={'x_x': 'x', 'y_x': 'y', 
                                                    'val_x': 'val'}).fillna(0)
    tdf['nval1'] = tdf.val_y/tdf.dist**1
    tdf['nval2'] = tdf.val_y/tdf.dist**2

    # aggregate
    odf = tdf.groupby(['x', 'y'])[['val', 'nval1', 'nval2']].agg(
            {'val': 'first', 'nval1': 'sum', 'nval2': 'sum'}).reset_index()

    return odf

def fit_central_valley(df, county_code_col='county_code'): # the central valley CA problem for dairy
    df = df.copy()
    counties = loader.load('usa_county_shapes', contiguous_us=True)
    central_valley_counties = ["butte", "colusa", "fresno", "glenn", "kern",
                               "kings", "madera", "merced", "plumas", 
                               "sacramento", "san joaquin", "shasta",
                               "stanislaus", "sutter", "tehama", "tulare", 
                               "yolo", "yuba"]
    cvdf = counties[(counties.state=='california') & 
                   (counties.county.isin(central_valley_counties))]
    geom = cvdf.geometry.union_all()
    fips = cvdf.county_code.tolist()

    df.loc[(df[county_code_col].isin(fips)), county_code_col] = CENTRAL_VALLEY

    if 'geometry' in df.columns:
        df.loc[df[county_code_col]==CENTRAL_VALLEY, 'geometry'] = geom

    return df

@cli.command()
def livestock_features():
    # load features
    features = pd.read_parquet('bird_features.parquet')
    birds_cols = [f'birds{x}' for x in range(1,13)]
    birds_W_cols = [f'birds{x}_W2' for x in range(1,13)]
    features = features[['x', 'y'] + birds_cols + birds_W_cols]

    for q in range(1,13):
        features[f'birds{q}_W2'] = features[f'birds{q}_W2'] + features[f'birds{q}']
    features = features.drop(birds_cols, axis=1)

    poultry = load_poultry_farms(commercial=True, 
                                 commercial_threshold=POULTRY_COMMERCIAL_THRESHOLD)
    dairy = load_dairy_farms(commercial=True, 
                             commercial_threshold=DAIRY_COMMERCIAL_THRESHOLD)
    dairy = dairy.drop('livestock', axis=1)
    ldf = pd.concat([poultry, dairy])

    df = ldf.merge(features, on=['x', 'y'], suffixes=('_f', ''), how='left')
    df.to_parquet('livestock_birds_features.parquet')

def h5n1_birds(agg_by=None):
    birds = pd.read_csv('../../data/h5n1/birds.csv')
    if agg_by == 'quarter':
        df = birds[['state_code', 'county_code', 'year', 'quarter']].value_counts()
        df = df.reset_index(name='incidences')
        #df = df.rename(columns={'count': 'incidences'})
        df = df[df.incidences!=0]
    elif agg_by == 'month':
        df = birds[['state_code', 'county_code', 'year', 'month']].value_counts()
        df = df.reset_index(name='incidences')
        #df = df.rename(columns={'count': 'incidences'})
        df = df[df.incidences!=0]
    else:
        df = birds
    return df

def h5n1_bird_types(agg_by_quarter=False):
    birds = pd.read_csv('../../data/h5n1/birds.csv')
    if agg_by_quarter:
        df = birds[['state_code', 'county_code', 'year',
                    'quarter', 'bird species']].value_counts()
        df = df.reset_index(name='incidences')
        #df = df.rename(columns={'count': 'incidences'})
        df = df[df.incidences!=0]
    elif agg_by == 'month':
        df = birds[['state_code', 'county_code', 'year', 'month', 'bird species']].value_counts()
        df = df.reset_index(name='incidences')
        #df = df.rename(columns={'count': 'incidences'})
        df = df[df.incidences!=0]
    else:
        df = birds
    return df

def h5n1_poultry(agg_by=None, commercial=False):
    pdf = pd.read_csv('../../data/h5n1/poultry.csv')
    pdf.start_date = pd.to_datetime(pdf.start_date)
    pdf['year'] = pdf.start_date.dt.year
    pdf['month'] = pdf.start_date.dt.month

    if commercial:
        pdf = pdf[pdf.commercial]

    if agg_by == 'quarter':
        ppdf = pdf.groupby(['state_code', 'county_code', 'year', 'quarter', 
                            'type'], as_index=False)['start_date'].count()
        ppdf = ppdf.rename(columns={'start_date': 'reports'})
    elif agg_by == 'month':
        ppdf = pdf.groupby(['state_code', 'county_code', 'year', 'month', 
                            'type'], as_index=False)['start_date'].count()
        ppdf = ppdf.rename(columns={'start_date': 'reports'})
    elif agg_by == 'county':
        ppdf = pdf.groupby(['state_code', 'county_code', 'type'], 
                           as_index=False)['start_date'].count()
        ppdf = ppdf.rename(columns={'start_date': 'reports'})
    else:
        ppdf = pdf
    return ppdf

def h5n1_poultry_with_county_neighbors(commercial=False):
    cdf = pd.read_csv('../../data/h5n1/poultry.csv')
    cdf.start_date = pd.to_datetime(cdf.start_date)
    cdf['year'] = cdf.start_date.dt.year
    cdf['month'] = cdf.start_date.dt.month

    if commercial:
        cdf = cdf[cdf.commercial]
    cdf = cdf[['county_code', 'type', 'year', 'month']].drop_duplicates()

    cn = pd.read_parquet('county_neighbors.parquet')
    cn['county_code'] = cn.statefp_x.astype('int')*1000 + cn.countyfp_x.astype('int')
    cn['county_code_n'] = cn.statefp_y.astype('int')*1000 + cn.countyfp_y.astype('int')
    cn = cn[['county_code', 'county_code_n']]
    cn = cn[cn.county_code!=cn.county_code_n].drop_duplicates()

    odf = cdf.groupby('type', as_index=False).apply(
        __h5n1_poultry_subtype_with_county_neighbors, cn=cn)

    odf = odf.groupby(['county_code', 'type', 'year', 'month'],
                       as_index=False)['report'].sum()

    return odf

def __h5n1_poultry_subtype_with_county_neighbors(df, cn=None):
    ndf = df.merge(cn, on='county_code', how='left')
    # One entry if at least one neighboring county reports
    ndf = ndf.drop('county_code', axis=1).rename(columns={'county_code_n': 'county_code'}).drop_duplicates()

    # 1 means reported in the county, 2 means reported in neighbor county
    df['report'] = 1
    ndf['report'] = 2
    odf = pd.concat([df, ndf])
    return odf

def h5n1_dairy(agg_by=None):
    df = pd.read_csv('../../data/h5n1/dairy.csv')
    df.start_date = pd.to_datetime(df.start_date)
    df['year'] = df.start_date.dt.year
    df['month'] = df.start_date.dt.month

    if agg_by == 'quarter':
        ddf = df.groupby(['state_code', 'county_code', 'year', 'quarter'], 
                         as_index=False)['start_date'].count()
        ddf = ddf.rename(columns={'start_date': 'reports'})
    elif agg_by == 'month':
        ddf = df.groupby(['state_code', 'county_code', 'year', 'month', 
                            'type'], as_index=False)['start_date'].count()
        ddf = ddf.rename(columns={'start_date': 'reports'})
    elif agg_by == 'county':
        ddf = df.groupby('county_code', as_index=False)['start_date'].count()
        ddf = ddf.rename(columns={'start_date': 'reports'})
    else:
        ddf = df
    return ddf

def h5n1_dairy_with_county_neighbors():
    df = pd.read_csv('../../data/h5n1/dairy.csv')
    df.start_date = pd.to_datetime(df.start_date)
    df['year'] = df.start_date.dt.year
    df['month'] = df.start_date.dt.month
    cdf = df[['county_code', 'year', 'month']].drop_duplicates()
    cdf = fit_central_valley(cdf).drop_duplicates()

    cn = pd.read_parquet('county_neighbors.parquet')
    cn['county_code'] = cn.statefp_x.astype('int')*1000 + cn.countyfp_x.astype('int')
    cn['county_code_n'] = cn.statefp_y.astype('int')*1000 + cn.countyfp_y.astype('int')
    cn = cn[['county_code', 'county_code_n']]
    cn = cn[cn.county_code!=cn.county_code_n].drop_duplicates()
    cn = fit_central_valley(cn[['county_code', 'county_code_n']], county_code_col='county_code')
    cn = fit_central_valley(cn, county_code_col='county_code_n')

    ndf = cdf.merge(cn[['county_code', 'county_code_n']], on='county_code', how='left')
    # One entry if at least one neighboring county reports
    ndf = ndf.drop('county_code', axis=1).rename(columns={'county_code_n': 'county_code'}).drop_duplicates()

    # 1 means reported in the county, 2 means reported in neighbor county
    cdf['report'] = 1
    ndf['report'] = 2

    odf = pd.concat([cdf, ndf])
    odf = odf.groupby(['county_code', 'year', 'month'], as_index=False)['report'].sum()

    return odf

## @cli.command()
## @click.option('--mode', default='month')
def wastewater(mode='month'):
    # load and process ww
    wwo = pd.read_csv('../../data/NWSS_wastewater/WWS_H5_detections.csv.gz')
    if mode == 'month':
        wwo['time'] = wwo.rec_date.str[:7]
    elif mode == 'day':
        wwo['time'] = wwo.rec_date

    ### positive
    tdf = wwo[(wwo.rec_y!=0) & (~wwo.rec_y.isnull())].copy()
    if mode == 'month':
        wwp = tdf[['time', 'counties_served']].drop_duplicates()
    elif mode == 'day':
        wwp = tdf[['time', 'counties_served']].copy()
    wwp['present'] = 1

    ### negative
    tdf = wwo[(wwo.rec_y==0)].copy()
    if mode == 'month':
        wwn = tdf[['time', 'counties_served']].drop_duplicates()
    elif mode == 'day':
        wwn = tdf[['time', 'counties_served']].copy()
    wwn['present'] = 0

    ww = pd.concat([wwp, wwn])

    ww.counties_served = ww.counties_served.apply(ast.literal_eval)
    ww = ww.explode('counties_served').rename(columns=
                                              {'counties_served': 'county_code'})
    ww = ww[~ww.county_code.isnull()]
    ww.county_code = ww.county_code.astype(int)
    ww = ww.drop_duplicates()

    return ww

@cli.command()
def county_neighbors():

    counties = loader.load('usa_county_shapes')
    counties = counties.astype({'statefp': 'int', 'statefp': 'int'})

    # first find intersection between states
    sg = counties.groupby('statefp').geometry.apply(lambda x: x.unary_union)
    states = gpd.GeoDataFrame(sg, geometry='geometry').reset_index()
    states.statefp = states.statefp.astype(int)
    sgdf = gpd.GeoDataFrame(states.merge(states, how='cross'), 
                            geometry='geometry_x')
    sgdf.geometry_x = sgdf.geometry_x.buffer(.05)
    sgdf['adj'] = sgdf.geometry_x.intersects(sgdf.geometry_y)
    sgdf = sgdf[sgdf.adj]

    # now find intersection between adjacent states
    cadj = sgdf[['statefp_x', 'statefp_y']].merge(
            counties[['statefp', 'countyfp', 'geometry']],
            left_on='statefp_x', right_on='statefp')
    cadj = cadj.drop('statefp', axis=1)
    cadj = cadj.merge(
            counties[['statefp', 'countyfp', 'geometry']],
            left_on='statefp_y', right_on='statefp')
    cgdf = gpd.GeoDataFrame(cadj, geometry='geometry_x')
    cgdf.geometry_x = cgdf.geometry_x.buffer(.005)
    cgdf['adj'] = cgdf.geometry_x.intersects(cgdf.geometry_y)
    cgdf = cgdf[cgdf.adj]
    cgdf = cgdf.drop(['geometry_x', 'geometry_y', 'statefp', 'adj'], axis=1)

    cgdf.to_parquet('county_neighbors.parquet')

    # graph
    cgdf['source'] = cgdf.statefp_x.astype('str').str.zfill(2) + \
            cgdf.countyfp_x.astype('str').str.zfill(3)
    cgdf['target'] = cgdf.statefp_y.astype('str').str.zfill(2) + \
            cgdf.countyfp_y.astype('str').str.zfill(3)
    G = nx.from_pandas_edgelist(cgdf)

    # shortest path
    print('shortest path ...')
    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G)) 
    # Convert to DataFrame
    rows = []
    for source, targets in shortest_path_lengths.items():
        for target, length in targets.items():
            rows.append({"source": source, "target": target, "length": length})
    
    df = pd.DataFrame(rows)
    df['state_code_x'] = df.source.str[0:2].astype(int)
    df['state_code_y'] = df.target.str[0:2].astype(int)
    df['county_code_x'] = df.source.str[2:].astype(int)
    df['county_code_y'] = df.target.str[2:].astype(int)
    df.to_parquet('county_hops.parquet')

    return

@cli.command()
def county_distances():
    cn = pd.read_parquet('../intermediate_data/county_hops.parquet')
    cn_shapes = loader.load('usa_county_shapes', contiguous_us=True)
    cn_shapes = cn_shapes.to_crs(epsg=32633)  # Example: UTM zone 33N

    cn.source = cn.source.astype('int')
    cn.target = cn.target.astype('int')
    
    # merging is tricky due to gpd
    src = cn_shapes[['county_code', 'geometry']].merge(cn[['source']],
                         left_on='county_code',
                         right_on='source', how='right')
    src['centroid'] = src.geometry.centroid
    tgt = cn_shapes[['county_code', 'geometry']].merge(cn[['target']],
                         left_on='county_code',
                         right_on='target', how='right')
    tgt['centroid'] = tgt.geometry.centroid

    dist = src.centroid.distance(tgt.centroid) / 1609.344
    cn['dist'] = dist

    cn[~cn.dist.isnull()].to_parquet('county_distances.parquet')

@cli.command()
@click.option('--agg_by', default='month', help="aggregate by quarter/month")
def birds_h5n1_ind_buffer(agg_by):
    bdf = h5n1_birds()
    cn = pd.read_parquet('county_neighbors.parquet')
    bdf = bdf.merge(cn, left_on=['state_code', 'county_code'],
                    right_on=['statefp_x', 'countyfp_x'], how='left')
    ind = (bdf.state_code==bdf.statefp_y) & (bdf.county_code==bdf.countyfp_y)
    bdf['dist'] = 1
    bdf.loc[ind, 'dist'] = 0
    bdf = bdf.drop(['state_code', 'county_code', 'statefp_x', 'countyfp_x'],
                   axis=1)
    bdf = bdf.rename(columns={'statefp_y': 'state_code',
                              'countyfp_y': 'county_code'})
    if agg_by == 'quarter':
        print(f'Aggregating by {agg_by}')
        bdf = bdf.groupby(['year', 'quarter', 'state_code', 'county_code'])[
                'dist'].min().reset_index()
    elif agg_by == 'month':
        print(f'Aggregating by {agg_by}')
        bdf = bdf.groupby(['year', 'month', 'state_code', 'county_code'])[
                'dist'].min().reset_index()
    bdf.to_parquet('birds_h5n1_ind_buffer.parquet')
    return

def bin_farms(df):
    nbins = 4
    tot = df.farms.sum()
    name = df.name
    df = df.sort_values('farms')
    df['cs'] = df.farms.cumsum()
    df['county'] = df.state_code*1000 + df.county_code
    for i in range(nbins):
        df.loc[df.cs>tot/nbins*i, 'bin'] = i + 1

    odf = df.groupby('bin', as_index=False).agg({'farms': 'sum', 
                                                'reports': 'sum',
                                                'county': 'count'})
    odf_ = df[df.reports!=0].groupby('bin', as_index=False)['county'].count()
    odf = odf.merge(odf_, on='bin', suffixes=('', '_x'), how='left').fillna(0)
    odf = odf.rename(columns={'county_x': 'rep_counties'})
    odf['subtype'] = name 
    return odf

@cli.command()
@click.option('--commercial_threshold', default=100, help="commercial threshold")
@click.option('--livestock', default='poultry', help="poultry/milk")
def farm_neighborhood(commercial_threshold, livestock):
    nmdf = pd.read_parquet('../intermediate_data/glw_moore_10.parquet') # obtained from neighborhood_graph.py
    cells = nmdf[['x', 'y']].drop_duplicates()
    cells[['x_', 'y_']] = cells[['x', 'y']]
    cells['dist'] = 1
    nmdf = pd.concat([nmdf, cells])
    nmdf = nmdf[nmdf.dist<=20]

    if livestock == 'poultry':
        farms = load_poultry_farms(commercial=True, commercial_threshold=100)
    elif livestock == 'milk':
        farms = load_dairy_farms(commercial=True, commercial_threshold=100)
    else:
        raise ValueError('Unsupported livestock:', livestock)
    farms = farms[(farms.subtype!='all')]

    # merge with farms by neighborhood
    fdf = farms.merge(nmdf, on=['x', 'y']) 
    fdf = fdf.merge(farms, left_on=['x_', 'y_'], right_on=['x', 'y'],
                        suffixes=['', '_n'])
    fdf = fdf[fdf.fid!=fdf.fid_n]

    fdf.to_parquet(f'farm_neighborhood_{livestock}.parquet')

@cli.command()
def outbreak_discovery():
    print('Poultry')
    print('--------------------------------------------------')
    pdf = h5n1_poultry(commercial=False)
    pdf.start_date = pd.to_datetime(pdf.start_date)
    poultry = pdf.copy()
    
    cn = pd.read_parquet('county_neighbors.parquet')
    cn['county_x'] = cn.statefp_x*1000 + cn.countyfp_x
    cn['county_y'] = cn.statefp_y*1000 + cn.countyfp_y

    dfl = []
    for delta in [15, 30, 60]:
        df = _outbreak_discovery_subroutine(pdf, delta, cn)
        df.type = 'all_poultry'
        sdf = pdf.groupby('type').apply(_outbreak_discovery_subroutine, 
                                        delta, cn, include_groups=False).reset_index(drop=False).drop('level_1', axis=1)
        sdf.type = sdf.type
        tdf = pd.concat([df, sdf])
        tdf['delta'] = delta
        dfl.append(tdf)
    odf = pd.concat(dfl)
    odf = odf.drop(['timediff'], axis=1)
    odf.to_csv('poultry_outbreaks.csv', index=False)

    print('Dairy')
    print('--------------------------------------------------')
    pdf = h5n1_dairy()
    pdf.start_date = pd.to_datetime(pdf.start_date)
    pdf = fit_central_valley(pdf)
    cn_dairy = fit_central_valley(cn, county_code_col='county_x')
    cn_dairy = fit_central_valley(cn_dairy, county_code_col='county_y')
    cn_dairy = cn_dairy[['county_x', 'county_y']].drop_duplicates()
    dairy = pdf.copy()
    
    dfl = []
    for delta in [15, 30, 60]:
        tdf = _outbreak_discovery_subroutine(pdf, delta, cn_dairy)
        tdf['delta'] = delta
        dfl.append(tdf)
    odf = pd.concat(dfl)
    odf = odf.drop(['timediff'], axis=1)
    odf.to_csv('dairy_outbreaks.csv', index=False)

    print('All')
    print('--------------------------------------------------')
    poultry = fit_central_valley(poultry)
    dairy['type'] = 'milk'
    df = pd.concat([dairy, poultry])
    df['type_'] = df.type
    df.type = 'all'

    dfl = []
    for delta in [15, 30, 60]:
        tdf = _outbreak_discovery_subroutine(df, delta, cn_dairy)
        tdf['delta'] = delta
        dfl.append(tdf)
    odf = pd.concat(dfl)
    odf = odf.drop(['timediff'], axis=1)
    odf.type = odf.type_
    odf = odf.drop('type_', axis=1)
    odf.to_csv('all_host_outbreaks.csv', index=False)

def _outbreak_discovery_subroutine(df, delta, cn):
    df = df.sort_values('start_date').reset_index(drop=True)
    df['event0'] = -1
    df['event1'] = -1
    event0 = 0
    event1 = 0
    for i,row in df.iterrows():
        df['timediff'] = (row.start_date - df.start_date).dt.days
        try:
            neighbors = cn[cn.county_x==row.county_code].county_y.tolist()
        except:
            set_trace()

        tdf = df.head(i)
        tdf = tdf[(tdf.timediff<=delta) & (tdf.county_code.isin(neighbors))]

        if not tdf.shape[0]:
            df.loc[i, 'event0'] = event0
            df.loc[i, 'event1'] = event1
            event0 += 1
            event1 += 1
        elif row.county_code in tdf.county_code.tolist():
            df.loc[i, ['event0', 'event1']] = tdf[
                    tdf.county_code==row.county_code].head(1)[
                            ['event0', 'event1']].values[0]
        else:
            df.loc[i, 'event1'] = tdf.head(1).event1.values[0]
            df.loc[i, 'event0'] = event0
            event0 += 1
    return df

def percentile_categorize(df, timecol=None, riskcol=None,
                          start_time=None, end_time=None,
                          percentiles=[0, 50, 75, 90, 95, 100],
                          exclude_zeros=False,
                          labels=['vl', 'l', 'm', 'h', 'vh']):
    ranking = df[(df[timecol]>=start_time) & (df[timecol]<=end_time)].copy()

    if exclude_zeros:
        zdf = ranking[ranking[riskcol]==0].copy()
        ranking = ranking[ranking[riskcol]>0].copy()
        labels_ = labels[1:]
    else:
        labels_ = labels

    tdf = ranking.groupby(timecol, group_keys=True)[riskcol].apply(
            _ranking_per_period, percentiles=percentiles, labels=labels_
            ).reset_index().set_index('level_1').rename(
                    columns={riskcol: 'rank_per_period'})
    ranking = ranking.join(tdf['rank_per_period'])

    ranking['rank_across_periods'] = _ranking_per_period(ranking[riskcol], 
                                                         percentiles=percentiles, 
                                                         labels=labels_)

    if exclude_zeros:
        zdf['rank_per_period'] = labels[0]
        zdf['rank_across_periods'] = labels[0]
        ranking = pd.concat([ranking, zdf])
    return ranking

def _ranking_per_period(ds, percentiles=None, labels=None):
    bins = np.unique(np.percentile(ds, percentiles))
    plabels = labels[-len(bins)+1:]
    ranking = pd.cut(ds, bins=bins, labels=plabels,
                     include_lowest=True)
    return ranking

def _type_ranking_period(df, percentiles=None, percentiles_labels=None):
    ranking = df.copy()
    for per in df.columns:
        if per[0:2] != '20':
            continue
        tds = df[per]
        bins = np.unique(np.percentile(tds, percentiles))
        labels = percentiles_labels[-len(bins)+1:]
        ranking[per] = pd.cut(tds, bins=bins, labels=labels,
                              include_lowest=True)
    return ranking

def _type_ranking_total(df, percentiles=None, percentiles_labels=None):
    ranking = df.melt(id_vars=['county_code', 'subtype'], var_name='time',
                      value_name='risk')
    bins = np.unique(np.percentile(ranking.risk, percentiles))
    labels = percentiles_labels[-len(bins)+1:]
    ranking['risk_profile'] = pd.cut(ranking.risk, bins=bins, labels=labels,
                                     include_lowest=True)
    return ranking


    # output
    outprefix = f'poultry_risk_bp_{int(bird_h5_prevalence)}_sp_{int(spillover_risk_model)}_cr_{int(conditional_risk_model)}_ad_{int(adaptive)}'
    farms_risk.to_csv(f'{outprefix}_farms.csv')
    cdf.to_csv(f'{outprefix}_counties.csv')
    events[['confirmed', 'state_code', 'county_code', 'type', 'risk']].to_csv(
            f'{outprefix}_evaluation.csv')
    eval = events.risk.value_counts()
    print(eval)

def combine_probs(df, id_vars=None):
    df = df.set_index(id_vars)
    invdf = (1 - df).reset_index()
    return (1 - invdf.groupby(id_vars).prod()).reset_index()

@cli.command()
def ww_coevents():
    # load ww
    ww = wastewater(mode='day')
    ww = ww[ww.present==1].copy()
    ww.time = pd.to_datetime(ww.time) 

    dfl = []
    for delta in [30, 60]:
        tdf = _covents_discovery(ww, delta)
        tdf['delta'] = delta
        dfl.append(tdf)
    odf = pd.concat(dfl)
    odf = odf.drop(['timediff'], axis=1)
    odf.to_csv('ww_colocation_events.csv', index=False)

def _covents_discovery(df, delta):
    df = df.sort_values('time').reset_index(drop=True)
    df['event'] = -1
    event = 0
    for i,row in df.iterrows():
        df['timediff'] = (row.time - df.time).dt.days
        tdf = df.head(i)
        tdf = tdf[(tdf.timediff<=delta) & (tdf.county_code==row.county_code)]

        if not tdf.shape[0]:
            df.loc[i, 'event'] = event
            event += 1
        else:
            df.loc[i, 'event'] = tdf[
                    tdf.county_code==row.county_code].head(1)['event'].values[0]
    return df

# Function to remove outliers based on IQR per group
def remove_outliers(df, column=None, q=0.25):
    q1 = df[column].quantile(q)
    q3 = df[column].quantile(1-q)
    IQR = q3 - q1
    lower_bound = q1 - 1.5 * IQR
    upper_bound = q3 + 1.5 * IQR
    return df[(df[column]>=lower_bound) & (df[column]<=upper_bound)]

def timediff(left_date, right_date, mode='month'):
    df = pd.DataFrame(columns=['left', 'right'])
    df.left = left_date
    df.right = right_date
    if mode == 'month':
        return df.apply(lambda row: 
                        relativedelta(row.left, row.right).years * 12 +
                        relativedelta(row.left, row.right).months, axis=1)
    else:
        raise ValueError('Only mode="month" supported')

if __name__ == '__main__':
    cli()
