DESC = '''
This has become a scratch area for any exploration.

By: AA
'''

from itertools import product
import geopandas as gpd
from matplotlib.patches import Patch
import numpy as np
import os
import pandas as pd
from pdb import set_trace
from scipy.stats import pearsonr
import stat

from aadata import loader
from aaviz import plot
from parlist import POULTRY_PARLIST as PARLIST

FARM_SIZE = [1, 99, 999, 99999, 1000000000] 
FARM_SIZE_NAMES = ['s', 'm', 'l', 'vl']

def pcc(df):
    df = df[~df.incidences.isnull()]
    try:
        return pearsonr(df.incidences, df.abundance).statistic
    except:
        return -2


def dairy():
    df = pd.read_csv('../../data/h5n1/dairy.csv').groupby(
            ['fips', 'quarter'], as_index=False)['Confirmed'].count()
    df.to_parquet('h5n1.parquet')
    set_trace()
    birds = pd.read_csv('../../data/h5n1/birds.csv')
    df = birds[['statefp', 'countyfp', 'year', 'quarter']].value_counts()
    df = df.reset_index()
    df = df.rename(columns={'count': 'incidences'})
    df = df[df.incidences!=0]
    return df


def birds_dairy():
    bdf = birds()
    pdf = dairy()

    pbdf = bdf.merge(pdf, on=['statefp', 'countyfp', 'year', 'quarter'], 
                     how='outer', indicator=True)
    pbdf._merge = pbdf._merge.map({'left_only': 'birds', 
                                   'right_only': 'poultry',
                                   'both': 'both'})
    pbdf._merge = pbdf._merge.astype(str)

    fig, gs = plot.initiate_figure(x=5*4, y=4*3, 
                                   gs_nrows=3, gs_ncols=4,
                                   gs_wspace=.1, gs_hspace=.1,
                                   color='tableau10')

    colors = plot.COLORS['tableau10']
    colormap = {
            "birds": colors[0],
            "poultry": colors[1],
            "both": colors[2],
            }
    legend_elements = [Patch(facecolor=colormap[key], label=key) 
                       for key in colormap]
    regions = loader.load('usa_county_shapes')
    regions = regions.astype({'statefp': 'int', 'countyfp': 'int'})
    regions = regions[~regions.statefp.isin([2, 15, 72, 66, 69, 78, 60])]
    states = regions[['statefp', 'geometry']].dissolve(by='statefp')
    rpbdf = regions.merge(pbdf, on=['statefp', 'countyfp']).fillna(0)

    i = 0
    for year in [2022, 2023, 2024]:
        j = 0
        for quarter in [1,2,3,4]:
            tdf = rpbdf[(rpbdf.year==year) & (rpbdf.quarter==quarter)]
            bi = tdf.incidences.sum().astype(int)
            pi = tdf.farms.sum().astype(int)
            ax = plot.subplot(fig=fig, grid=gs[i,j], 
                              func='gpd.boundary.plot',
                              pf_facecolor='white', pf_edgecolor='grey',
                              pf_linewidth=.1, data=states) 
            ax = plot.subplot(fig=fig, ax=ax, grid=gs[i,j], func='gpd.plot',
                              data=tdf, 
                              pf_color=tdf._merge.map(colormap),
                              pf_markersize=2,
                              la_ylabel=f'{year}', fs_ylabel='Large',
                              la_title=f'b: {bi}, p: {pi}',
                              la_xlabel=f'{quarter}', fs_xlabel='Large')
            j += 1
        i += 1
    ax.legend(handles=legend_elements, 
              loc="lower left", bbox_to_anchor=(1.1,1), 
              fontsize=15, title_fontsize=12)
    plot.savefig('poultry_birds_incidences.pdf')

def bird_abundance():

    features = pd.read_parquet('../intermediate_data/risk_features.parquet')
    bdf = features[['state_code', 'county_code', 'birds1', 'birds2', 'birds3', 
                    'birds4']].groupby(['state_code', 'county_code']).sum()
    bdf.columns = [1,2,3,4]
    odf = bdf.melt(var_name='quarter', value_name='abundance',
                   ignore_index=False)
    odf.abundance = np.log10(odf.abundance)

    regions = loader.load('usa_county_shapes')
    regions = regions.astype({'statefp': 'int', 'countyfp': 'int'})
    regions = regions[~regions.statefp.isin([2, 15, 72, 66, 69, 78, 60])]
    states = regions[['statefp', 'geometry']].dissolve(by='statefp')
    rodf = regions.merge(odf, 
                         left_on=['statefp', 'countyfp'],
                         right_on=['state_code', 'county_code']).fillna(0)

    fig, gs = plot.initiate_figure(x=5*4, y=4, 
                                   gs_nrows=1, gs_ncols=4,
                                   gs_wspace=.1, gs_hspace=.1,
                                   color='tableau10')
    i = 0
    for quarter in [1,2,3,4]:
        tdf = rodf[rodf.quarter==quarter]
        ax = plot.subplot(fig=fig, grid=gs[0,i], 
                          func='gpd.boundary.plot',
                          pf_facecolor='white', pf_edgecolor='grey',
                          pf_linewidth=.1, data=states) 
        ax = plot.subplot(fig=fig, ax=ax, func='gpd.plot',
                          data=tdf, pf_column='abundance',
                          pf_legend=True, pf_legend_kwds={'shrink': 0.28},
                          la_ylabel='', fs_ylabel='Large',
                          la_title=f'',
                          la_xlabel=f'{quarter}', fs_xlabel='Large')
        i += 1
    plot.savefig('bird_abundance.pdf')

def abundance_incidences():

    features = pd.read_parquet('../intermediate_data/risk_features.parquet')
    bdf = birds()
    pdf = poultry()

    farms = pd.read_csv('../../livestock/results/farms_to_cells.csv.zip')
    pf = farms[(farms.livestock=='poultry') & (farms.subtype=='all')]
    pfc = pf.groupby(['state_code', 'county_code'])['fid'].count().reset_index()

    # birds
    tdf = features[['state_code', 'county_code', 'birds1', 'birds2', 'birds3',
                    'birds4']].groupby(['state_code', 'county_code']).sum()
    tdf.columns = [1,2,3,4]
    adf = tdf.melt(var_name='quarter', value_name='abundance',
                   ignore_index=False).reset_index()
    
    df = bdf.merge(adf, left_on=['statefp', 'countyfp', 'quarter'],
                   right_on=['state_code', 'county_code', 'quarter'],
                   how='right')
    df['report'] = 'no incidence'
    df.loc[~df.incidences.isnull(), 'report'] = 'incidence'

    odf = df.groupby(['year', 'quarter']).apply(pcc)
    sodf = df.groupby(['year', 'statefp', 'quarter']).apply(pcc).reset_index()
    ttdf = adf.groupby(['state_code', 'quarter'])['abundance'].sum()

    sodf = sodf.merge(ttdf, left_on=['statefp', 'quarter'],
                      right_on=['state_code', 'quarter'])
    sodf = sodf.rename(columns={0: 'pcc'})
    sodf = sodf[~((sodf.pcc==-2) | (sodf.pcc.isnull()))]

    fig, gs = plot.initiate_figure(x=5*3, y=4, 
                                   gs_nrows=1, gs_ncols=3,
                                   gs_wspace=.3, gs_hspace=.1,
                                   color='tableau10')
    ax = plot.subplot(fig=fig, grid=gs[0,1], func='sns.scatterplot', 
                      data=df[~df.incidences.isnull()],
                      sp_xscale='log',
                      pf_x='abundance', pf_y='incidences', pf_hue='quarter',
                      la_title='\\parbox{10cm}{\\center (county, quarter) instances with non-zero incidence}'
                      )
    ax = plot.subplot(fig=fig, grid=gs[0,0], func='sns.kdeplot', data=df,
                      pf_x='abundance', pf_hue='report',
                      la_title='Distribution of abundance'
                      )
    ax = plot.subplot(fig=fig, grid=gs[0,2], func='sns.scatterplot', 
                      data=sodf,
                      pf_x='abundance', pf_y='pcc',
                      sp_ylim=[-1,1], sp_xlim=[0,'default'],
                      la_title='\\parbox{10cm}{\\center Correlation bet. abundance and incidences for (state, quarter) instances}',
                      fs_title='normalsize'
                      )
    plot.savefig('birds_abundance_incidences.pdf')

    # poultry
    tdf = features.groupby(['state_code', 'county_code']
                           )['poultry'].sum().reset_index()
    df = pdf.merge(tdf, left_on=['statefp', 'countyfp'],
                   right_on=['state_code', 'county_code'], how='right')
    df = df.rename(columns={'farms': 'incidences'})
    
    df['report'] = 'no incidence'
    df.loc[~df.incidences.isnull(), 'report'] = 'incidence'

    odf = df.groupby(['year', 'quarter']).apply(pcc)
    sodf = df.groupby(['year', 'statefp', 'quarter']).apply(pcc).reset_index()
    ttdf = adf.groupby(['state_code', 'quarter'])['abundance'].sum()

    sodf = sodf.merge(ttdf, left_on=['statefp', 'quarter'],
                      right_on=['state_code', 'quarter'])
    sodf = sodf.rename(columns={0: 'pcc'})
    sodf = sodf[~((sodf.pcc==-2) | (sodf.pcc.isnull()))]
    df = df.merge(pfc, on=['state_code', 'county_code'])

    fig, gs = plot.initiate_figure(x=5*3, y=3, 
                                   gs_nrows=1, gs_ncols=3,
                                   gs_wspace=.3, gs_hspace=.1,
                                   color='tableau10')
    ax = plot.subplot(fig=fig, grid=gs[0,0], func='sns.kdeplot', data=df,
                      pf_x='poultry', pf_hue='report',
                      la_title='Distribution of abundance'
                      )
    ax = plot.subplot(fig=fig, grid=gs[0,1], func='sns.scatterplot', 
                      data=df[~df.incidences.isnull()],
                      sp_xscale='log',
                      pf_x='poultry', pf_y='incidences', pf_hue='quarter',
                      la_title='\\parbox{8cm}{\\center (county, quarter) with non-zero incidence}'
                      )
    ax = plot.subplot(fig=fig, grid=gs[0,2], func='sns.scatterplot', 
                      data=df[~df.incidences.isnull()],
                      sp_xscale='log',
                      pf_x='fid', pf_y='incidences', pf_hue='quarter',
                      la_title='\\parbox{8cm}{\\center (county, quarter) with non-zero incidence}',
                      la_xlabel=r'\#farms'
                      )
    ## ax = plot.subplot(fig=fig, grid=gs[0,2], func='sns.scatterplot', 
    ##                   data=sodf,
    ##                   pf_x='abundance', pf_y='pcc',
    ##                   sp_ylim=[-1,1], sp_xlim=[0,'default'],
    ##                   la_title='\\parbox{10cm}{\\center Correlation bet. abundance and incidences for (state, quarter) instances}',
    ##                   fs_title='normalsize'
    ##                   )
    plot.savefig('poultry_abundance_incidences.pdf')
    return



    set_trace()

    pbdf = bdf.merge(pdf, on=['statefp', 'countyfp', 'year', 'quarter'], 
                     how='outer', indicator=True)
    pbdf._merge = pbdf._merge.map({'left_only': 'birds', 
                                   'right_only': 'poultry',
                                   'both': 'both'})
    pbdf._merge = pbdf._merge.astype(str)
    set_trace()

    fig, gs = plot.initiate_figure(x=5*4, y=4*3, 
                                   gs_nrows=3, gs_ncols=4,
                                   gs_wspace=.1, gs_hspace=.1,
                                   color='tableau10')

    colors = plot.COLORS['tableau10']
    colormap = {
            "birds": colors[0],
            "poultry": colors[1],
            "both": colors[2],
            }
    legend_elements = [Patch(facecolor=colormap[key], label=key) 
                       for key in colormap]
    regions = loader.load('usa_county_shapes')
    regions = regions.astype({'statefp': 'int', 'countyfp': 'int'})
    regions = regions[~regions.statefp.isin([2, 15, 72, 66, 69, 78, 60])]
    states = regions[['statefp', 'geometry']].dissolve(by='statefp')
    rpbdf = regions.merge(pbdf, on=['statefp', 'countyfp']).fillna(0)

    i = 0
    for year in [2022, 2023, 2024]:
        j = 0
        for quarter in [1,2,3,4]:
            tdf = rpbdf[(rpbdf.year==year) & (rpbdf.quarter==quarter)]
            bi = tdf.incidences.sum().astype(int)
            pi = tdf.farms.sum().astype(int)
            ax = plot.subplot(fig=fig, grid=gs[i,j], 
                              func='gpd.boundary.plot',
                              pf_facecolor='white', pf_edgecolor='grey',
                              pf_linewidth=.1, data=states) 
            ax = plot.subplot(fig=fig, ax=ax, grid=gs[i,j], func='gpd.plot',
                              data=tdf, 
                              pf_color=tdf._merge.map(colormap),
                              pf_markersize=2,
                              la_ylabel=f'{year}', fs_ylabel='Large',
                              la_title=f'b: {bi}, p: {pi}',
                              la_xlabel=f'{quarter}', fs_xlabel='Large')
            j += 1
        i += 1
    ax.legend(handles=legend_elements, 
              loc="lower left", bbox_to_anchor=(1.1,1), 
              fontsize=15, title_fontsize=12)
    plot.savefig('poultry_birds_incidences.pdf')

def distance_weighted_farm_counts():

    # processing neighborhood cells
    neighbors = pd.read_parquet('../intermediate_data/glw_moore_10.parquet')
    tdf = neighbors[['x', 'y']].drop_duplicates()
    tdf['x_'] = tdf.x
    tdf['y_'] = tdf.y
    tdf['dist'] = 1
    neighbors = pd.concat([neighbors, tdf])

    # farms
    farms = pd.read_csv('../../livestock/results/farms_to_cells.csv.zip')
    pf = farms[(farms.livestock=='poultry') & (farms.subtype=='all')]
    pfc = pf.groupby(['x', 'y'])['fid'].count().reset_index()
    county_map = pf[['x', 'y', 'state_code', 'county_code']].drop_duplicates()

    # merge with neigbhorhood cells
    df = neighbors.merge(pfc, left_on=['x_', 'y_'],
                         right_on=['x', 'y'], how='left', 
                         suffixes=['', '_f']).drop(['x_f', 'y_f'], axis=1
                                                   ).fillna(0)

    df = df[df.dist<=93]    # 93 makes it a circle
    dfl = []
    ddf = df[['x', 'y']].drop_duplicates()
    for pow in [0, 1, 2]:
        col = f'd{pow}'
        df[col]  = df.fid / df.dist**pow
        ddf = ddf.merge(df.groupby(['x', 'y'], as_index=False
                                   )[col].sum(), on=['x', 'y'])
    ddf = ddf.merge(pfc, on=['x', 'y'])
    ddf = ddf.merge(county_map, on=['x', 'y'])
    ddf.to_parquet('distance_weighted_farm_counts.parquet')

def county_buffered_counts():

    # processing neighborhood cells
    neighbors = pd.read_parquet('../intermediate_data/glw_moore_10.parquet')
    tdf = neighbors[['x', 'y']].drop_duplicates()
    tdf['x_'] = tdf.x
    tdf['y_'] = tdf.y
    tdf['dist'] = 1
    neighbors = pd.concat([neighbors, tdf])
    neighbors = neighbors[neighbors.dist<=50]

    # farms
    farms = pd.read_csv('../../livestock/results/farms_to_cells.csv.zip')
    pf = farms[(farms.livestock=='poultry') & (farms.subtype=='all')]
    pfc = pf.groupby(['x', 'y'])['fid'].count().reset_index()
    pfc = pfc.rename(columns={'fid': 'farms'})
    county_map = pf[['x', 'y', 'state_code', 'county_code']].drop_duplicates()

    # merge with neigbhorhood cells
    df = neighbors.merge(pfc, left_on=['x_', 'y_'],
                         right_on=['x', 'y'], how='left', 
                         suffixes=['', '_f']).drop(['x_f', 'y_f'], axis=1
                                                   ).fillna(0)
    df = df.merge(county_map, on=['x', 'y'])
    ddf = df.groupby(['state_code', 'county_code'], as_index=False)[
            ['x_', 'y_', 'farms']].apply(
                    lambda _df: _df.drop_duplicates().sum())
    ddf = ddf.drop(['x_', 'y_'], axis=1)
    ddf.to_parquet('county_buffered_counts.parquet')

def neighborhood():
    # processing neighborhood cells
    neighbors = pd.read_parquet('../intermediate_data/glw_moore_10.parquet')
    tdf = neighbors[['x', 'y']].drop_duplicates()
    tdf['x_'] = tdf.x
    tdf['y_'] = tdf.y
    tdf['dist'] = 1
    neighbors = pd.concat([neighbors, tdf])
    neighbors.to_parquet('neighbors.parquet')
    return

    set_trace()
    neighbors_orig = neighbors_orig.merge(yqs, how='cross')
    pd.to_parquet('neighbors_yqs.parquet')


def county_buffered_choropleths():
    df = pd.read_parquet('birds_h5n1_ind_buffer.parquet').reset_index()
    regions = loader.load('usa_county_shapes', contiguous_us=True)

    df = df.merge(regions, left_on=['state_code', 'county_code'],
                  right_on=['statefp', 'countyfp'])
    df = gpd.GeoDataFrame(df, geometry='geometry')

    df = df[(df.year==2022) & (df.quarter==2)]
    states = regions[['statefp', 'geometry']].dissolve(by='statefp')

    fig, gs = plot.initiate_figure(x=5, y=4, 
                                   gs_nrows=1, gs_ncols=1,
                                   gs_wspace=.1, gs_hspace=.1,
                                   color='tableau10')
    colors = plot.COLORS['tableau10']
    colormap = {
            0: colors[0],
            1: colors[1],
            }
    ax = plot.subplot(fig=fig, grid=gs[0,0], 
                      func='gpd.boundary.plot',
                      pf_facecolor='white', pf_edgecolor='grey',
                      pf_linewidth=.1, data=states) 
    plot.subplot(fig=fig, ax=ax, grid=gs[0,0], func='gpd.plot',
                 data=df,
                 pf_color=df.dist.map(colormap),
                 pf_markersize=2,
                 la_ylabel='', fs_ylabel='Large',
                 la_title=f'',
                 la_xlabel='', fs_xlabel='Large')
    plot.savefig('county_buffered_choropleths.pdf')

def county_buffered_counts_with_birds_ind():
    farms = pd.read_csv('../../livestock/results/farms_to_cells.csv.zip')
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
    pf['type'] = pf.subtype
    pf = pf.drop(['livestock', 'subtype'], axis=1)
    county_map = pf[['x', 'y', 'state_code', 'county_code']].drop_duplicates()
    pf = pf[pf.heads>100]
    pfc = pf.groupby(['x', 'y', 'type'])['fid'].count().reset_index()
    pfc = pfc.rename(columns={'fid': 'farms'})
    # pf = pd.read_parquet('../intermediate_data/risk_features.parquet')

    species = ['waterfowl', 'ckn-broilers', 'ckn-layers', 'ckn-pullets',
               'turkeys', 'ducks']
    ## tp = list(product([2022, 2023, 2024], [1,2,3,4], species))
    ## yqs = pd.DataFrame.from_records(tp, columns=['year', 'quarter', 'type'])
    tp = list(product([2022, 2023, 2024], [1,2,3,4]))
    yq = pd.DataFrame.from_records(tp, columns=['year', 'quarter'])

    print('loading neighbors ...')
    neighbors_orig = pd.read_parquet('neighbors.parquet')

    birds_inc = pd.read_parquet('birds_h5n1_ind_buffer.parquet')

    # farms
    dfl = []
    for dist in [1, 10, 20, 50]:
        print(dist)
        neighbors = neighbors_orig[neighbors_orig.dist<=dist]
        ### merge with neigbhorhood cells
        fdf = neighbors.copy().merge(pfc, left_on=['x_', 'y_'],
                             right_on=['x', 'y'], how='left', 
                             suffixes=['', '_f']).drop(['x_f', 'y_f'], axis=1
                                                       ).fillna(0)
        fdf = fdf.merge(county_map, on=['x', 'y'], how='left')
        fdf = fdf.merge(county_map, left_on=['x_', 'y_'], 
                      right_on=['x', 'y'], suffixes=('', '_n'), how='left')
        fdf = fdf.drop(['x_n', 'y_n'], axis=1)

        # birds indicator
        fdf = fdf.merge(birds_inc, left_on=['state_code_n', 'county_code_n'],
                        right_on=['state_code', 'county_code'],
                        suffixes=('', '_y'))
        fdf = fdf.rename(columns={'dist_x': 'dist', 'dist_y': 'cneighbor'})
        fdf = fdf.drop(['state_code_y', 'county_code_y'], axis=1)
        fdf = fdf[fdf.type!=0]

        # county-level aggregation for county-buffered incidences
        odf = fdf.groupby(['state_code', 'county_code', 'type']
                )[['x_', 'y_', 'year', 'quarter', 'farms', 'state_code_n', 
                   'county_code_n', 'cneighbor']].apply(
                        poultry_county_agg_with_bird_incidence)
        odf['buffer'] = dist
        dfl.append(odf)

    ddf = pd.concat(dfl)
    ddf = ddf.reset_index().drop('level_3', axis=1)

    ddf.to_parquet('county_buffered_farms_incidences.parquet')

def poultry_county_agg_with_bird_incidence(df):
    name = df.name

    ## if name[0]==41 and name[1]==5 and name[2]==2022 and name[3]==1:
    ##     set_trace()
    tdf = df.drop_duplicates().groupby(
            ['year', 'quarter'])[['farms', 'cneighbor']].agg(
                    {'farms': 'sum', 'cneighbor': 'min'}).reset_index()
    tdf['cneighbor'] = df.cneighbor.min()
    return tdf

# Among counties in the bird incidence zone, what are the ranks
def county_ranks_per_quarter():
    df = pd.read_parquet('county_buffered_farms_incidences.parquet'
                         ).reset_index()
    df = df.drop('index', axis=1)
    pdf = utils.h5n1_poultry(commercial=True, agg_by_quarter=True)

    df = df.merge(ppdf, on=['state_code', 'county_code', 'year', 
                            'quarter', 'type'], how='outer')
    df.reports = df.reports.fillna(0)
    
    print(f'{df[df.farms.isnull()].reports.sum()}/{df.reports.sum()} correspond to no-bird-incidence region')
    
    buffer = 10
    year = 2024
    quarter = 2

    tdf = df[(df.year==year) & (df.quarter==quarter) & (df.buffer==buffer)].copy()
    # print((tdf.inf_farms>0).sum(), (tdf[col]>0).sum())

    type = 'turkeys'
    ttdf = tdf[tdf.type==type].copy()

    ttdf['r'] = ttdf.farms.rank(ascending=False)
    ttdf = ttdf.sort_values('r')
    xx = ttdf[ttdf.reports>0]
    set_trace()
    print(tdf[tdf.inf_farms>0][f'r{dist}'].sort_values().tolist())

    set_trace()

if __name__ == '__main__':
    # birds_poultry()
    # abundance_incidences()
    # bird_abundance()
    # county_buffered_counts()
    # poultry_incidence_county_buffered()
    # poultry_ranked_by_bird_incidence()
    # neighborhood()

    # county_neighbors()
    # birds_h5n1_ind_buffer()
    # county_buffered_choropleths()
    # county_buffered_counts_with_birds_ind()
    county_ranks_per_quarter()

