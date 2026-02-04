DESC = '''
Prepare all layers. Final output should contain x,y,var1,var2,...

AA
'''

import pandas as pd
from pdb import set_trace

from aautils.display import pf
from aautils.geometry import glw_to_lonlat
import h5n1_data_analysis
import utils

LIVESTOCK = '../../livestock/results/farms_to_cells.csv.zip'
BIRDS = '../../../../data/ldt/avian_layer.csv.zip'
POP = '../../population/results/population.csv.zip'

NEIGHBOR = '../intermediate_data/glw_moore_10.parquet'

IGNORED_NAICS = [115, 4245, 114]

ndf = pd.read_parquet(NEIGHBOR) # obtained from neighborhood_graph.py

# For mobility related considerations
# assumes that x,y,val are given
# Weighted aggregation of adjacent grid cells to provide the W variable.

@pf
def livestock():
    # load
    fdf = pd.read_csv(LIVESTOCK)
    df = fdf.groupby(['x', 'y', 'livestock', 'subtype'], as_index=False
                    )['heads'].sum()
    
    # aggregate where necessary
    ### cattle
    print('cattle')
    tdf = df[df.livestock=='cattle'].drop('livestock', axis=1)
    cattle = tdf.pivot(index=['x', 'y'], columns='subtype',
                       values='heads').rename(
                               columns={'all': 'cattle'}).reset_index()
    cattle = cattle.fillna(0)
    for st in ['cattle', 'beef', 'milk', 'other']:
        tdf = cattle[['x', 'y', st]].rename(columns={st: 'val'}).copy()
        ndf = neighborhood(tdf)
        cattle.loc[:, st+'_W1'] = ndf['nval1']
        cattle.loc[:, st+'_W2'] = ndf['nval2']

    ### poultry
    print('poultry')
    tdf = df[df.livestock=='poultry'].drop('subtype', axis=1)
    poultry = tdf.groupby(['x', 'y', 'livestock'], as_index=False).sum()
    tdf = poultry[['x', 'y', 'heads']].rename(columns={'heads': 'val'}).copy()
    ndf = neighborhood(tdf)
    poultry.loc[:, 'poultry_W1'] = ndf['nval1']
    poultry.loc[:, 'poultry_W2'] = ndf['nval2']
    poultry = poultry.drop('livestock', axis=1).rename(
            columns={'heads': 'poultry'})
    
    ### hogs
    print('hogs')
    hogs = df[df.livestock=='hogs'].drop('subtype', axis=1)
    tdf = hogs[['x', 'y', 'heads']].rename(columns={'heads': 'val'}).copy()
    ndf = neighborhood(tdf)
    hogs.loc[:, 'hogs_W1'] = ndf['nval1']
    hogs.loc[:, 'hogs_W2'] = ndf['nval2']
    hogs = hogs.drop('livestock', axis=1).rename(
            columns={'heads': 'hogs'})

    ### merge
    odf = cattle.merge(poultry, on=['x', 'y'], how='outer')
    odf = odf.merge(hogs, on=['x', 'y'], how='outer')

    ### fips
    cells = fdf[['x', 'y', 'state_code', 'county_code']].drop_duplicates()
    odf = odf.merge(cells, on=['x', 'y'], how='left')
    if odf.state_code.isnull().any():
        raise ValueError('State code is null for some cells.')

    print('Number of cells:', odf.shape)
    print('Number of nulls:', odf.isnull().sum())
    odf = odf.fillna(0)
    
    # save as parquet
    odf.to_parquet('livestock.parquet')
    return

@pf
def pop():
    # load
    df = pd.read_csv(POP)
    df.loc[df.ag_naics.isin(IGNORED_NAICS), 'ag_naics'] = -1
    df.loc[df.ag_naics==-1, 'ag'] = 'pop_non_ag'
    df.loc[df.ag_naics!=-1, 'ag'] = 'pop_ag_' + df[
            df.ag_naics!=-1].ag_naics.astype('str').values
    pdf = df[['x', 'y', 'ag', 'count']
             ].groupby(['x', 'y', 'ag'], as_index=False
                       ).sum().pivot(index=['x', 'y'], columns='ag', 
                                     values='count').reset_index().fillna(0)
    
    for col in pdf.columns:
        if col in ['x', 'y']:
            continue
        tdf = pdf[['x', 'y', col]].rename(columns={col: 'val'}).copy()
        ndf = neighborhood(tdf)
        pdf.loc[:, col+'_W1'] = ndf['nval1']
        pdf.loc[:, col+'_W2'] = ndf['nval2']

    if pdf.isnull().sum(axis=1).max() == 5:
        raise ValueError('Not all columns can be nulls.')
    
    pdf.fillna(0).to_parquet('pop.parquet')

@pf
def merge():
    ldf = pd.read_parquet('../intermediate_data/livestock.parquet')
    bdf = pd.read_parquet('../intermediate_data/birds.parquet')
    pdf = pd.read_parquet('../intermediate_data/pop.parquet')
    df = ldf.merge(bdf, on=['x', 'y'], how='left')
    df = df.merge(pdf, on=['x', 'y'], how='left')
    df.loc[:, ['lon', 'lat']] = glw_to_lonlat(df.x, df.y)
    df = df.fillna(0)

    df.fillna(0).to_parquet('risk_features.parquet')

## @pf
## def birds_h5n1_indicator():
##     bdf = h5n1_data_analysis.birds()
##     glw = pd.read_parquet('../../../../data/ldt/avian_layer.parquet')[
##             ['x', 'y', 'state_code', 'county_code']].drop_duplicates()
##     odf = bdf.merge(glw, right_on=['state_code', 'county_code'],
##                     left_on=['statefp', 'countyfp'], how='left')
##     odf.to_parquet('birds_h5n1_indicator.parquet')
##     return
##     tdf = ndf[['x', 'y', 'x_', 'y_']].merge(glw, left_on=['x_', 'y_'], 
##                                             right_on=['x', 'y'], how='left')
##     tdf = tdf[~tdf.state_code.isnull()]
##     tdf = tdf.drop(['x_y', 'y_y'], axis=1)
##     tdf = tdf.rename(columns={'x_x': 'x', 'y_x': 'y'})
##     tdf = tdf.merge(bdf, left_on=['state_code', 'county_code'],
##                     right_on=['statefp', 'countyfp'], how='left')
##     tdf = tdf[~tdf.incidences.isnull()]
##     odf = tdf.groupby(['x', 'y'], as_index=False)[
##             ['year', 'quarter', 'incidences']].max()
## 
##     odf.to_parquet('birds_h5n1_indicator_w_neighborhood.parquet')
##     return

if __name__ == '__main__':

    # birds_h5n1_indicator()
    set_trace()
    ## merge()
    ## df = pd.read_parquet('risk_features.parquet')
    ## livestock()
    # ldf = pd.read_parquet('../intermediate_data/livestock.parquet')
    ## birds()
    ## pop()
    ## merge()




