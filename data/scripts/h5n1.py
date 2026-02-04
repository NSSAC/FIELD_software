DESC = '''
Processing h5n1 files.

AA
'''

import click
import pandas as pd
from pdb import set_trace

from kbdata import loader
from kbutils.display import pf

FARM_SIZE = [-1, 1, 99, 999, 99999, 1000000000] 
FARM_SIZE_NAMES = ['', 's', 'm', 'l', 'vl']

@click.group()
def cli():
    pass

# dairy
@cli.command()
def dairy():
    df = pd.read_csv('../h5n1/wahis_bovine_reports.csv')
    df['start_date'] = pd.to_datetime(df.start_date)
    df['end_date'] = pd.to_datetime(df.end_date).fillna(-1)
    df.loc[:, 'quarter'] = df.start_date.dt.quarter
    df.loc[:, 'month'] = df.start_date.dt.month
    df.loc[:, 'year'] = df.start_date.dt.year
    
    counties = loader.load('usa_counties')
    df = df.merge(counties, on=['state', 'county'], how='left')
    df = df.rename(columns={'statefp': 'state_code', 'countyfp': 'county_code'})

    df.to_csv('dairy.csv', index=False)

# poultry
@cli.command()
def poultry():
    tdf = pd.read_excel('../h5n1/poultry.xlsx', skiprows=1)
    tdf = tdf.drop(['Active', 'Special Id', 'NA'], axis=1)
    cols = [x for x in tdf.columns if '-' in x]
    tdf['heads'] = tdf[cols].max(axis=1).fillna(-1)
    tdf['heads_date'] = tdf[cols].idxmax(axis=1).fillna(-1)
    ## dates = [x for x in list(tdf.columns) if x[0].isdigit()]
    ## df = pd.melt(tdf, id_vars=['Confirmed', 'State', 'County Name', 
    ##                            'Production'],
    ##              var_name='row_date')
    df = tdf.drop(cols, axis=1)

    # standardization
    df.columns = [x.lower() for x in df.columns]
    df.production = df.production.str.lower()
    df.state = df.state.str.lower()
    df = df.rename(columns={'county name': 'county'})
    df.county = df.county.str.lower()

    # forward fills and absence of information
    df.confirmed = df.confirmed.ffill()
    df.state = df.state.ffill()
    df.county = df.county.ffill()

    # attributes
    df['commercial'] = df.production.str.contains('commercial')
    for k,v in {'woah non-poultry': 'backyard',
                'woah poultry': 'backyard',
                'turkey': 'turkeys',
                'duck': 'ducks',
                'broiler breeder pullets': 'ckn-pullets',
                'table egg': 'ckn-layers',
                'broiler production': 'ckn-broilers',
                'broiler breeder': 'ckn-broilers',
                'upland gamebird': 'mixed',
                'upland game bird': 'mixed',
                'commercial breeder operation': 'mixed',
                r'commercial breeder \(multiple bird species\)': 'mixed',
                'live bird market': 'mixed',
                'live bird sales': 'mixed',
                'waterfowl': 'waterfowl'}.items(): 
        ind = df.production.str.contains(k)
        df.loc[ind, 'type'] = v
        df.loc[ind, 'production'] = ''
    ## df = df.groupby(['confirmed', 'state', 'county', 'production'],
    ##                 as_index=False)['value'].max()
    df.confirmed = pd.to_datetime(df.confirmed)
    df.loc[:, 'quarter'] = df.confirmed.dt.quarter
    df.loc[:, 'year'] = df.confirmed.dt.year
    df.loc[:, 'month'] = df.confirmed.dt.month

    df = df.rename(columns={'confirmed': 'start_date', 
                            'heads_date': 'end_date'})
    df.end_date = pd.to_datetime(df.end_date).fillna(-1)
    df.loc[df.end_date<df.start_date, 'end_date'] = -1

    # fips
    counties = loader.load('usa_counties')
    df = df.merge(counties[['state_code', 'state', 'county_code', 'county']], 
                  on=['state', 'county'])

    # farm size
    df['size_category'] = pd.cut(df.heads, bins=FARM_SIZE,
                                 labels=FARM_SIZE_NAMES, right=False)
    df.size_category = df.size_category.astype('str')

    df.to_csv('poultry.csv', index=False)
    return

# wild birds
@cli.command()
def birds():
    df = pd.read_csv('../h5n1/birds_original.csv')

    df.columns = [x.lower() for x in df.columns]
    df = df.rename(columns={'collection date': 'collection_date'}) 

    df.loc[df.collection_date=='Unknown', 'collection_date'] = df[
            df.collection_date=='Unknown']['date detected']
    df.state = df.state.str.lower()
    df.county = df.county.str.lower()

    df.collection_date = pd.to_datetime(df.collection_date)
    df.loc[:, 'quarter'] = df.collection_date.dt.quarter
    df.loc[:, 'year'] = df.collection_date.dt.year.astype(int)
    df.loc[:, 'month'] = df.collection_date.dt.month.astype(int)

    counties = loader.load('usa_counties')
    df = df.merge(counties, on=['state', 'county'])
    df.to_csv('birds.csv', index=False)
    return

# human
@cli.command()
def human_preprocess():
    df = pd.read_csv('../h5n1/h5n1_global_health.csv', delimiter='\t',
                     names=['date', 'notes', 'source', 'category'])
    df[['state', 'county', 'region', 'news_type']] = '' 
    df = df.drop('category', axis=1)
    #df.date = pd.to_datetime(df.date)
    df = df.sort_values('date').reset_index(drop=True)
    pd.set_option("display.max_colwidth", None)
    df.to_excel("human.xlsx", index=False)
    return

if __name__ == '__main__':
    cli()
