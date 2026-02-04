DESC='''
Agcensus data preparation.

By: AA & DX
'''

import geopandas as gpd
import logging
import numpy as np
import pandas as pd
from pdb import set_trace
from re import sub

def size_to_int(string):
    try:
        return int(sub(',', '', string))
    except:
        return -1

def load_data():
    df = pd.read_csv('../agcensus/agcensus_full_data.csv.zip')

    # Process
    df = df[df.state_fips_code!=99]
    df.loc[(df.value=='(D)') | (df.value=='(Z)'), 'value'] = '-1'
    df.loc[df.county_code.isnull(), ['county_code', 'county_name']] = (-1, ' ')
    df.value = df.value.str.replace(',','').astype(float)
    df['subtype'] = 'unassigned'
    df['category'] = 'unassigned'
    df = df.rename(columns={'commodity_desc': 'livestock'})
    df = df[~df.domain_desc.isin(
        ['ORGANIZATION', 'NAICS CLASSIFICATION', 'AREA OPERATED', 
         'CONCENTRATION', 'ECONOMIC CLASS', 'FARM SALES', 
         'PRODUCERS', 'IRRIGATION STATUS', 'TENURE'])]
    df = df[~df.domain_desc.str.contains(
        'ORGANIZATION|NAICS|AREA|CONCENTRATION|ECONOMIC|SALES|PRODUCERS|IRRIGATION|STATUS|TENURE', regex=True)]
    return df
                   
def sanity_check(df):
    if df.subtype.isnull().sum():
        raise('Missed some types.')
    if (df.category=='unassigned').any():
        raise('Missed some categories.')

    print(df.category.value_counts())

def farmsize(df):
    # get sizes
    df['size_min'] = df.domaincat_desc.str.replace(
            r'.*\(', '', regex=True).replace(' .*', '', regex=True)
    df['size_max'] = df.domaincat_desc.str.replace(
            '.* TO ', '', regex=True).replace(' .*', '', regex=True)

    df.size_min = df.size_min.apply(size_to_int)
    df.size_max = df.size_max.apply(size_to_int)

    df = df.drop(df[(df.livestock=='cattle') &
                    (df.size_min==10) &
                    (df.size_max==49)].index)
    df = df.drop(df[(df.livestock=='cattle') &
                     (df.size_min>500)].index)
    df = df.drop(df[(df.livestock=='cattle') &
                     (df.size_min==500) &
                     (df.size_max!=-1)].index)

    df = df.drop(df[(df.livestock=='hogs') &
                             (df.size_min>1000)].index)
    df = df.drop(df[(df.livestock=='hogs') &
                             (df.size_min==500) &
                             (df.size_max==-1)].index)
    df = df.drop(df[(df.livestock=='hogs') &
                             (df.size_min==1000) &
                             (df.size_max!=-1)].index)

    df = df.drop(df[(df.livestock=='sheep') &
                    (df.size_min>1000)].index)
    df = df.drop(df[(df.livestock=='sheep') &
                    (df.size_min==1000) &
                    (df.size_max!=-1)].index)

    df = df.drop(df[(df.livestock=='ckn-layers') &
                    (df.size_min==1) &
                    (df.size_max==399)].index)

    return df

def get_counts():
    df = load_data()
    df = df[(df.unit_desc=='HEAD') | (df.unit_desc=='OPERATIONS')]
    df = df[df.statisticcat_desc=='INVENTORY'].copy()
    #df = df.drop(['unit_desc'], axis=1)

    heads = []

    # cattle
    print('--------------------------------------------------\ncattle')
    dfs = df[df.livestock=='CATTLE'].copy()
    dfs = dfs[~dfs.class_desc.isin(['ALL CLASSES', 'COWS'])]
    dfs = dfs[~dfs.domaincat_desc.str.contains(r'\(0 HEAD\)')]
    dfs = dfs[~dfs.domaincat_desc.str.contains('1 OR MORE HEAD')]
    dfs = dfs[~((dfs.domain_desc.isin(['INVENTORY OF MILK COWS',
                                       'INVENTORY OF BEEF COWS',
                                       'INVENTORY OF COWS'])) &
              (dfs.class_desc.isin(['INCL CALVES', '(EXCL COWS)'])))]
    dfs.subtype = dfs.class_desc
    dfs = dfs[~((dfs.subtype=='(EXCL COWS)') & 
                (dfs.domain_desc=='INVENTORY OF CATTLE, INCL CALVES'))]
    dfs.subtype = dfs.subtype.map({'COWS, MILK': 'milk', 'COWS, BEEF': 'beef',
                             '(EXCL COWS)': 'other',
                             'INCL CALVES': 'all',
                             })

    ### state totals
    dfs.loc[(dfs.domain_desc=='TOTAL') &
            (dfs.agg_level_desc=='STATE'), 'category'] = 'state_total'
    dfs.loc[(dfs.domain_desc!='TOTAL') &
            (dfs.agg_level_desc=='STATE'),
                'category'] = 'state_by_farmsize'
    dfs.loc[(dfs.domain_desc=='TOTAL') & (dfs.agg_level_desc=='COUNTY'), 
            'category'] = 'county_total'
    dfs.loc[(dfs.domain_desc!='TOTAL') &
            (dfs.agg_level_desc=='COUNTY'), 'category'] = 'county_by_farmsize'
    dfs.to_csv('cattle_mod.csv', index=False)

    sanity_check(dfs)
    heads.append(dfs)

    # sheep
    print('--------------------------------------------------\nsheep')
    dfs = df[(df.livestock=='SHEEP') &
             (df.class_desc=='INCL LAMBS')].copy()
              
    dfs.subtype = 'all'

    dfs.loc[(dfs.domain_desc=='TOTAL') & (dfs.agg_level_desc=='COUNTY'), 
            'category'] = 'county_total'
    dfs.loc[(dfs.domain_desc!='TOTAL') &
            (dfs.agg_level_desc=='COUNTY'), 'category'] = 'county_by_farmsize'
    dfs.loc[(dfs.domain_desc!='TOTAL') &
            (dfs.agg_level_desc=='STATE') &
            (dfs.domaincat_desc.str.contains(
                r'INVENTORY OF .*:', regex=True)), 
                'category'] = 'state_by_farmsize'
    dfs.loc[(dfs.domain_desc=='TOTAL') & 
            (dfs.agg_level_desc=='STATE'), 'category'] = 'state_total'

    ### Introducing a dummy subtype for downstream operations
    tdf = dfs.copy()
    tdf.subtype = 'dummy'
    dfs = pd.concat([dfs, tdf])

    sanity_check(dfs)
    heads.append(dfs)

    # hogs
    # AA: Look at short_desc for hog types
    print('--------------------------------------------------\nhogs')
    dfs = df[(df.short_desc=='HOGS - INVENTORY') |
             (df.short_desc=='HOGS - OPERATIONS WITH INVENTORY')].copy()
    dfs.subtype = 'all'

    dfs.loc[(dfs.domain_desc=='TOTAL') & (dfs.agg_level_desc=='COUNTY'), 
            'category'] = 'county_total'
    dfs.loc[(dfs.domain_desc!='TOTAL') &
            (dfs.agg_level_desc=='COUNTY'), 'category'] = 'county_by_farmsize'
    dfs.loc[(dfs.domain_desc!='TOTAL') &
            (dfs.agg_level_desc=='STATE') &
            (dfs.domaincat_desc.str.contains(
                r'INVENTORY OF HOGS:', regex=True)), 
                'category'] = 'state_by_farmsize'
    dfs.loc[(dfs.domain_desc=='TOTAL') & 
            (dfs.agg_level_desc=='STATE'), 'category'] = 'state_total'
    dfs = dfs[dfs.category!='unassigned']

    ### Introducing a dummy subtype for downstream operations
    tdf = dfs.copy()
    tdf.subtype = 'dummy'
    dfs = pd.concat([dfs, tdf])

    sanity_check(dfs)
    heads.append(dfs)

    # poultry
    print('--------------------------------------------------\npoultry except chicken')
    dfs = df[(df.group_desc=='POULTRY') &
             (df.livestock!='CHICKENS') & 
             (df.livestock!='POULTRY TOTALS')].copy()
    dfs.loc[dfs.livestock=='POULTRY, OTHER', 'livestock'] = 'POULTRY-OTHER'
    dfs.loc[dfs.livestock=='PIGEONS & SQUAB', 'livestock'] = 'PIGEONS'
    dfs = dfs.drop(dfs[(dfs.livestock=='POULTRY-OTHER') &
                       (dfs.class_desc=='INCL DUCKS & GEESE')].index)
    print('Poultry totals ignored')
    #dfs.loc[dfs.livestock=='POULTRY TOTALS', 'livestock'] = 'ALL'
    dfs.subtype = 'all'
    ## dfs.subtype = dfs.livestock
    ## dfs.livestock = 'POULTRY'

    dfs.loc[(dfs.domain_desc=='TOTAL') & (dfs.agg_level_desc=='COUNTY'), 
            'category'] = 'county_total'
    dfs.loc[(dfs.domain_desc!='TOTAL') &
            (dfs.agg_level_desc=='COUNTY'), 
            'category'] = 'county_by_farmsize'
    dfs.loc[(dfs.domain_desc!='TOTAL') &
            (dfs.agg_level_desc=='STATE') &
            (dfs.domaincat_desc.str.contains(
                r'INVENTORY:', regex=True)), 
                'category'] = 'state_by_farmsize'
    dfs.loc[(dfs.domain_desc=='TOTAL') & 
            (dfs.agg_level_desc=='STATE'), 'category'] = 'state_total'

    ### Introducing a dummy subtype for downstream operations
    tdf = dfs.copy()
    tdf.subtype = 'dummy'
    dfs = pd.concat([dfs, tdf])

    sanity_check(dfs)
    heads.append(dfs)

    # chickens except layers
    print('--------------------------------------------------\npoultry except chicken layers')
    dfs = df[(df.livestock=='CHICKENS') & (df.class_desc!='LAYERS')].copy()
    dfs.livestock = dfs.class_desc
    dfs.livestock = dfs.livestock.map({'ROOSTERS': 'ckn-roosters', 
                             'PULLETS, REPLACEMENT': 'ckn-pullets',
                             'BROILERS': 'ckn-broilers',
                             })
    dfs.subtype = 'all'

    dfs.loc[(dfs.domain_desc=='TOTAL') & (dfs.agg_level_desc=='COUNTY'), 
            'category'] = 'county_total'
    dfs.loc[(dfs.domain_desc!='TOTAL') &
            (dfs.agg_level_desc=='COUNTY'), 'category'] = 'county_by_farmsize'
    dfs.loc[(dfs.domain_desc!='TOTAL') &
            (dfs.agg_level_desc=='STATE') &
            (dfs.domaincat_desc.str.contains(
                r'INVENTORY:', regex=True)), 
                'category'] = 'state_by_farmsize'
    dfs.loc[(dfs.domain_desc=='TOTAL') & 
            (dfs.agg_level_desc=='STATE'), 'category'] = 'state_total'
    #dfs.livestock = 'POULTRY'

    ## dfs = dfs.drop(dfs[dfs.category=='state_by_farmsize'].index)
    ## dfs = dfs.drop(dfs[dfs.category=='county_by_farmsize'].index)

    ### Introducing a dummy subtype for downstream operations
    tdf = dfs.copy()
    tdf.subtype = 'dummy'
    dfs = pd.concat([dfs, tdf])

    sanity_check(dfs)
    heads.append(dfs)

    # chickens layers
    print('--------------------------------------------------\nchicken layers')
    dfs = df[(df.livestock=='CHICKENS') & (df.class_desc=='LAYERS')].copy()
    dfs.livestock = 'ckn-layers'
    dfs.subtype = 'all'

    dfs.loc[(dfs.domain_desc=='TOTAL') & (dfs.agg_level_desc=='COUNTY'), 
            'category'] = 'county_total'
    dfs.loc[(dfs.domain_desc!='TOTAL') &
            (dfs.agg_level_desc=='COUNTY'), 'category'] = 'county_by_farmsize'
    dfs.loc[(dfs.domain_desc!='TOTAL') &
            (dfs.agg_level_desc=='STATE') &
            (dfs.domaincat_desc.str.contains(
                r'INVENTORY:', regex=True)), 
                'category'] = 'state_by_farmsize'
    dfs.loc[(dfs.domain_desc=='TOTAL') & 
            (dfs.agg_level_desc=='STATE'), 'category'] = 'state_total'

    ## dfs = dfs.drop(dfs[dfs.category=='state_by_farmsize'].index)
    ## dfs = dfs.drop(dfs[dfs.category=='county_by_farmsize'].index)

    ### Introducing a dummy subtype for downstream operations
    tdf = dfs.copy()
    tdf.subtype = 'dummy'
    dfs = pd.concat([dfs, tdf])

    sanity_check(dfs)
    heads.append(dfs)

    # Post process
    adf = pd.concat(heads)
    adf = adf.rename(columns={'unit_desc': 'unit'})

    for col in ['livestock', 'state_name', 'unit', 'subtype', 'county_name']:
        adf[col] = adf[col].str.lower()
    adf.loc[adf.unit=='head', 'unit'] = 'heads'

    adf = farmsize(adf)
    cols = ['livestock', 'subtype', 'unit', 
            'state_fips_code', 'state_name', 'county_code', 'county_name', 
            'size_min', 'size_max', 'category']
    adf= adf[cols + ['value']]

    adf.subtype = adf.subtype.str.replace('&', 'and')
    adf.livestock = adf.livestock.str.replace('&', 'and')

    adf = adf.rename(columns={'state_fips_code': 'state_code',
                              'state_name': 'state', 'county_name': 'county'})

    adf.to_csv('agcensus_heads_farms.csv.zip', index=False)

    print('-----\nSummary\n-----')
    adf.loc[adf.value==-1, 'value'] = 0
    df1 = adf[adf.category=='state_total'][
            ['livestock', 'subtype', 'unit', 'value']].groupby(
                    ['livestock', 'subtype', 'unit']).sum()
    df2 = adf[adf.category=='state_by_farmsize'][
            ['livestock', 'subtype', 'unit', 'value']].groupby(
                    ['livestock', 'subtype', 'unit']).sum()
    df3 = adf[adf.category=='county_total'][
            ['livestock', 'subtype', 'unit', 'value']].groupby(
                    ['livestock', 'subtype', 'unit']).sum()
    df4 = adf[adf.category=='county_by_farmsize'][
            ['livestock', 'subtype', 'unit', 'value']].groupby(
                    ['livestock', 'subtype', 'unit']).sum()
    df1 = df1.join(df2, lsuffix='-state', rsuffix='-state-farmsize')
    df1 = df1.join(df3)
    df1 = df1.join(df4, lsuffix='-county', rsuffix='-county-farmsize')
    df1 = df1.fillna(0)
    df1.columns = [x[6:] for x in df1.columns]
    summary = df1.astype({'state': 'int', 'county': 'int',
                          'state-farmsize': 'int', 
                          'county-farmsize': 'int'})
    print(summary)

    summary = summary.reset_index()
    summary_ = pd.melt(summary, id_vars=['livestock', 'subtype', 'unit'],
                       var_name='value_type', value_name='value')

    ref_summary = pd.read_csv('../agcensus/agcensus_totals.csv')
    tdf = summary_.merge(ref_summary, on=['livestock', 'subtype', 
                                               'unit', 'value_type'],
                              how='outer')

    diff = (tdf.value_x-tdf.value_y).abs()
    set_trace()

    if diff.isnull().sum() or diff.sum():
        summary_.to_csv('agcensus_totals.csv', index=False)
        raise ValueError('Current data has deviated from the reference. If the current counts are correct, then replace "agcensus_totals.csv" in ../agcensus/.')
    else:
        print('Current data matches reference totals.')

if __name__ == '__main__':
    get_counts()
