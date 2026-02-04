DESC='''
Comparing GLW with AgLand based on MWs analysis.
'''

from aaviz import plot
from aadata import loader
import geopandas as gpd
import logging
import numpy as np
from os import getenv
import pandas as pd
from pdb import set_trace
import psycopg2 as pg
from re import sub

def correlate(df):
    return df.agcensus.corr(df.glw)

def weighted_correlate(df):
    return df.weighted.corr(df.glw)


def main():
    # Load data
    glw = gpd.read_file('../../data/glw/glw.shp.zip')
    agcensus = pd.read_csv('../../data/agcensus/agcensus_farms.csv.zip')
    agcensus = agcensus.drop(['state_fips', 'data_item', 'domain', 
                              'domain_category'], axis=1)
    agcensus['avg_size'] = (agcensus.size_min + agcensus.size_max)/2
    agcensus.loc[agcensus.avg_size.isnull(), 'avg_size'] = 1
    agcensus['weighted'] = agcensus.avg_size * agcensus.num_farm
    counties = loader.load('usa_county_shapes')
    counties = counties[counties.statefp=='53'][
            ['countyfp', 'name', 'geometry']]
    counties = counties.astype({'countyfp': 'int'})
    counties = counties.to_crs('EPSG:4326')
    
    # Assign each glw cell to a county and get county-level counts
    glw_county = gpd.sjoin(left_df=glw, right_df=counties, 
                     how="left", predicate="intersects")
    glw_county = glw_county.drop(['x', 'y', 'geometry', 'index_right'], 
                                 axis=1).groupby(
                                         ['livestock', 'countyfp', 'name']
                                         ).sum().reset_index()
    glw_county = glw_county.rename(columns={'val': 'glw'})

    # Rename livestock names
    name_map = {'goat': 'goats', 'chickens': 'poultry', 'duck': 'poultry',
                'pig': 'hogs', 'cattle': 'cattle'}
    glw_county.livestock = glw_county.livestock.map(name_map)

    # Get total farm counts from agcensus
    agt = agcensus[['county_fips', 'commodity', 'num_farm', 'weighted']
                   ].groupby(['county_fips', 'commodity']).sum().reset_index()
    agt = agt.rename(columns={'num_farm': 'agcensus'})

    # Join the two
    df = glw_county.merge(agt, how='inner', left_on=['countyfp', 'livestock'],
                          right_on=['county_fips', 'commodity'])

    # Correlations
    corr = df.groupby("livestock").apply(correlate)
    corr_weighted = df.groupby("livestock").apply(weighted_correlate)
    print(f'General correlation: {df.agcensus.corr(df.glw)}')
    print(f'Livestock-specific correlation: {corr}')
    print(f'Weighted correlation: {df.weighted.corr(df.glw)}')
    print(f'Weighted livestock-specific correlation: {corr_weighted}')

    # Plots
    fig, gs = plot.initiate_figure(x=25, y=5, color='tableau10',
                                   gs_nrows=1, gs_ncols=5,
                                   gs_wspace=.3, gs_hspace=.4)

    ax = plot.subplot(fig=fig, grid=gs[0,0], func='sns.barplot', 
                      data=df[['livestock', 'agcensus']
                              ].groupby(['livestock']).sum().reset_index(), 
                      pf_y='agcensus', pf_x='livestock',
                      la_title='AgCensus: \\#farms',
                      pf_color=plot.get_style('color',1),
                      fs_xlabel='Large',
                      xt_rotation=25, la_xlabel='(a)', la_ylabel='')

    ax = plot.subplot(fig=fig, grid=gs[0,1], func='sns.barplot', 
                      data=df[['livestock', 'weighted']
                              ].groupby(['livestock']).sum().reset_index(), 
                      pf_y='weighted', pf_x='livestock',
                      pf_color=plot.get_style('color',2),
                      la_title='\\parbox{10cm}{\\center AgCensus: sum of approx. farm sizes}',
                      fs_xlabel='Large',
                      xt_rotation=25, la_xlabel='(b)', la_ylabel='')
    plot.text(ax=ax, data='No sizes given', x=3, y=25000, rotation=90, 
              backgroundcolor='white', fontsize='large')

    ax = plot.subplot(fig=fig, grid=gs[0,2], func='sns.barplot', 
                      data=df[['livestock', 'glw']
                              ].groupby(['livestock']).sum().reset_index(), 
                      pf_y='glw', pf_x='livestock',
                      la_title='GLW: Total weight',
                      sp_yscale='log',
                      pf_color=plot.get_style('color',0),
                      fs_xlabel='Large',
                      xt_rotation=25, la_xlabel='(c)', la_ylabel='')

    ax = plot.subplot(fig=fig, grid=gs[0,3], func='sns.barplot', data=corr, 
                      la_title='County-level correlation',
                      pf_color=plot.get_style('color',3),
                      fs_xlabel='Large',
                      xt_rotation=25, la_xlabel='(d)', la_ylabel='')

    ax = plot.subplot(fig=fig, grid=gs[0,4], func='sns.barplot', 
                      data=corr_weighted, 
                      pf_color=plot.get_style('color',4),
                      la_title='\\parbox{10cm}{\\center County-level correlation accounting for farm sizes}',
                      fs_xlabel='Large',
                      xt_rotation=25, la_xlabel='(e)', la_ylabel='')

    plot.savefig('glw_agcensus.pdf')

if __name__ == '__main__':
    main()

