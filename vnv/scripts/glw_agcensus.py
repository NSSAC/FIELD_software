DESC='''
Comparing GLW with AgCensus on head and farm counts/density.
'''

from aaviz import plot
# from aadata import loader
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

def farms():
    # Load data
    glw = pd.read_csv('../../data/glw/glw_sans_geom_processed.csv.zip')
    agcensus = pd.read_csv('../../data/agcensus/agcensus_farms_processed.csv.zip')
    agcensus['weighted'] = agcensus.avg_size * agcensus.num_farm

    glw_county = glw[['statefp', 'countyfp', 'livestock', 'val']].groupby(
                                         ['livestock', 'statefp', 'countyfp']
                                         ).sum().reset_index()
    glw_county = glw_county.rename(columns={'val': 'glw'})

    # Get total farm counts from agcensus
    agt = agcensus[['state_fips', 'county_fips', 'commodity', 
                    'num_farm', 'weighted']
                   ].groupby(['state_fips', 'county_fips', 'commodity']
                             ).sum().reset_index()
    agt = agt.rename(columns={'num_farm': 'agcensus'})

    # Join the two
    df = glw_county.merge(agt, how='inner', 
                          left_on=['statefp', 'countyfp', 'livestock'],
                          right_on=['state_fips', 'county_fips', 'commodity'])

    # Correlations
    corr = df.groupby("livestock").apply(correlate)
    corr_weighted = df.groupby("livestock").apply(weighted_correlate)

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
                      sp_yscale='log',
                      xt_rotation=25, la_xlabel='(a)', la_ylabel='')

    ax = plot.subplot(fig=fig, grid=gs[0,1], func='sns.barplot', 
                      data=df[['livestock', 'weighted']
                              ].groupby(['livestock']).sum().reset_index(), 
                      pf_y='weighted', pf_x='livestock',
                      pf_color=plot.get_style('color',2),
                      sp_yscale='log',
                      la_title='\\parbox{10cm}{\\center AgCensus: sum of approx. farm sizes}',
                      fs_xlabel='Large',
                      xt_rotation=25, la_xlabel='(b)', la_ylabel='')
    plot.text(ax=ax, data='Ignore. No sizes given', x=2.75, y=100000, 
              rotation=90, backgroundcolor='white', fontsize='large')
    plot.text(ax=ax, data='Ignore. No sizes given', x=0.75, y=100000, 
              rotation=90, backgroundcolor='white', fontsize='large')

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
                      la_title='\\parbox{9.5cm}{\\center County-level correlation based on number of farms}',
                      pf_color=plot.get_style('color',3),
                      fs_xlabel='Large',
                      xt_rotation=25, la_xlabel='(d)', la_ylabel='')

    ax = plot.subplot(fig=fig, grid=gs[0,4], func='sns.barplot', 
                      data=corr_weighted, 
                      pf_color=plot.get_style('color',4),
                      la_title='\\parbox{9.5cm}{\\center County-level correlation accounting for farm sizes}',
                      fs_xlabel='Large',
                      xt_rotation=25, la_xlabel='(e)', la_ylabel='')
    plot.text(ax=ax, data='Ignore. No sizes given', x=2.75, y=0.01, 
              rotation=90, backgroundcolor='white', fontsize='large')
    plot.text(ax=ax, data='Ignore. No sizes given', x=0.75, y=0.01, 
              rotation=90, backgroundcolor='white', fontsize='large')

    plot.savefig('glw_agcensus_farms.pdf')

def heads():
    # Load data
    glw = pd.read_csv('../../data/glw/glw_sans_geom_processed.csv.zip')

    agcensus = pd.read_csv(
            '../../data/agcensus/agcensus_heads_processed.csv.zip')

    glw_county = glw[['statefp', 'countyfp', 'livestock', 'val']].groupby(
                                         ['livestock', 'statefp', 'countyfp']
                                         ).sum().reset_index()
    glw_county = glw_county.rename(columns={'val': 'glw'})

    # Join the two
    df = glw_county.merge(agcensus, how='inner', 
                          left_on=['statefp', 'countyfp', 'livestock'],
                          right_on=['state_fips_code', 'county_code', 
                                    'commodity_desc'])
    df = df.rename(columns={'value': 'agcensus'})

    if df.livestock.isnull().sum():
        raise('Found nulls after mapping.')

    # Correlations
    corr = df.groupby("livestock").apply(correlate)

    # Plots
    fig, gs = plot.initiate_figure(x=20, y=5, color='tableau10',
                                   gs_nrows=1, gs_ncols=3,
                                   gs_wspace=.3, gs_hspace=.4)

    ax = plot.subplot(fig=fig, grid=gs[0,0], func='sns.barplot', 
                      data=df[['commodity_desc', 'agcensus']
                              ].groupby(
                                  ['commodity_desc']).sum().reset_index(), 
                      pf_y='agcensus', pf_x='commodity_desc',
                      la_title='AgCensus: \\#livestock',
                      pf_color=plot.get_style('color',1),
                      fs_xlabel='Large',
                      sp_yscale='log',
                      xt_rotation=25, la_xlabel='(a)', la_ylabel='')

    ax = plot.subplot(fig=fig, grid=gs[0,1], func='sns.barplot', 
                      data=df[['livestock', 'glw']
                              ].groupby(['livestock']).sum().reset_index(), 
                      pf_y='glw', pf_x='livestock',
                      la_title='GLW: Total weight',
                      sp_yscale='log',
                      pf_color=plot.get_style('color',0),
                      fs_xlabel='Large',
                      xt_rotation=25, la_xlabel='(b)', la_ylabel='')

    ax = plot.subplot(fig=fig, grid=gs[0,2], func='sns.barplot', data=corr, 
                      la_title='County-level correlation',
                      pf_color=plot.get_style('color',4),
                      fs_xlabel='Large',
                      xt_rotation=25, la_xlabel='(c)', la_ylabel='')

    plot.savefig('glw_agcensus_heads.pdf')
    return

if __name__ == '__main__':
    farms()
    #heads()
