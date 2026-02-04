DESC = '''
Statistics of the farms to cell assignment.

By: AA
'''

from aadata import loader
from aaviz import plot
import geopandas as gpd
import numpy as np
import pandas as pd
from pdb import set_trace

from aautils.display import pf

@pf
def farm_size(farms, agcensus):
    livestocks = farms.livestock.drop_duplicates()

    fig, gs = plot.initiate_figure(x=20, y=4, 
                                   gs_nrows=1, gs_ncols=4,
                                   gs_wspace=.1, gs_hspace=.4,
                                   color='tableau10')

    # cattle
    axs = {}
    for i,livestock in enumerate(['cattle', 'poultry', 'hogs', 'sheep']):
        hue_order = ['all']
        if livestock=='poultry':
            df = farms[(farms.subtype.str.contains('ckn'))
                       ].sort_values('heads').groupby('subtype').apply(
                               lambda x: x.reset_index(drop=True).reset_index())
            livestock = 'chickens'
            hue_order = df.subtype.drop_duplicates().sort_values().tolist()
        else:
            df = farms[(farms.livestock==livestock)
                       ].sort_values('heads').groupby('subtype').apply(
                               lambda x: x.reset_index(drop=True).reset_index())

        ylabel = ''
        if livestock == 'cattle':
            ylabel = r'\#Heads'
            hue_order = ['all', 'beef', 'milk', 'other']

        xmin = {'cattle': 300000,
                'chickens': 10000,
                'hogs': 5000,
                'sheep': 5000}

        ax = plot.subplot(fig=fig, grid=gs[0,i], func='sns.lineplot', 
                          data=df, pf_hue='subtype',
                          pf_y='heads', pf_x=df['index'],
                          pf_hue_order = hue_order,
                          sp_yscale='log',
                          sp_xscale='linear',
                          la_xlabel='Farms ordered by size',
                          la_ylabel=ylabel, 
                          la_title=livestock)
        ## ax_inset = inset_axes(ax, width="50%", height="40%", 
        ##                       bbox_to_anchor=(-.25, -.45, 1, 1), 
        ##                       bbox_transform=ax.transAxes)
        ## plot.subplot(ax=ax_inset, fig=fig, func='sns.lineplot', 
        ##              data=df, pf_hue='subtype',
        ##              pf_y='heads', pf_x='index',
        ##              pf_hue_order = hue_order,
        ##              pf_legend=False, 
        ##              sp_yscale='log',
        ##              sp_xscale='log',
        ##              sp_xlim=(xmin[livestock],'default'),
        ##              la_xlabel='',
        ##              la_ylabel='',
        ##              la_title='')

    # poultry
    plot.savefig('livestock_farm_sizes.pdf')

@pf
def heads(farms, agcensus):
    tdf = agcensus[(agcensus.unit=='heads') & 
                   (agcensus.category=='state_total')]
    tdf = tdf.drop(tdf[(tdf.livestock!='cattle') & (tdf.subtype=='all')].index)
    tdf.loc[(tdf.livestock!='poultry') & (tdf.subtype=='dummy'), 'subtype'
            ] = 'all'

    ag_count = tdf.groupby(['state_code', 'livestock', 'subtype']
                           )['value'].sum().reset_index()
    poultry = ag_count.livestock.drop_duplicates().tolist()

    for ll in ['cattle', 'hogs', 'sheep']:
        poultry.remove(ll)
    pind = ag_count.livestock.isin(poultry)
    ag_count.loc[pind, 'subtype'] = ag_count[pind].livestock
    ag_count.loc[pind, 'livestock'] = 'poultry'

    fc_count = farms.groupby(
            ['state_code', 'livestock', 'subtype']
            )['heads'].sum().reset_index()

    fc_count = fc_count.merge(ag_count, 
                              on=['state_code', 'livestock', 'subtype'],
                              how='left')
    if fc_count.value.isnull().sum():
        print('Nulls found, some assignments have not gone through.')
        fc_count = fc_count[~fc_count.value.isnull()]

    fc_count.loc[:, 'comp'] = 2 * abs(fc_count.value-fc_count.heads)/(
            fc_count.value+fc_count.heads) * 100

    # initiate figure
    types = fc_count.livestock.drop_duplicates().tolist()
    fig, gs = plot.initiate_figure(x=16, y=4, 
                                   gs_nrows=1, gs_ncols=10,
                                   gs_wspace=.5, gs_hspace=.4,
                                   color='tableau10')
    ## print('''Pending:
    ## * outliers
    ## * set yticks to max, otherwise, sharey creates issues.
    ## ''')
    
    for i,livestock in enumerate(types):
        if i == 0:
            ylabel = r'\% mean-normalized abs. difference'
        else:
            ylabel = ''
        if livestock == 'cattle':
            hue_order = ['all', 'beef', 'milk', 'other']
            grid = gs[0,0:2]
        elif livestock == 'hogs':
            grid = gs[0,2]
        elif livestock == 'sheep':
            grid = gs[0,3]
        elif livestock == 'poultry':
            grid = gs[0,4:]
        ax = plot.subplot(fig=fig, grid=grid, func='sns.boxplot', 
                          data=fc_count[fc_count.livestock==livestock], 
                          pf_x='subtype', pf_y='comp',
                          pf_color=plot.get_style('color',i),
                          pf_showfliers=True,
                          xt_rotation=90, la_ylabel=ylabel,
                          la_xlabel='', 
                          la_title=livestock)
                          #sp_sharey=0,

    plot.savefig('livestock_heads_comparison.pdf')

@pf
def gen_farms(stats, agcensus):

    # county totals
    dfn = agcensus[(agcensus.category=='county_total') &
                  (agcensus.unit=='heads') &
                  (agcensus.subtype!='all')]
    dfa = agcensus[(agcensus.category=='county_total') &
                  (agcensus.unit=='heads') &
                  (agcensus.subtype=='all')]
    dfn_ = dfn[['state_code', 'county_code', 'livestock', 'value']
             ].groupby(['state_code', 'county_code', 'livestock']).sum().reset_index()
    dfa_ = dfa[['state_code', 'county_code', 'livestock', 'value_original']
             ].groupby(['state_code', 'county_code', 'livestock']).sum().reset_index()
    stats = stats.merge(
            dfn_[['state_code', 'county_code', 'livestock', 'value']], 
            left_on=['state', 'county', 'livestock'], 
            right_on=['state_code', 'county_code', 'livestock'], 
            how='left')
    stats = stats.merge(
            dfa_[['state_code', 'county_code', 'livestock', 'value_original']], 
            left_on=['state', 'county', 'livestock'], 
            right_on=['state_code', 'county_code', 'livestock'], 
            how='left')

    # initiate figure
    fig, gs = plot.initiate_figure(x=5, y=4, 
                                   gs_nrows=1, gs_ncols=1,
                                   gs_wspace=.3, gs_hspace=.4,
                                   color='tableau10')

    stats['l1'] = stats.lambda1/stats.value
    ax = plot.subplot(fig=fig, grid=gs[0,0], func='sns.boxplot', data=stats, 
                      pf_x='livestock', pf_y='l1',
                      xt_rotation=50, 
                      la_ylabel=r'$\lambda_1$/\#heads',
                      pf_color=plot.get_style('color',4),
                      la_title='Generation of farms',
                      la_xlabel='')
    ## ax = plot.subplot(fig=fig, grid=gs[0,1], func='sns.boxplot', 
    ##                   data=stats[stats.value_original!=-1], 
    ##                   pf_x='livestock', pf_y='l1',
    ##                   pf_color=plot.get_style('color',4),
    ##                   xt_rotation=50, 
    ##                   la_title=r'\parbox{7cm}{\centering Generation of farms (County totals present)}',
    ##                   la_ylabel=r'$\lambda_1$/\#heads',
    ##                   la_xlabel='')
    plot.savefig('livestock_gen_farms.pdf')

@pf
def farms_to_cells(stats):
    # initiate figure
    fig, gs = plot.initiate_figure(x=5*2, y=4, 
                                   gs_nrows=1, gs_ncols=2,
                                   gs_wspace=.3, gs_hspace=.4,
                                   color='tableau10')
    stats['l2'] = stats.lambda4/stats.heads
    ax = plot.subplot(fig=fig, grid=gs[0,0], func='sns.boxplot', data=stats, 
                      pf_x='livestock', pf_y='l2',
                      pf_color=plot.get_style('color',5),
                      xt_rotation=50, la_ylabel=r'$\lambda_5$/\#heads',
                      la_title='Farms to cells',
                      la_xlabel='')

    ax = plot.subplot(fig=fig, grid=gs[0,1], func='sns.boxplot', data=stats, 
                      pf_x='livestock', pf_y='corr',
                      pf_color=plot.get_style('color',5),
                      xt_rotation=50, la_title=r'Correlation with GLW',
                      la_xlabel='',
                      la_ylabel=r'Corr. coeff. $\rho$')
    plot.savefig('livestock_farms_to_cells.pdf')

def main():

    # load data
    stats = pd.read_csv(
            '../../livestock/results/stats_farms_to_cells.csv.zip')
    stats = stats.sort_values('livestock')

    farms = pd.read_csv('../../livestock/results/farms_to_cells.csv.zip')
    farms.loc[farms.subtype=='pigeons and squab', 'subtype'] = 'pigeons'

    agcensus = pd.read_csv(
            '../../livestock/results/agcensus_filled_gaps.csv.zip')
    agcensus.loc[agcensus.subtype=='pigeons and squab', 'subtype'] = 'pigeons'

    #livestock_order = ['cattle', 'chickens', 'hogs', 'sheep']

    # heads
    heads(farms, agcensus)

    # farm generation
    gen_farms(stats, agcensus)

    # farms to cells
    farms_to_cells(stats)

    # farm sizes
    farm_size(farms, agcensus)
    return

if __name__ == '__main__':
    main()
