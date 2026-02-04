DESC='''
Poultry data analysis, model analysis, and risk maps. This script will generate
all plots for the paper.

AA
'''

import ast
import click
import datetime as dt
import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pdb import set_trace
import re
import sys
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, roc_curve

import risk
import utils

try:
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt
except:
    pass

try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
except:
    pass

from aadata import loader
from aautils.display import pf
from aaviz import plot

from parlist import POULTRY_PARLIST as PARLIST

import utils

@click.group()
def cli():
    pass

# poultry incidence plots
@cli.command()
def incidence():
    pdf = utils.h5n1_poultry()
    ddf = utils.h5n1_dairy()
    ddf['type'] = 'dairy'
    bdf = utils.h5n1_birds()
    bdf['type'] = 'birds'

    df = pd.concat([pdf[['year', 'quarter', 'type']], 
                    ddf[['year', 'quarter', 'type']],]) 

    df['yq'] = df.year.astype(str) + ',' + df.quarter.astype(str)
    df = df.sort_values(by=['yq'])
    bdf['yq'] = bdf.year.astype(str) + ',' + bdf.quarter.astype(str)
    bdf = bdf.sort_values(by=['yq'])

    df.loc[df['type'].isnull(), 'type'] = 'other'
    hue_order = df.type.drop_duplicates().sort_values().tolist()
    hue_order.remove('dairy')
    hue_order.append('dairy')

    fig, gs = plot.initiate_figure(x=10, y=5, 
                                   gs_nrows=4, gs_ncols=1,
                                   gs_wspace=-.1, gs_hspace=.2,
                                   color='tableau10')
    ax = plot.subplot(fig=fig, grid=gs[1:,0],
                      func='sns.histplot', 
                      data=df[(df.year>=2022) & (df.year<=2024)], 
                      pf_x='yq', pf_multiple='stack',
                      pf_hue='type', pf_hue_order=hue_order, la_title='', 
                      la_ylabel='Livestock', la_xlabel='',
                      lg_visible=True,
                      lg_columnspacing=1, lg_ncol=3,
                      xt_rotation=50,
                      lg_title=False, fs_legend='small')
    ax = plot.subplot(fig=fig, grid=gs[0,0],
                      func='sns.histplot', 
                      data=bdf[(bdf.year>=2022) & (bdf.year<=2024)], 
                      sp_sharex=0,
                      pf_x='yq', la_title='', 
                      pf_color=plot.get_style('color', 9),
                      la_xlabel='', la_ylabel='Wild birds',
                      lg_title=False)
    plot.savefig('h5n1_incidence.pdf')

@cli.command()
def duration():
    pdf = utils.h5n1_poultry()
    ddf = utils.h5n1_dairy()
    ddf['type'] = 'dairy'

    df = pd.concat([pdf[pdf.end_date!='-1'], ddf[ddf.end_date!='-1']])
    df = df.reset_index(drop=True)
    df.end_date = pd.to_datetime(df.end_date)
    df.start_date = pd.to_datetime(df.start_date)

    df['duration'] = (df.end_date - df.start_date).dt.days

    hue_order = df.type.drop_duplicates().sort_values().tolist()
    hue_order.remove('dairy')
    hue_order.append('dairy')

    # end_date reporting fraction
    edf = df.type.value_counts()
    tdf = pd.concat([pdf.type.value_counts(), ddf.type.value_counts()])
    cdf = pd.DataFrame({'end': edf, 'tot': tdf})
    cdf['frac'] = cdf.end / cdf.tot
    cdf = cdf.reset_index()

    fig, gs = plot.initiate_figure(x=5, y=5, 
                                   gs_nrows=5, gs_ncols=1,
                                   gs_wspace=-.1, gs_hspace=.7,
                                   color='tableau10')
    ax = plot.subplot(fig=fig, grid=gs[2:,0],
                      func='sns.boxplot', data=df, 
                      pf_y='duration', pf_x='type', 
                      pf_order=hue_order,
                      pf_color=plot.get_style('color',1),
                      sp_ylim=(0,100),
                      la_title='', la_xlabel='', la_ylabel='Event duration',
                      xt_rotation=50,
                      lg_title=False)
    ax = plot.subplot(fig=fig, grid=gs[0:2,0],
                      func='sns.barplot', data=cdf,
                      pf_order=hue_order,
                      sp_sharex=0,
                      pf_x='index', pf_y='frac',
                      la_ylabel='\\parbox{3cm}{\\center Frac. with end date}',
                      la_xlabel='',
                      lg_title=False, fs_legend='small')
    plot.savefig('event_duration.pdf')

@cli.command()
@click.option('--delta', default=15)
@click.option('--hop', default=0)
def outbreaks(delta, hop):

    preps = pd.read_csv('../results/poultry_outbreaks.csv').fillna(-1)
    preps.loc[preps.type=='all', 'type'] = 'all poultry'

    dreps = pd.read_csv('../results/dairy_outbreaks.csv').fillna(-1)
    dreps['type'] = 'dairy'
    dreps = utils.fit_central_valley(dreps)

    reps = pd.concat([preps, dreps])

    reps = reps[reps.delta==delta]
    reps = reps[['start_date', 'county_code', 'type', f'event{hop}']] 
    reps = reps.rename(columns={f'event{hop}': 'outbreak'})

    # get birds data
    bset = [f'birds{x}_W2' for x in range(1,13)]
    bpop = pd.read_parquet('livestock_birds_features.parquet', 
            columns=['x', 'y', 'county_code', 'subtype'] + bset)
    bpop = pd.melt(bpop, id_vars=['x', 'y', 'county_code', 'subtype'],
                   var_name='month', value_name='bpop')
    bpop.month = bpop.month.str[5:].str[:-3].astype('int')
    h5p = pd.read_parquet(
            '../../data/birds_prevalence/bird_h5_prevalence.parquet')
    bpop = bpop.merge(h5p, on=['x', 'y', 'month'], how='left').fillna(0)
    bpop.bpop = bpop.bpop * bpop.value
    cbpop = bpop.groupby(['county_code', 'subtype', 'month'],
                         as_index=False)['bpop'].sum()
    cbpop = cbpop.rename(columns={'subtype': 'type'})
    cbpop.loc[cbpop.type=='milk', 'type'] = 'dairy'
    cbpop_dairy = utils.fit_central_valley(cbpop[cbpop.type=='dairy'])
    cbpop = pd.concat([cbpop[cbpop.type!='dairy'], cbpop_dairy])

    reps.start_date = pd.to_datetime(reps.start_date)
    reps['month'] = reps.start_date.dt.month
    reps = reps.merge(cbpop, on=['county_code', 'type', 'month'], 
                      how='left')
    reps = reps.sort_values('start_date')

    sdf = reps.groupby(['outbreak', 'type']).agg(
            county_code=('county_code', 'first'), 
            osize=('county_code', 'count'),
            start_date=('start_date', 'first'),
            end_date=('start_date', 'last'),
            bpop=('bpop', 'mean')).reset_index()
    sdf.end_date = pd.to_datetime(sdf.end_date)
    sdf['length'] = (sdf.end_date-sdf.start_date).dt.days

    # get farms
    dfarms = utils.load_dairy_farms(aggregate='county', commercial=True,
                                     commercial_threshold=100)
    dfarms = utils.fit_central_valley(dfarms)
    dfarms = dfarms.groupby(['state_code', 'county_code', 'subtype'],
                            as_index=False).sum()
    farms = pd.concat([
        utils.load_poultry_farms(commercial=True, aggregate='county',
                                 commercial_threshold=1000),
        dfarms])

    farms.loc[farms.subtype=='milk', 'subtype'] = 'dairy'
    farms = farms[['county_code', 'subtype', 'farms', 'heads']].rename(
            columns={'subtype': 'type'})

    sdf = sdf.merge(farms, on=['county_code', 'type'], how='left').fillna(0)

    postfix = f'{delta}_{hop}'

    types = ['ckn-broilers', 'ckn-layers', 'ckn-pullets',  
             'ducks', 'turkeys', 'dairy']

    ax = plot.oneplot(fg_x=5, fg_y=4, 
                      func='sns.scatterplot', 
                      data=sdf[sdf.type.isin(types)], 
                      pf_x='farms', pf_y='osize',
                      pf_hue='type', pf_hue_order=types,
                      pf_style='type', pf_s=75,
                      sp_ylim=(1,'default'), sp_xlim=(1,'default'),
                      sp_yscale='log', sp_xscale='log',
                      lg_ncol=2, lg_columnspacing=.3,
                      lg_handlelength=1,
                      la_ylabel='Outbreak size', 
                      la_xlabel='County farm count',
                      la_title=fr'$\delta={delta}$, $h={hop}$')
    plot.savefig(f'outbreaks_vs_farms_{postfix}.pdf')

    types = sdf.type.drop_duplicates().tolist()
    types.remove('dairy')
    types.append('dairy')
    sdf['tp'] = 2
    sdf.loc[sdf.type!='dairy', 'tp'] = 1
    sdf = sdf.sort_values(['tp', 'type'])

    # outbreak size frequency
    tdf = sdf[['type', 'osize']
              ].value_counts().reset_index().sort_values('osize').rename(
                      columns={0: 'outbreaks'})
    tdf['cumsum'] = tdf.groupby('type')['outbreaks'].cumsum()
    tot = tdf.groupby('type', as_index=False)['cumsum'].agg('last').rename(
            columns={'cumsum': 'tot'})
    tdf = tdf.merge(tot, on='type')
    tdf['cumsum'] = tdf['cumsum'] / tdf.tot

    ax = plot.oneplot(fg_x=5, fg_y=4, 
                      func='sns.lineplot', 
                      data=tdf,
                      pf_x='osize', pf_y='cumsum',
                      pf_hue='type', pf_hue_order=types,
                      pf_style='type', pf_markers=True, pf_markersize=10,
                      sp_xscale='log',
                      la_xlabel='Outbreak size', 
                      la_ylabel='Frac. of outbreaks (cummul.)', 
                      lg_ncol=2, lg_columnspacing=.3,
                      lg_handlelength=1,
                      la_title=fr'$\delta={delta}$, $h={hop}$')
    ## ax.set_xticks([np.linspace(2, 9), 
    ##                np.linspace(20, 90, 10),
    ##                np.linspace(200, 900, 10),
    ##                ], minor=True)
    plot.savefig(f'outbreaks_size_{postfix}.pdf')

    ax = plot.oneplot(fg_x=4, fg_y=4, 
                      func='sns.scatterplot', 
                      data=sdf[sdf.type.isin(types)], 
                      pf_x='length', pf_y='osize',
                      pf_hue='type', pf_hue_order=types,
                      pf_style='type', pf_s=75,
                      sp_yscale='log', sp_xscale='log',
                      la_ylabel='Outbreak size',
                      la_xlabel='Outbreak length (days)',
                      lg_ncol=1,
                      lg_loc='center right', 
                      lg_bbox_to_anchor=(1.5,0.5),
                      lg_labelspacing=.1, lg_handletextpad=0,
                      la_title=fr'$\delta={delta}$, $h={hop}$')
    plot.savefig(f'outbreaks_length_{postfix}.pdf')
                      # sp_yscale='log',
                      # sp_ylim=(1,'default'), sp_xlim=(1,'default'),

    # correlation with bird population
    types = ['ckn-broilers', 'ckn-layers', 'ckn-pullets',  
             'ducks', 'turkeys', 'dairy']

    ax = plot.oneplot(fg_x=4, fg_y=4, 
                      func='sns.scatterplot', 
                      data=sdf[sdf.type.isin(types)], 
                      pf_x='bpop', pf_y='osize',
                      pf_hue='type', pf_hue_order=types,
                      pf_style='type', pf_s=75,
                      sp_yscale='log', sp_xscale='log',
                      la_ylabel='Outbreak size',
                      la_xlabel='Avg. estimated prevalence in birds',
                      lg_ncol=1,
                      lg_loc='center right', 
                      lg_bbox_to_anchor=(1.5,0.5),
                      lg_labelspacing=.1, lg_handletextpad=0,
                      la_title=fr'$\delta={delta}$, $h={hop}$')
    plot.savefig(f'outbreaks_birds_{postfix}.pdf')

    sdf = sdf[sdf.county_code!=6999]
    corr = sdf[sdf.type.isin(types)].groupby('type').apply(
            lambda x: spearmanr(x.osize, x.bpop))
    print('Spearman r:', corr)
    corr = sdf[sdf.type.isin(types)].groupby('type').apply(
            lambda x: pearsonr(x.osize, x.bpop))
    print('Pearson r:', corr)

@cli.command()
def ranks():
    pdf = utils.h5n1_poultry(agg_by='county')
    ddf = utils.h5n1_dairy(agg_by='county')
    ddf['type'] = 'dairy'
    ddf = utils.fit_central_valley(ddf)
    hdf = pd.concat([pdf, ddf])
    hdf = hdf[~hdf.type.isin(['mixed', 'backyard'])].copy()

    dfarms = utils.load_dairy_farms(aggregate='county', commercial=True,
                                     commercial_threshold=100)
    dfarms = utils.fit_central_valley(dfarms)
    dfarms = dfarms.groupby(['state_code', 'county_code', 'subtype'],
                            as_index=False).sum()
    farms = pd.concat([
        utils.load_poultry_farms(commercial=True, aggregate='county',
                                 commercial_threshold=1000),
        dfarms])

    farms.loc[farms.subtype=='milk', 'subtype'] = 'dairy'
    farms = farms[['county_code', 'subtype', 'farms', 'heads']].rename(
            columns={'subtype': 'type'})

    hdf = hdf.merge(farms, on=['county_code', 'type'])

    chs = hdf.groupby('type').apply(lambda x: spearmanr(x.heads, x.reports))
    chs = pd.DataFrame(chs.tolist(), index=chs.index, 
                      columns=["correlation", "p_value"])
    chs.columns = pd.MultiIndex.from_tuples(
            [(r'Heads-Spear.', r'$\rho$'), (r'Heads-Spear.', '$p$')])

    chp = hdf.groupby('type').apply(lambda x: pearsonr(x.heads, x.reports)) 
    chp = pd.DataFrame(chp.tolist(), index=chp.index, 
                      columns=["correlation", "p_value"])
    chp.columns = pd.MultiIndex.from_tuples(
            [(r'Heads-Pear.', r'$\rho$'), (r'Heads-Pear.', '$p$')])

    cfs = hdf.groupby('type').apply(lambda x: spearmanr(x.farms, x.reports))
    cfs = pd.DataFrame(cfs.tolist(), index=cfs.index, 
                      columns=["correlation", "p_value"])
    cfs.columns = pd.MultiIndex.from_tuples(
            [(r'Farms-Spear.', r'$\rho$'), (r'Farms-Spear.', '$p$')])

    cfp = hdf.groupby('type').apply(lambda x: pearsonr(x.farms, x.reports)) 
    cfp = pd.DataFrame(cfp.tolist(), index=cfp.index, 
                      columns=["correlation", "p_value"])
    cfp.columns = pd.MultiIndex.from_tuples(
            [(r'Farms-Pear.', r'$\rho$'), (r'Farms-Pear.', '$p$')])

    df = chs.join(chp)
    df = df.join(cfs)
    df = df.join(cfp)
    df = df.round(3)

    notes = {
            'ckn-broilers': 'Incidences are mostly isolated events.',
            'ckn-layers': 'Incidences are mostly isolated events.',
            'ckn-pullets': 'Very few incidences.',
            'dairy': 'Central valley counties bunched together',
            'ducks': 'Many farm and head data missing.',
            'turkeys': 'Many farm and head data missing.',
             }
    df = df.join(pd.Series(notes, name='Notes'))
    col = df.columns.tolist()
    col.remove('Notes')
    col.append(('Notes', ''))
    df.columns = pd.MultiIndex.from_tuples(col)

    df.style.format(precision=3).to_latex(
            buf='table_corr_reports_heads_farms.tex',
            clines='skip-last;data',
            multicol_align="c", 
            hrules=True)

@cli.command()
def adjacency():
    preps = pd.read_csv('../results/poultry_outbreaks.csv').fillna(-1)
    preps.loc[preps.type=='all', 'type'] = 'all poultry'

    dreps = pd.read_csv('../results/dairy_outbreaks.csv').fillna(-1)
    dreps['type'] = 'dairy'
    dreps = utils.fit_central_valley(dreps)

    reps = pd.concat([preps, dreps])
    reps = reps[reps.delta!=60]


    tdf = reps[['type', 'delta', 'event0', 'event1']]

    e0 = reps.groupby(['type', 'delta'], as_index=False)['event0'].value_counts()
    e1 = reps.groupby(['type', 'delta'], as_index=False)['event1'].value_counts()

    tdf = tdf.merge(e0, on=['type', 'delta', 'event0']).rename(columns={'count': 'e0'})
    tdf = tdf.merge(e1, on=['type', 'delta', 'event1']).rename(columns={'count': 'e1'})

    summ0 = tdf.groupby(['type', 'delta'], as_index=False).agg({'event0': 'nunique'})
    summ1 = tdf.groupby(['type', 'delta'], as_index=False).agg({'event1': 'nunique'})

    summ = summ0.merge(summ1, on=['type', 'delta'])
    summ['rel'] = (summ.event0 - summ.event1) / summ.event0 * 100
    summ = summ.rename(columns={'delta': r'$\delta$'})

    types = summ.type.drop_duplicates().tolist()
    types.remove('dairy')
    types.append('dairy')

    ax = plot.oneplot(fg_x=5, fg_y=4, 
                      func='sns.barplot', fg_color='tableau10',
                      data=summ,
                      pf_x='type', pf_y='rel',
                      pf_hue=r'$\delta$', pf_order=types,
                      pf_palette=[plot.get_style('color', 2), plot.get_style('color', 3)],
                      la_ylabel=r'$(\mathcal{O}_0-\mathcal{O}_1)/\mathcal{O}_0)\times100$',
                      la_xlabel='', lg_title=r'$\delta$',
                      xt_rotation=40,
                      fs_legend='small')
    plot.savefig('county_adjacency.pdf')

    ## summ = summ.rename(columns={'event0': r'$h=0$', 'event1': r'$h=1$'})

    ## summ.style.format(precision=3).to_latex(
    ##         buf='table_outbreaks_adjacency.tex',
    ##         clines='skip-last;data',
    ##         multicol_align="c", 
    ##         hrules=True)

@cli.command()
def risk_maps():
    # load risk maps
    df = pd.read_csv('../results/risk_scores_sp1_cr1.csv')

    # combine poultry
    df.loc[df.subtype.isin(['ckn-broilers', 'ckn-layers', 'ckn-pullets', 
                            'turkeys', 'ducks']), 'subtype'] = 'poultry'
    df = risk.combine_probs(df, id_vars=['county_code', 'subtype'])
    livestock_list = df.subtype.drop_duplicates().tolist()

    # time range
    months = pd.date_range(start='2024-11', end='2025-02', freq="MS"
                           ).strftime('%Y-%m').tolist()

    # rank
    pdf = risk.percentile_categorize(
            df, cols=months,
            percentiles=[0, 50, 75, 90, 95, 100],
            labels=['Very low', 'Low', 'Medium', 'High', 'Very high'])
    
    # set color
    colors = plot.COLORS['cbYlOrRd']
    colormap = {
            "Very high": colors[7],
            "High": colors[6],
            "Medium": colors[4],
            "Low": colors[2],
            'Very low': colors[0]
            }

    # load maps
    regions, states = utils.load_shapes()
    # qmap = {1: 'Jan-Mar', 2: 'Apr-Jun', 3: 'Jul-Sep', 4: 'Oct-Dec'}

    # AA: will need central valley adjustment
    fig, gs = plot.initiate_figure(x=5*4, y=4*3, 
                                   gs_nrows=3, gs_ncols=4,
                                   gs_wspace=-.1, gs_hspace=-.5,
                                   color='tableau10',
                                   scilimits=[-2,2])
    i = 0
    for livestock in livestock_list:
        print(livestock)
        j = 0
        for month in months:
            print(i,j)
            if i == 2:
                xlabel = month
            else:
                xlabel = ''
            if j == 0:
                ylabel = livestock
            else:
                ylabel = ''

            tdf = pdf[pdf.subtype==livestock][['county_code', 'subtype', month]]
            tdf = tdf.rename(columns={month: 'risk'})
            tdf.risk = tdf.risk.astype(str)
            gdf = regions[['county_code', 'geometry']
                          ].merge(tdf, on=['county_code'], how='left')
            gdf.loc[gdf.risk.isnull(), 'risk'] = 'Very low'

            ax = plot.subplot(fig=fig, grid=gs[i,j], 
                              func='gpd.plot',
                              pf_facecolor='white', pf_edgecolor='grey',
                              pf_linewidth=.1, data=states) 
            ax = plot.subplot(fig=fig, ax=ax, grid=gs[i,j], func='gpd.plot',
                              data=gdf, 
                              pf_color=gdf.risk.map(colormap),
                              la_xlabel=xlabel, 
                              la_ylabel=ylabel, fs_xlabel='large')
            j += 1
        i += 1
    legend_elements = [Patch(facecolor=colormap[key], label=key)
                       for key in colormap]
    ax.legend(handles=legend_elements, 
              loc="lower right", bbox_to_anchor=(0.25,-.17), fontsize=14, 
              title_fontsize=10)
    plot.savefig(f'risk_maps.pdf')

@cli.command()
@click.option('--cr', type=int)
def performance(cr):
    # download datasets
    fname = f'../results/risk_scores_sp1_cr{cr}.csv'
    rs = pd.read_csv(fname)
    __, events, __, __ = risk.load_features()
    outprefix = re.sub('.csv', '', os.path.basename(fname))

    # set time range
    poultry_range = pd.date_range(start='2023-01', end='2024-12', freq="MS").strftime('%Y-%m').tolist()
    dairy_range = pd.date_range(start='2024-09', end='2024-12', freq="MS").strftime('%Y-%m').tolist()

    pdf = rs[(rs.subtype!='milk') & (rs.time.isin(poultry_range))]
    pdf = pd.melt(pdf, id_vars=['county_code', 'subtype', 'time'],
                  var_name='ahead', value_name='risk')
    ddf = rs[(rs.subtype=='milk') & (rs.time.isin(dairy_range))]
    ddf = pd.melt(ddf, id_vars=['county_code', 'subtype', 'time'],
                  var_name='ahead', value_name='risk')

    # rank across the time ranges per subtype
    labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']
    percentiles = [0, 50, 75, 90, 95, 100]
    prap = pdf.groupby(['subtype', 'ahead'], as_index=False, group_keys=True).apply(
            utils.percentile_categorize, timecol='time', riskcol='risk',
            start_time=poultry_range[0], end_time=poultry_range[-1],
            percentiles=percentiles, labels=labels)
    drap = ddf.groupby(['subtype', 'ahead'], as_index=False, group_keys=True).apply(
            utils.percentile_categorize, timecol='time', riskcol='risk',
            start_time=dairy_range[0], end_time=dairy_range[-1],
            percentiles=percentiles, labels=labels)
    rap = pd.concat([prap, drap], ignore_index=False)
    rap.ahead = rap.ahead.astype('int')

    # merge with ground truth 
    events['ym'] = events.start_date.str[0:7]
    events = events.rename(columns={'type': 'subtype'})
    events_all = events.copy()
    events_all.subtype = 'all'
    events = pd.concat([events, events_all])

    # per subtype counts
    ### evaluation period
    pe = events[(events.subtype!='milk') & (events.ym.isin(poultry_range))]
    de = events[(events.subtype=='milk') & (events.ym.isin(dairy_range))]
    events = pd.concat([pe, de])

    subtypes = ['turkeys', 'ckn-layers', 'ckn-broilers', 'ducks', 'ckn-pullets', 'milk']
    events = events[events.subtype.isin(subtypes)]

    res = events.merge(rap, left_on=['county_code', 'subtype', 'ym'],
                       right_on=['county_code', 'subtype', 'time'], how='left')
    if res.start_date.isnull().sum():
        raise ValueError('Found some NaNs in the merge of events and rap.')

    # plots
    for subtype in subtypes:
        ax = plot.oneplot(fg_x=5, fg_y=4, 
                          func='sns.histplot', data=res[res.subtype==subtype], 
                          pf_x='ahead', pf_hue='rank_across_periods', 
                          pf_hue_order=['Very high', 'High', 'Medium', 'Low', 'Very low'],
                          pf_multiple='dodge',
                          la_xlabel=r'$k$-step ahead forecast', 
                          la_ylabel='', la_title=subtype, lg_title='')
        plot.savefig(f'eval_{outprefix}_{subtype}.pdf')
        plt.clf()

    res = events.merge(rap, left_on=['county_code', 'subtype', 'ym'],
                       right_on=['county_code', 'subtype', 'time'], 
                       how='outer')
    res['present'] = False
    res.loc[~res.ym.isnull(), 'present'] = True

    # get farms
    dfarms = utils.load_dairy_farms(aggregate='county', commercial=True,
                                     commercial_threshold=100)
    dfarms = utils.fit_central_valley(dfarms)
    dfarms = dfarms.groupby(['state_code', 'county_code', 'subtype'],
                            as_index=False).sum()
    farms = pd.concat([
        utils.load_poultry_farms(commercial=True, aggregate='county',
                                 commercial_threshold=1000), dfarms])

    sfarms = farms.groupby(['state_code', 'subtype'], 
                        as_index=False)['farms'].sum() 
    sfarms = sfarms.rename(columns={'state_code': 'state'})

    res['state'] = res.county_code//1000

    # plots
    for subtype in subtypes:
        ax = plot.oneplot(fg_x=5, fg_y=4, 
                          func='sns.boxplot', 
                          data=res[res.subtype==subtype], 
                          pf_x='ahead', pf_y='risk', pf_hue='present', 
                          la_xlabel=r'$k$-step ahead forecast', 
                          la_ylabel='', la_title=subtype, lg_title='')
        plot.savefig(f'values_{outprefix}_{subtype}.pdf')
        plt.clf()
    # without fliers
    for subtype in subtypes:
        ax = plot.oneplot(fg_x=5, fg_y=4, 
                          func='sns.boxplot', 
                          data=res[res.subtype==subtype], 
                          pf_x='ahead', pf_y='risk', pf_hue='present', 
                          pf_showfliers=False,
                          la_xlabel=r'$k$-step ahead forecast', 
                          la_ylabel='', la_title=subtype, lg_title='')
        plot.savefig(f'values-wo-outliers_{outprefix}_{subtype}.pdf')
        plt.clf()

    # False positives
    tdf = res[['state', 'county_code', 'subtype', 'ahead', 'time', 
               'rank_across_periods', 'present']].drop_duplicates()
    tdf = tdf[(tdf.time>='2024-07')]

    ### Check if incidence ever happened
    cpres = tdf.groupby(['state', 'county_code', 'subtype', 'ahead'], 
                        as_index=False)['present'].max()
    ### Compute counties persistent risk counts
    cpers = tdf[['state', 'county_code', 'time', 'subtype', 'ahead', 
                 'rank_across_periods']].drop_duplicates().drop(
                      'time', axis=1).value_counts().reset_index()
    cdf = cpres.merge(cpers, on=['state', 'county_code', 'subtype', 'ahead'])
    ### Merge with farms
    cdf = cdf.merge(farms, on=['county_code', 'subtype'], how='left').fillna(0)
    cdf = cdf.rename(columns={'farms': 'county_farms', 'heads': 'county_heads'})
    
    ### Now, do the same at the state level
    spres = tdf.groupby(['state', 'subtype', 'ahead'], as_index=False)[
             'present'].max()
    spers = tdf[['state', 'time', 'subtype', 'ahead', 'rank_across_periods']
             ].drop_duplicates().drop('time', axis=1).value_counts().reset_index()
    sdf = spres.merge(spers, on=['state', 'subtype', 'ahead'])
    sdf = sdf.merge(sfarms, on=['state', 'subtype'], how='left').fillna(0)

    ### Merge county and state
    sdf = sdf[['state', 'subtype', 'ahead', 'present', 'farms']].rename(columns={
        'farms': 'state_farms', 'present': 'state_present'})
    cdf = cdf.merge(sdf, on=['state', 'subtype', 'ahead'])

    ### Merge will have more rows due to risk profile duplicates
    regions, states = utils.load_shapes()
    events['state'] = events.county_code // 1000
    states = states.reset_index()
    for subtype in subtypes:
        print(subtype)
        states_present = events[events.subtype==subtype].state.drop_duplicates().tolist()
        tdf = cdf[(cdf.subtype==subtype) & (cdf.ahead==1) &
                  (~cdf.state.isin(states_present)) &
                  (cdf.rank_across_periods=='Very high') &
                  (cdf['count']>=3)].drop_duplicates()
        states_high_risk = tdf.state.drop_duplicates().tolist()
        gcdf = regions.merge(tdf, on='county_code')
    
        states['fp'] = '#ffffff'
        states.loc[(~states.state_code.isin(states_present)) & 
                   (states.state_code.isin(states_high_risk)), 'fp'] = '#dddddd'

        fig, gs = plot.initiate_figure(x=5, y=4, 
                                       gs_nrows=1, gs_ncols=1,
                                       gs_wspace=-.1, gs_hspace=-.5,
                                       color='tableau10',
                                       scilimits=[-2,2])
        ax = plot.subplot(fig=fig, grid=gs[0,0], 
                          func='gpd.plot',
                          pf_color=states.fp, data=states) 
        ax = plot.subplot(fig=fig, ax=ax,
                          func='gpd.plot',
                          pf_facecolor='none', pf_edgecolor='black',
                          la_title=subtype,
                          pf_linewidth=.1, data=states)
        ax = plot.subplot(fig=fig, ax=ax, func='gpd.plot',
                          data=gcdf, 
                          pf_column='county_farms', pf_legend=True, 
                          pf_legend_kwds={'shrink': 0.4, 'pad': -.02})
        plot.savefig(f'false-pos_{outprefix}_{subtype}.pdf')
        plt.clf()
        del ax, fig

    ## ctdf = czz[(czz.state_present==False) & (czz.ahead==1) & 
    ##            (czz.subtype=='milk') & (czz.rank_across_periods=='Very high')
    ##            & (czz['count']>=3)].copy()
    ## tdf = szz[(szz.present==False) & (szz.ahead==1) & (szz.subtype=='milk')].copy()

def risk_persistence(rs, threshold=None):
    k = rs.time.drop_duplicates().shape[0]
    rs['persistence'] = rs.score >= threshold
    df = (rs.groupby('county_code')['persistence'].sum()/k).reset_index()
    df['risk_threshold'] = threshold
    return df

@cli.command()
def wastewater():
    # pending load farms
    # load risk maps
    rs = pd.read_csv('../results/risk_scores_sp1_cr1.csv')
    tdf = utils.combine_probs(rs.drop('subtype', axis=1), 
                              id_vars=['time', 'county_code'])
    tdf['subtype'] = 'all'
    rs = pd.concat([rs, tdf])

    # load events
    __, events, __, __ = risk.load_features()

    # load ww
    ww = utils.wastewater()

    # load county neighbors
    cn = pd.read_parquet('../intermediate_data/county_hops.parquet',
                         columns=['source', 'target', 'length'])

    # evaluation period
    eval_period = ['2024-04', '2025-01']

    # get prevalence for the eval period
    ww = ww[(ww.month>=eval_period[0]) & (ww.month<=eval_period[1])].copy()



    set_trace()

    # merge persistence with ground truth
    pdf = pdf.reset_index().merge(ww, on='county_code', how='left').fillna(0)

    df = pdf[(pdf.subtype=='all') & (pdf.risk_threshold==0.1)]

    # do roc stuff
    score = pdf.groupby(['subtype', 'risk_threshold']).apply(
            lambda x: roc_auc_score(x.present, x.persistence))

    # county neighbors
    cn = pd.read_parquet('county_neighbors.parquet')
    cn['county_x'] = cn.statefp_x*1000 + cn.countyfp_x
    cn['county_y'] = cn.statefp_y*1000 + cn.countyfp_y
    cn = cn[['county_x', 'county_y']]

    pdf = pdf.drop('present', axis=1)

    cnpdf = cn.merge(pdf, left_on='county_y', right_on='county_code', how='left')
    cnpdf = cnpdf[~cnpdf.county_code.isnull()]
    cndf = cnpdf.groupby(['subtype', 'risk_threshold', 'county_x']
                         )['persistence'].max().reset_index()
    cndf = cndf.merge(ww, left_on='county_x', right_on='county_code')

    df = cndf[(cndf.subtype=='all') & (cndf.risk_threshold==0.01)]
    set_trace()


    # do roc stuff
    score = cndf.groupby(['subtype', 'risk_threshold']).apply(
            lambda x: roc_auc_score(x.present, x.persistence))

    # rank rs by period and total
    labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']
    rrs = utils.percentile_categorize(
            rs, timecol='month', riskcol='risk',
            start_time=start_month, end_time=end_month,
            percentiles=[0, 50, 75, 90, 95, 100],
            labels=labels)
    
    # merge risks and ww
    rrs = rrs.merge(ww, on=['month', 'county_code'], how='left').fillna(-1)
    rrs[rrs.present!=-1].to_csv(
            'ww_rank_across_periods_without_county_adjacency.csv', 
            index=False)
    set_trace()

    cndf.to_csv('ww_rank_across_periods_with_county_adjacency.csv', index=False)

    df = rrs[rrs.present==1].rank_across_periods.value_counts() \
            / rrs[rrs.present==1].count().values[0] * 100
    df = df.reset_index()
    ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
                      func='sns.barplot', data=df, 
                      pf_y='rank_across_periods', pf_x='index', 
                      pf_order=labels,
                      pf_color=2,
                      la_title='County-based evaluation', la_xlabel='', 
                      la_ylabel=r'\% +ve reports',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('ww_county_pos.pdf')
    df = rrs[rrs.present==0].rank_across_periods.value_counts() \
            / rrs[rrs.present==0].count().values[0] * 100
    df = df.reset_index()
    ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
                      func='sns.barplot', data=df, 
                      pf_y='rank_across_periods', pf_x='index', 
                      pf_order=labels,
                      pf_color=4,
                      la_title='County-based evaluation', la_xlabel='', 
                      la_ylabel=r'\% -ve reports',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('ww_county_neg.pdf')

    df = cndf[cndf.present==1].rank_across_periods.value_counts() \
            / cndf[cndf.present==1].count().values[0] * 100
    df = df.reset_index()
    ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
                      func='sns.barplot', data=df, 
                      pf_y='rank_across_periods', pf_x='index', 
                      pf_order=labels,
                      pf_color=2,
                      la_title='Adj. County-based evaluation', la_xlabel='', 
                      la_ylabel=r'\% +ve reports',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('ww_adj_county_pos.pdf')
    df = cndf[cndf.present==0].rank_across_periods.value_counts() \
            / cndf[cndf.present==0].count().values[0] * 100
    df = df.reset_index()
    ax = plot.oneplot(fg_x=4, fg_y=2, fg_color='tableau10',
                      func='sns.barplot', data=df, 
                      pf_y='rank_across_periods', pf_x='index', 
                      pf_order=labels,
                      pf_color=4,
                      la_title='Adj. County-based evaluation', la_xlabel='', 
                      la_ylabel=r'\% -ve reports',
                      xt_rotation=50,
                      lg_title=False)
    plot.savefig('ww_adj_county_neg.pdf')

    # for choropleth
    start_month = '2024-11'
    end_month = '2025-01'
    tdf = rs[['county_code', 'month', 'risk']][(rs.month>=start_month) & 
                                               (rs.month<=end_month)].drop(
                                                       'month', axis=1)
    tdf = utils.combine_probs(tdf, id_vars='county_code')
    tdf['month'] = 0    # dummy
    rrs = utils.percentile_categorize(
            tdf, timecol='month', riskcol='risk',
            start_time=0, end_time=0,
            percentiles=[0, 50, 75, 90, 95, 100],
            labels=labels)

    # merge risks and ww
    wwa = ww[(ww.month>=start_month) & (ww.month<=end_month)].groupby(
            'county_code', as_index=False)['present'].max()
    rrs = rrs.merge(wwa, on='county_code', how='left').fillna(-1)

    df = rrs[['county_code', 'rank_across_periods']].copy()
    df.rank_across_periods = df.rank_across_periods.astype(str)
    regions = loader.load('usa_county_shapes', contiguous_us=True)
    states = regions[['state_code', 'geometry']].dissolve(by='state_code')
    regions = regions.merge(df, on='county_code', how='left').fillna('Very low')
    #regions.risk = np.log(regions.risk+1)
    colors = plot.COLORS['cbYlGnBu']
    colormap = {
            "Very high": colors[4],
            "High": colors[3],
            "Medium": colors[2],
            "Low": colors[1],
            'Very low': colors[0]
            }
    ww_ = ww.county_code.drop_duplicates().reset_index()
    rww = regions.merge(ww_, on='county_code')
    
    # plot
    fig, gs = plot.initiate_figure(x=10, y=8, 
                                   gs_nrows=1, gs_ncols=1,
                                   gs_wspace=.3, gs_hspace=.6,
                                   color='tableau10')
    ax = plot.subplot(fig=fig, grid=gs[0,0], func='gpd.plot',
                      data=regions, 
                      pf_color=regions.rank_across_periods.map(colormap),
                      la_title=f'Period: {start_month} to {end_month}',
                      fs_title='normalsize')
    ax = plot.subplot(fig=fig, ax=ax, grid=gs[0,0], 
                      func='gpd.plot',
                      pf_edgecolor='black', pf_facecolor='none',
                      pf_linewidth=.3, data=states) 

    legend_elements = [Patch(facecolor=colormap[key], label=key)
                       for key in colormap]
    ax.legend(handles=legend_elements, 
              loc="lower right", bbox_to_anchor=(1,.01), fontsize=15, 
              title_fontsize=10)
    ax = plot.subplot(fig=fig, ax=ax, grid=gs[:,:2], 
                      func='gpd.plot',
                      pf_edgecolor='red', pf_facecolor='none',
                      pf_linewidth=1, data=rww) 

    plot.savefig('ww_riskmap.pdf')


if __name__ == '__main__':
    cli()

