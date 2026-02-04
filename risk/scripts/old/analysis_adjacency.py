DESC='''
Poultry data analysis, model analysis, and risk maps. This script will generate
all plots for the paper.

AA
'''

import ast
import click
import datetime as dt
from dateutil.relativedelta import relativedelta
import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pdb import set_trace
import re
import sys
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns

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

from kbdata import loader
from kbutils.display import pf
from kbviz import plot

#from parlist import POULTRY_PARLIST as PARLIST

import utils

MAX_MILES = 300

CONDITIONAL_RISK_MODEL = '../intermediate_data/baseline_conditional_risk.pkl'

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

    df = pd.concat([pdf[['start_date', 'type']], 
                    ddf[['start_date', 'type']]]) 
    df.start_date = pd.to_datetime(df.start_date)
    df['yq'] = df.start_date.dt.to_period('Q').astype(str)  # e.g., '2023Q1'
    df = df.sort_values(by=['yq'])
    df.loc[df['type'].isnull(), 'type'] = 'other'

    bdf.collection_date = pd.to_datetime(bdf.collection_date)
    bdf['yq'] = bdf.collection_date.dt.to_period('Q').astype(str)  # e.g., '2023Q1'
    bdf = bdf.sort_values(by=['yq'])

    ## df['yq'] = df.yq.str[4:] + '\n' + df.yq.str[0:4]
    ## bdf['yq'] = bdf.yq.str[4:] + '\n' + bdf.yq.str[0:4]
    ## bdf['yq'] = bdf.year.astype(str) + ',' + bdf.quarter.astype(str)

    hue_order = df.type.drop_duplicates().sort_values().tolist()
    hue_order.remove('dairy')
    hue_order.append('dairy')

    fig = plot.Fig(x=10, y=5)
    fig.grid(nrows=4, ncols=1, hspace=.3, wspace=-.1)

    sp = plot.Subplot(fig=fig, row=slice(1,4), col=0)
    pe = plot.Histplot(subplot=sp, data=df, x='yq', multiple='stack', hue='type',
                       hue_order=hue_order)
    pe.legend(ncols=3, handlelength=1)
    sp.ylabel('Livestock')
    sp.yticks(prune='right')
    __, labels = sp.get_ticks(axis='x')
    
    ytrig = ''
    new_labels = []
    for lab in labels:
        if ytrig == lab[0:4]:
            new_labels.append(lab[4:])
        else:
            ytrig = lab[0:4]
            new_labels.append(lab[4:] + '\n' + lab[0:4])
    sp.xticks(labels=new_labels)

    sp = plot.Subplot(fig=fig, row=0, col=0, sharex=sp, ylim=(0,2500))
    pe = plot.Histplot(subplot=sp, data=bdf, x='yq', color=9)
    sp.ylabel('Wild birds', width_ratio=1.5, fontsize='small')
    sp.title('Distribution of host-specific incidence reports')
    fig.savefig('h5n1_incidence.pdf')

@cli.command()
def duration():
    ptdf = utils.h5n1_poultry()
    pdf = ptdf[ptdf.end_date!='-1'].copy()
    pdf.end_date = pd.to_datetime(pdf.end_date)

    dtdf = utils.h5n1_dairy()
    dtdf['type'] = 'dairy'
    ddf = dtdf[~dtdf.end_date.isnull()].copy()
    ddf.end_date = pd.to_datetime(ddf.end_date)

    df = pd.concat([pdf, ddf])
    df = df.reset_index(drop=True)
    df.start_date = pd.to_datetime(df.start_date)

    df['duration'] = (df.end_date - df.start_date).dt.days

    hue_order = df.type.drop_duplicates().sort_values().tolist()
    hue_order.remove('dairy')
    hue_order.append('dairy')

    # end_date reporting fraction
    edf = df.type.value_counts()
    tdf = pd.concat([ptdf.type.value_counts(), dtdf.type.value_counts()])
    cdf = pd.DataFrame({'end': edf, 'tot': tdf})
    cdf['frac'] = cdf.end / cdf.tot
    cdf = cdf.reset_index()

    fig = plot.Fig(x=5, y=5)
    fig.grid(nrows=5, ncols=1, hspace=.7, wspace=-.1)

    sp = plot.Subplot(fig=fig, row=slice(2,5), col=0, ylim=(0,100))
    pe = plot.Boxplot(subplot=sp, data=df, x='type', y='duration', order=hue_order,
                      color=1)
    sp.ylabel('Days')
    sp.xticks(rotation=30, labeldict=dict(ha='right', rotation_mode='anchor'))

    sp = plot.Subplot(fig=fig, row=slice(0,2), col=0, sharex=sp, ylim=(0,1))
    pe = plot.Barplot(subplot=sp, data=cdf, x='type', y='frac', order=hue_order)
    sp.ylabel('Frac. with end date', fontsize='small')
    sp.title('Event duration')

    fig.savefig('event_duration.pdf')
    return

@cli.command()
@click.option('--delta', default=15)
@click.option('--hop', default=0)
def outbreaks(delta, hop):

    preps = pd.read_csv('../results/poultry_outbreaks.csv').fillna(-1)
    preps = preps[preps.type!='all_poultry'].copy()
    ## preps.loc[preps.type=='all', 'type'] = 'all poultry'

    dreps = pd.read_csv('../results/dairy_outbreaks.csv').fillna(-1)
    dreps['type'] = 'dairy'
    dreps = utils.fit_central_valley(dreps)

    reps = pd.concat([preps, dreps])

    reps = reps[reps.delta==delta]
    reps = reps[['start_date', 'county_code', 'type', f'event{hop}']] 
    reps = reps.rename(columns={f'event{hop}': 'outbreak'})

    reps = reps[reps.type!='all'].copy()

    ## # get birds data
    ## bset = [f'birds{x}_W2' for x in range(1,13)]
    ## bpop = pd.read_parquet('livestock_birds_features.parquet', 
    ##         columns=['x', 'y', 'county_code', 'subtype'] + bset)
    ## bpop = pd.melt(bpop, id_vars=['x', 'y', 'county_code', 'subtype'],
    ##                var_name='month', value_name='bpop')
    ## bpop.month = bpop.month.str[5:].str[:-3].astype('int')
    ## h5p = pd.read_parquet(
    ##         '../../data/birds_prevalence/bird_h5_prevalence.parquet')
    ## bpop = bpop.merge(h5p, on=['x', 'y', 'month'], how='left').fillna(0)
    ## bpop.bpop = bpop.bpop * bpop.value
    ## cbpop = bpop.groupby(['county_code', 'subtype', 'month'],
    ##                      as_index=False)['bpop'].sum()
    ## cbpop = cbpop.rename(columns={'subtype': 'type'})
    ## cbpop.loc[cbpop.type=='milk', 'type'] = 'dairy'
    ## cbpop_dairy = utils.fit_central_valley(cbpop[cbpop.type=='dairy'])
    ## cbpop = pd.concat([cbpop[cbpop.type!='dairy'], cbpop_dairy])

    reps.start_date = pd.to_datetime(reps.start_date)
    reps['month'] = reps.start_date.dt.month
    ## reps = reps.merge(cbpop, on=['county_code', 'type', 'month'], 
    ##                   how='left')
    reps = reps.sort_values('start_date')

    sdf = reps.groupby(['outbreak', 'type']).agg(
            county_code=('county_code', 'first'), 
            osize=('county_code', 'count'),
            start_date=('start_date', 'first'),
            end_date=('start_date', 'last')).reset_index()
            ## bpop=('bpop', 'mean')).reset_index()
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

    farms.loc[farms.subtype=='milk', 'subtype'] = 'dairy cattle'
    farms = farms[['county_code', 'subtype', 'farms', 'heads']].rename(
            columns={'subtype': 'type'})

    sdf = sdf.merge(farms, on=['county_code', 'type'], how='left').fillna(0)

    postfix = f'{delta}_{hop}'

    types = ['ckn-broilers', 'ckn-layers', 'ckn-pullets',  
             'ducks', 'turkeys', 'dairy']

    # outbreaks vs. farms

    fig = plot.Fig(x=5, y=4, constrained_layout=True)
    sp = plot.Subplot(fig=fig, ylim=(10,100), xlim=(10,100),
                      yscale='log', xscale='log', square_cells=True)
    pe = plot.Scatterplot(subplot=sp, data=sdf[sdf.type.isin(types)], x='farms',
                          y='osize', hue='type', hue_order=types, style='type',
                          s=75)
    pe.legend(ncols=1, columnspacing=.3, handlelength=1, bbox_to_anchor=(1,.5), 
              loc='center left', handletextpad=.1)
    sp.ylabel(value='Cluster size')
    sp.xlabel(value='County farm count')
    #sp.yticks(prune='right')
    ## sp.title(value=fr'$\delta={delta}$, $h={hop}$')
    sp.title(rf'Cluster size with respect to farm counts $\delta={delta}$',
             width_ratio=1, fontsize='normalsize')
    fig.savefig(f'outbreaks_vs_farms_{postfix}.pdf')

    types = sdf.type.drop_duplicates().tolist()
    types.remove('dairy')
    types.append('dairy')
    sdf['tp'] = 2
    sdf.loc[sdf.type!='dairy', 'tp'] = 1
    sdf = sdf.sort_values(['tp', 'type'])

    # outbreak size frequency
    tdf = sdf[['type', 'osize']
              ].value_counts().reset_index().sort_values('osize').rename(
                      columns={'count': 'outbreaks'})
    tdf['cumsum'] = tdf.groupby('type')['outbreaks'].cumsum()
    tot = tdf.groupby('type', as_index=False)['cumsum'].agg('last').rename(
            columns={'cumsum': 'tot'})
    tdf = tdf.merge(tot, on='type')
    tdf['cumsum'] = tdf['cumsum'] / tdf.tot
    markers = ['o', '^', 's', 'o', '^', 's']

    fig = plot.Fig(x=5, y=4)
    sp = plot.Subplot(fig=fig, xscale='log', xlim=(10,100), ylim=(.4,1),
                      square_cells=True)
    pe = plot.Lineplot(subplot=sp, data=tdf, x='osize', y='cumsum',
                       hue='type', hue_order=types, style='type',
                       markersize=10)
    pe.legend(ncols=1, columnspacing=.3, handlelength=1, fontsize='small', 
              loc='center left', bbox_to_anchor=(1,.5))
    sp.ylabel(value='Frac. of outbreaks (cummul.)', width_ratio=1)
    sp.xlabel(value='Number of reports per outbreak')
    sp.yticks(ticks=np.round(np.arange(.4,1.2,.2),1), prune='right')
    sp.title(value=rf'Cummulative distribution of host-specific outbreak size $\delta={delta}$', 
             fontsize='normalsize')
    fig.savefig(f'outbreaks_size_{postfix}.pdf')

    fig = plot.Fig(x=4, y=4)
    sp = plot.Subplot(fig=fig, xscale='log', yscale='log', xlim=(10,100), 
                      ylim=(10,100))
    pe = plot.Scatterplot(subplot=sp, data=sdf[sdf.type.isin(types)], 
                          x='length', y='osize', hue='type', hue_order=types, 
                          style='type', s=75)
    pe.legend(ncols=1, handletextpad=.3, handlelength=1, fontsize='small',
              bbox_to_anchor=(1,.5), loc='center left')
    sp.ylabel(value=rf'Cluster size')
    sp.xlabel(value='Days')
    sp.title(value=rf'Cluster duration $\delta={delta}$', width_ratio=2)
    sp.yticks(prune='right')
    fig.savefig(f'outbreaks_length_{postfix}.pdf')

    ## # correlation with bird population
    ## types = ['ckn-broilers', 'ckn-layers', 'ckn-pullets',  
    ##          'ducks', 'turkeys', 'dairy']

    ## ax = plot.oneplot(fg_x=4, fg_y=4, 
    ##                   func='sns.scatterplot', 
    ##                   data=sdf[sdf.type.isin(types)], 
    ##                   pf_x='bpop', pf_y='osize',
    ##                   pf_hue='type', pf_hue_order=types,
    ##                   pf_style='type', pf_s=75,
    ##                   sp_yscale='log', sp_xscale='log',
    ##                   la_ylabel='Outbreak size',
    ##                   la_xlabel='Avg. estimated prevalence in birds',
    ##                   lg_ncol=1,
    ##                   lg_loc='center right', 
    ##                   lg_bbox_to_anchor=(1.5,0.5),
    ##                   lg_labelspacing=.1, lg_handletextpad=0,
    ##                   la_title=fr'$\delta={delta}$, $h={hop}$')
    ## plot.savefig(f'outbreaks_birds_{postfix}.pdf')

    ## sdf = sdf[sdf.county_code!=6999]
    ## corr = sdf[sdf.type.isin(types)].groupby('type').apply(
    ##         lambda x: spearmanr(x.osize, x.bpop))
    ## print('Spearman r:', corr)
    ## corr = sdf[sdf.type.isin(types)].groupby('type').apply(
    ##         lambda x: pearsonr(x.osize, x.bpop))
    ## print('Pearson r:', corr)

@cli.command()
@click.option('--delta', default=30)
@click.option('--hop', default=0)
@click.option('--startyear', default=2022)
@click.option('--endyear', default=2025)
def cross_species_outbreaks(delta, hop, startyear, endyear):

    df = pd.read_csv('../results/all_host_outbreaks.csv')
    df = df[df.delta==delta]
    df = df.rename(columns={f'event{hop}': 'outbreak'})
    df = df[(df.year>=startyear) & (df.year<endyear)].copy()

    # get farms
    dfarms = utils.load_dairy_farms(aggregate='county', commercial=True, 
                                    commercial_threshold=100)
    pfarms = utils.load_poultry_farms(commercial=True, aggregate='county',
                                      commercial_threshold=1000)
    farms = pd.concat([dfarms, pfarms])
    farms = utils.fit_central_valley(farms)
    farms = farms.groupby(['county_code', 'subtype']).agg(
        {'state_code': 'first', 'farms': 'sum', 'heads': 'sum'}
    ).reset_index()

    odf = df.groupby('outbreak')['type'].apply(
            lambda x: x.drop_duplicates().sort_values().tolist())
    
    odf = odf.apply(','.join)

    #types = df.type.drop_duplicates().sort_values().tolist()
    types = ['backyard', 'mixed', 'milk', 'turkeys', 'ckn-layers', 
             'ckn-broilers', 'ducks']

    colx = []
    coly = []
    count = []
    sof_list = [] # self
    pof_list = [] # positive
    nof_list = [] # negative
    for x in types:
        for y in types:
            colx.append(x)
            coly.append(y)
            if x == y: 
                pof_list.append(_outbreak_farms(odf, x, x, 'present', df,
                                                farms))
                count.append(odf[odf.str.contains(x)].shape[0])
            else:
                val = odf[(odf.str.contains(x)) & 
                          (odf.str.contains(y))].shape[0]
                count.append(val)

                if x in ['backyard', 'mixed'] or y in ['backyard', 'mixed']:
                    continue
                pof_list.append(_outbreak_farms(odf, x, y, 'present', df, farms))
                nof_list.append(_outbreak_farms(odf, x, y, 'absent', df, farms))

    pofdf = pd.concat(pof_list)
    nofdf = pd.concat(nof_list)

    tdf = pd.DataFrame({'x': colx, 'y': coly, 'outbreaks': count})

    # counts with annotation
    mdf_annot = tdf[['x', 'y', 'outbreaks']].pivot_table(
        index='x', columns='y', values='outbreaks')
    mdf_count = mdf_annot.copy()
    np.fill_diagonal(mdf_count.values, -1)  # for deciding color only

    # relative counts (annotation)
    mdf_rel_annot = mdf_annot.div(mdf_annot.values.diagonal(), axis=0)
    np.fill_diagonal(mdf_rel_annot.values, mdf_annot.values.diagonal())  

    np.fill_diagonal(mdf_rel_annot.values, 
                     mdf_rel_annot.values.diagonal()/odf.shape[0])
    mdf_rel_annot = mdf_rel_annot * 100  # convert to percentage

    mdf_rel_count = mdf_rel_annot.copy()
    np.fill_diagonal(mdf_rel_count.values, -1)  # for deciding color only
    mdf_rel_annot = mdf_rel_annot.map(lambda x: str(int(x)) if x == int(x) else str(round(x, 1)))

    if endyear == 2026:
        endmonth = '05'
    else:
        endmonth = '12'

    fig = plot.Fig(x=5, y=5)
    sp = plot.Subplot(fig=fig)
    pe = plot.Heatmap(subplot=sp, data=mdf_count, order=types, annot=mdf_annot, 
                      annot_kws={'color': 'black', 'fontsize': 'large'}, 
                      cmap='cbYlGnBu', cmap_kws={'set_under': 'white'},
                      transform='linear', lim=(0,100), linewidth=.5, fmt='.0f') 
    pe.legend(cbaxis=(.95,0.25,.05,.5), title=r'\#Outbreaks', prune='both')
    sp.xticks(rotation=90, labeldict=dict(ha='right', rotation_mode='anchor'), labelsize='normalsize')
    sp.yticks(rotation=-0, labeldict=dict(ha='right', rotation_mode='anchor'), labelsize='normalsize')
    sp.title(value=rf'Cooccurrence among different hosts {startyear}-01 to {endyear-1}-{endmonth}\\$\delta={delta}$, $a={hop}$, \#chains$={odf.shape[0]}$', 
             fontsize='normalsize')
    fig.savefig(f'cross_species_outbreaks_d{delta}_h{hop}_{startyear}-{endyear}.pdf')

    fig = plot.Fig(x=5, y=5)
    sp = plot.Subplot(fig=fig)
    pe = plot.Heatmap(subplot=sp, data=mdf_rel_count, order=types, 
                      annot=mdf_rel_annot, 
                      annot_kws={'color': 'black', 'fontsize': 
                                 'normalsize'}, 
                      cmap='cbYlGnBu', cmap_kws={'set_under': 'white'},
                      transform='linear', lim=(0,100), linewidth=.5, fmt='') 
    pe.legend(cbaxis=(.95,0.25,.05,.5), 
              title=r'\#Outbreaks (row normalized)', prune='both')
    sp.xticks(rotation=90, labeldict=dict(ha='right', rotation_mode='anchor'), labelsize='normalsize')
    sp.yticks(rotation=-0, labeldict=dict(ha='right', rotation_mode='anchor'), labelsize='normalsize')
    sp.title(value=rf'Relative cooccurrence among different hosts {startyear}-01 to {endyear-1}-{endmonth}\\$\delta={delta}$, $a={hop}$, \#clusters$={odf.shape[0]}$', 
             fontsize='normalsize')
    fig.savefig(f'cross_species_rel_outbreaks_d{delta}_h{hop}_{startyear}-{endyear}.pdf')

    # The relationship between spillovers and farm counts
    types = ['milk', 'turkeys', 'ckn-layers', 'ckn-broilers', 'ducks']
    num_types = len(types)
    fig = plot.Fig(x=num_types*3, y=num_types*3)
    fig.grid(nrows=num_types, ncols=num_types, hspace=.1, wspace=.1)

    pofdf['present'] = True
    nofdf['present'] = False
    ofdf = pd.concat([pofdf, nofdf])

    # Plotting the outbreaks and farms
    # Axis sharing leads to a particular order in which this needs to be plotted
    spd = {}
    n = len(types)
    # Order of plotting
    plot_order = [(n-1,0)] + [(n-1,x) for x in range(1,n)] \
        + [(x,0) for x in range(0,n-1)] \
            + [(x,y) for x in range(0,n-1) for y in range(1,n)]

    limval = 10^4
    for x,y in plot_order:
        print(x,y)
        if x == n-1 and y == 0:
            spd[(x,y)] = plot.Subplot(
                fig=fig, row=x, col=y, xlim=(1,limval), ylim=(1,limval), 
                xscale='log', yscale='log')
        elif x != n-1 and y == 0:
            spd[(x,y)] = plot.Subplot(
                fig=fig, row=x, col=y, xlim=(1,limval), ylim=(1,limval), 
                xscale='log', yscale='log', sharex=spd[(n-1,0)])
        elif x == n-1 and y != 0:
            spd[(x,y)] = plot.Subplot(
                fig=fig, row=x, col=y, xlim=(1,limval), ylim=(1,limval), 
                xscale='log', yscale='log', sharey=spd[(n-1,0)])
        else:
            spd[(x,y)] = plot.Subplot(
                fig=fig, row=x, col=y, xlim=(1,limval), ylim=(1,limval), 
                xscale='log', yscale='log', 
                sharex=spd[(n-1,y)], sharey=spd[(x,0)])
        pe = plot.Scatterplot(
            subplot=spd[(x,y)], 
            data=ofdf[(ofdf.h1==types[x]) & (ofdf.h2==types[y])],
            x='h2_farms', y='h1_farms', hue='present', hue_order=[True, False], 
            style='present', style_order=[True, False], s=100)
        if not x:
            spd[(x,y)].title(types[y])
        if not y:
            spd[(x,y)].ylabel(types[x], fontsize='large')
        spd[(x,y)].xticks(prune='left')
        spd[(x,y)].yticks(prune='left')
    pe.legend(title='Column host present', scope='figure',
              bbox_to_anchor=(0.5,0.1), loc="upper center", ncols=2,
              title_fontsize='Large', fontsize='large',
              handletextpad=.2)
    fig.title(value=rf'Relative cooccurrence among different hosts {startyear}-01 to {endyear-1}-{endmonth}\\$\delta={delta}$, $a={hop}$, \#clusters$={odf.shape[0]}$', 
             fontsize='Large')
    fig.savefig(f'cross_species_outbreaks_farms_d{delta}_h{hop}_{startyear}-{endyear}.pdf')


    return


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

    farms.loc[farms.subtype=='milk', 'subtype'] = 'dairy cattle'
    farms = farms[['county_code', 'subtype', 'farms', 'heads']].rename(
            columns={'subtype': 'type'})

    hdf = hdf.merge(farms, on=['county_code', 'type'])

    chs = hdf.groupby('type').apply(
            lambda x: spearmanr(x.heads, x.reports), include_groups=False)
    chs = pd.DataFrame(chs.tolist(), index=chs.index, 
                      columns=["correlation", "p_value"])
    chs.columns = [(r'Heads-Spear.', r'$\rho$'), (r'Heads-Spear.', '$p$')]

    chp = hdf.groupby('type').apply(
            lambda x: pearsonr(x.heads, x.reports), include_groups=False) 
    chp = pd.DataFrame(chp.tolist(), index=chp.index, 
                      columns=["correlation", "p_value"])
    chp.columns = [(r'Heads-Pear.', r'$\rho$'), (r'Heads-Pear.', '$p$')]

    cfs = hdf.groupby('type').apply(
            lambda x: spearmanr(x.farms, x.reports), include_groups=False)
    cfs = pd.DataFrame(cfs.tolist(), index=cfs.index, 
                      columns=["correlation", "p_value"])
    cfs.columns = [(r'Farms-Spear.', r'$\rho$'), (r'Farms-Spear.', '$p$')]

    cfp = hdf.groupby('type').apply(lambda x: pearsonr(x.farms, x.reports), include_groups=False) 
    cfp = pd.DataFrame(cfp.tolist(), index=cfp.index, 
                      columns=["correlation", "p_value"])
    cfp.columns = [(r'Farms-Pear.', r'$\rho$'), (r'Farms-Pear.', '$p$')]

    df = chs.join(chp)
    df = df.join(cfs)
    df = df.join(cfp)
    df = df.round(3)

    notes = pd.Series({
            'ckn-broilers': 'Incidences are mostly isolated events.',
            'ckn-layers': 'Incidences are mostly isolated events.',
            'ckn-pullets': 'Very few incidences.',
            'dairy': 'Central valley counties bunched together',
            'ducks': 'Many farm and head data missing.',
            'turkeys': 'Many farm and head data missing.',
             }).to_frame(name='Notes')

    df = df.join(notes)
    df = df.rename(columns={'Notes': ('Notes', '')})
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    df.style.format(precision=3).to_latex(
            buf='table_corr_reports_heads_farms.tex',
            clines='skip-last;data',
            multicol_align="c", 
            hrules=True)

@cli.command()
def adjacency():
    preps = pd.read_csv('../results/poultry_outbreaks.csv').fillna(-1)
    ## preps.loc[preps.type=='all', 'type'] = 'all poultry'
    preps = preps[preps.type!='all'].copy()

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

    fig = plot.Fig(x=5, y=4)
    sp = plot.Subplot(fig=fig, row=0, col=0)
    pe = plot.Barplot(subplot=sp, data=summ, x='type', y='rel', 
                      hue=r'$\delta$', order=types,
                      palette=[2, 3])
    sp.ylabel(value=r'$(\mathcal{O}_0-\mathcal{O}_1)/\mathcal{O}_0)\times100$')
    sp.title('Spatial extent of outbreaks')
    sp.xticks(rotation=30, labeldict=dict(ha='right', rotation_mode='anchor'))
    sp.yticks(prune='right')
    pe.legend(fontsize='small', title=r'$\delta$')
    fig.savefig('county_adjacency.pdf')
    return

def _outbreak_farms(odf, h1, h2, mode, df, farms):
    tdf = pd.DataFrame()
    if mode == 'present':
        tdf['outbreak'] = odf[(odf.str.contains(h1)) &
                             (odf.str.contains(h2))].index.tolist()
    elif mode == 'absent':
        tdf['outbreak'] = odf[(odf.str.contains(h1)) &
                             (~odf.str.contains(h2))].index.tolist()

    tdf['h1'] = h1
    tdf['h2'] = h2
    tdf = tdf.merge(df[['outbreak', 'county_code']], on='outbreak',
                    how='left').drop_duplicates()
    tdf = tdf.merge(
        farms[['county_code', 'subtype', 'farms']], 
        left_on=['county_code', 'h1'],
        right_on=['county_code', 'subtype'],
        how='left').fillna(0)
    tdf = tdf.rename(columns={'farms': 'h1_farms'})
    tdf = tdf.drop('subtype', axis=1)
    tdf = tdf.merge(
        farms[['county_code', 'subtype', 'farms']], 
        left_on=['county_code', 'h2'], 
        right_on=['county_code', 'subtype'],
        how='left').fillna(0)
    tdf = tdf.rename(columns={'farms': 'h2_farms'})
    tdf = tdf.drop('subtype', axis=1)
    return tdf

@cli.command()
@click.option('--cr', type=int)
def performance(cr):
    # download datasets
    rfname = f'../results/risk_scores_sp1_cr{cr}.csv.zip'
    sfname = f'../results/susceptibility_sp1_cr{cr}.csv.zip'

    # first, we do risk scores
    rs = pd.read_csv(rfname)
    rs = rs.drop('num_farms', axis=1)
    __, events, __, __ = risk.load_features()
    outprefix = re.sub('.csv.zip', '', os.path.basename(rfname))

    # set time range
    poultry_range = pd.date_range(start='2024-01', end='2025-12', freq="MS").strftime('%Y-%m').tolist()
    dairy_range = pd.date_range(start='2025-05', end='2025-12', freq="MS").strftime('%Y-%m').tolist()

    # collapse dataframes to prepare for ranking and plotting
    pdf = rs[(rs.subtype!='milk') & (rs.time.isin(poultry_range))]
    pdf = pd.melt(pdf, id_vars=['county_code', 'subtype', 'time', 'alpha'],
                  var_name='ahead', value_name='risk')
    ddf = rs[(rs.subtype=='milk') & (rs.time.isin(dairy_range))]
    ddf = pd.melt(ddf, id_vars=['county_code', 'subtype', 'time', 'alpha'],
                  var_name='ahead', value_name='risk')
    
    # rank across the time ranges per subtype
    # AA: to be written in the paper: sometimes, the risk scores are all
    ## zero for a large number of counties. We only get medium, high, very
    ## high ranks in those cases.
    labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']
    percentiles = [0, 50, 75, 90, 100]
    prap = pdf.groupby(['subtype', 'ahead', 'alpha'], as_index=False).apply(
            utils.percentile_categorize, timecol='time', riskcol='risk',
            start_time=poultry_range[0], end_time=poultry_range[-1],
            percentiles=percentiles, labels=labels, exclude_zeros=True)
    drap = ddf.groupby(['subtype', 'ahead', 'alpha'], as_index=False).apply(
            utils.percentile_categorize, timecol='time', riskcol='risk',
            start_time=dairy_range[0], end_time=dairy_range[-1],
            percentiles=percentiles, labels=labels, exclude_zeros=True)
    rap = pd.concat([prap, drap], ignore_index=False)
    rap.ahead = rap.ahead.astype('int')
    # convert time to datetime and add the number of months in the 'ahead' column
    rap['ym_ahead'] = (pd.to_datetime(rap['time']).dt.to_period('M') + 
                       rap['ahead']).dt.strftime('%Y-%m')
    rap.ahead = rap.ahead + 1

    # merge with ground truth 
    events['ym'] = events.start_date.dt.strftime('%Y-%m')
    events = events.rename(columns={'type': 'subtype'})

    # per subtype counts
    ### evaluation period
    pe = events[(events.subtype!='milk') & (events.ym.isin(poultry_range))]
    de = events[(events.subtype=='milk') & (events.ym.isin(dairy_range))]
    de = utils.fit_central_valley(de)
    events = pd.concat([pe, de])
    events = events.drop_duplicates()

    #subtypes = ['turkeys', 'ckn-layers', 'ckn-broilers', 'ducks', 'ckn-pullets', 'milk']
    subtypes = ['turkeys', 'ckn-layers', 'ckn-broilers', 'ducks', 'milk']
    events = events[events.subtype.isin(subtypes)]

    res = events.merge(rap, left_on=['county_code', 'subtype', 'ym'],
                       right_on=['county_code', 'subtype', 'ym_ahead'], how='left')
    if res.start_date.isnull().sum():
        raise ValueError('Found some NaNs in the merge of events and rap.')

    # plots
    print('Generating eval plots ...')
    for subtype in subtypes:
        for alpha in rs.alpha.drop_duplicates().tolist():
            print('Processing subtype:', subtype)
            tdf = res[(res.subtype==subtype) & (res.alpha==alpha)]
            counts = tdf.groupby(['ahead', 'rank_per_period']).size().reset_index(
                    name='count')

            # Calculate the total count per x value ('total_bill') to normalize
            total_counts = counts.groupby('ahead')['count'].transform('sum')

            # Create a new column for the percentage of each hue within each x value
            counts['percent'] = (counts['count'] / total_counts) * 100
            markers = ['o', '^', 's', 'o', 'x']
            hue_values = counts.rank_per_period.drop_duplicates().tolist()
            profiles = ['Very high', 'High', 'Medium', 'Low', 'Very low']
            hue_order = []
            current_palette = []
            for i,val in enumerate(profiles):
                if val in hue_values:
                    hue_order.append(val)
                    current_palette.append(i)
            fig = plot.Fig(x=4, y=3)
            sp = plot.Subplot(fig=fig, xlim=(1,counts.ahead.max()), ylim=(0,100),
                              square_cells=True)
            pe = plot.Lineplot(subplot=sp, data=counts, x='ahead', y='percent',
                               hue='rank_per_period', hue_order=hue_order,
                               style='rank_per_period', style_order=hue_order,
                               markers=True, markersize=10, palette=current_palette)
            sp.xlabel(value=r'$k$-step ahead forecast', fontsize='normalsize')
            sp.ylabel(value=r'Recall (\%)', fontsize='normalsize')
            if subtype == 'milk':
                subtype_title = 'dairy cattle'
            else:
                subtype_title = subtype
            sp.title(value=subtype_title + rf', $\alpha={alpha}$')
            pe.legend()
            fig.savefig(f'eval_{outprefix}_{subtype}_a{alpha}.pdf')

    # second, we do susceptibility scores
    ss = pd.read_csv(sfname)
    __, events, __, __ = risk.load_features()
    outprefix = re.sub('.csv', '', os.path.basename(rfname))

    # set time range
    poultry_range = pd.date_range(start='2024-01', end='2025-12', freq="MS").strftime('%Y-%m').tolist()
    dairy_range = pd.date_range(start='2025-05', end='2025-12', freq="MS").strftime('%Y-%m').tolist()

    # collapse dataframes to prepare for ranking and plotting
    pdf = ss[(ss.subtype!='milk') & (ss.time.isin(poultry_range))]
    pdf = pd.melt(pdf, id_vars=['county_code', 'subtype', 'time'],
                  var_name='ahead', value_name='risk')
    ddf = ss[(ss.subtype=='milk') & (ss.time.isin(dairy_range))]
    ddf = pd.melt(ddf, id_vars=['county_code', 'subtype', 'time'],
                  var_name='ahead', value_name='risk')
    
    # convert time to datetime and add the number of months in the 'ahead' column
    rap['ym_ahead'] = (pd.to_datetime(rap['time']).dt.to_period('M') + 
                       rap['ahead']).dt.strftime('%Y-%m')
    rap.ahead = rap.ahead + 1

    rap = pd.concat([pdf, ddf], ignore_index=False)
    rap.ahead = rap.ahead.astype('int')
    # convert time to datetime and add the number of months in the 'ahead' column
    rap['ym_ahead'] = (pd.to_datetime(rap['time']).dt.to_period('M') + 
                       rap['ahead']).dt.strftime('%Y-%m')
    rap.ahead = rap.ahead + 1

    # merge with ground truth 
    events['ym'] = events.start_date.dt.strftime('%Y-%m')
    events = events.rename(columns={'type': 'subtype'})

    res = events.merge(rap, left_on=['county_code', 'subtype', 'ym'],
                       right_on=['county_code', 'subtype', 'ym_ahead'], 
                       how='outer')
    res['present'] = 'absent'
    res.loc[~res.ym.isnull(), 'present'] = 'present'

    res['state'] = res.county_code//1000

    if cr:
        model = 'comp'
    else:
        model = 'cond'

    #subtypes = ['turkeys', 'ckn-layers', 'ckn-broilers', 'ducks', 'ckn-pullets', 'milk']
    subtypes = ['turkeys', 'ckn-layers', 'ckn-broilers', 'ducks', 'milk']

    # AA
    # In res, expect to find NaNs for two reasons:
    # 1. incidence reported for outside the eval period
    # 2. incidence reported for types for which risk was not generated

    # plots
    print('Generating risk value plots ...')
    for subtype in subtypes:
        print('Processing subtype:', subtype)
        fig = plot.Fig(x=4, y=3)
        tdf = res[res.subtype==subtype]
        sp = plot.Subplot(fig=fig, ylim=(0,None))
        pe = plot.Boxplot(subplot=sp, data=tdf, 
                          x='ahead', y='risk', hue='present')
        sp.xlabel(value=r'$k$-step ahead forecast', fontsize='large')
        sp.ylabel(value=rf'Incidence likelihood $P_\text{{{model}}}$', fontsize='large', width_ratio=1.3)
        if subtype == 'milk':
            subtype_title = 'dairy cattle'
        else:
            subtype_title = subtype
        sp.title(value=subtype_title)
        pe.legend()
        fig.savefig(f'values_{outprefix}_{subtype}.png', dpi=200)

    # without fliers
    print('Generating risk value plots without outliers ...')
    for subtype in subtypes:
        print('Processing subtype:', subtype)
        fig = plot.Fig(x=4, y=3)
        tdf = res[res.subtype==subtype]
        sp = plot.Subplot(fig=fig, ylim=(0,None))
        pe = plot.Boxplot(subplot=sp, data=tdf, 
                          x='ahead', y='risk', hue='present', showfliers=False)
        sp.xlabel(value=r'$k$-step ahead forecast', fontsize='large')
        sp.ylabel(value=rf'Incidence likelihood $P_\text{{{model}}}$', fontsize='large', width_ratio=1.3)
        if subtype == 'milk':
            subtype_title = 'dairy cattle'
        else:
            subtype_title = subtype
        sp.title(value=subtype_title)
        pe.legend()
        fig.savefig(f'values-wo-outliers_{outprefix}_{subtype}.pdf')

    # False positives
    print('Generating false positives plots ...')
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

    tdf = res[['state', 'county_code', 'subtype', 'ahead', 'time', 
               'rank_across_periods', 'present']].drop_duplicates()
    tdf.present = tdf.present.map({'present': True, 'absent': False})
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
    dairy_regions = utils.fit_central_valley(regions)
    regions = pd.concat([regions, dairy_regions[dairy_regions.county_code==6999]
                         ]).reset_index(drop=True) 
    states = states.reset_index()

    ## Reloading events
    __, events, __, __ = risk.load_features()
    pe = events[events.type!='milk']
    de = events[events.type=='milk']
    de = utils.fit_central_valley(de)
    events = events.rename(columns={'type': 'subtype'})
    events['state'] = events.county_code // 1000

    for subtype in subtypes:
        print('False positives:', subtype)
        states_present = events[events.subtype==subtype].state.drop_duplicates().tolist()
        tdf = cdf[(cdf.subtype==subtype) & (cdf.ahead==1) &
                  (~cdf.state.isin(states_present)) &
                  (cdf.rank_across_periods=='Very high') &
                  (cdf['count']>=3)].drop_duplicates()
        states_high_risk = tdf.state.drop_duplicates().tolist()
        if subtype == 'milk':
            set_trace()
        gcdf = regions.merge(tdf, on='county_code')
    
        fp_states = states[(~states.state_code.isin(states_present)) & 
                           (states.state_code.isin(states_high_risk))]

        fig = plot.Fig(x=5, y=4, constrained_layout=True)
        sp = plot.Subplot(fig=fig, projection='gcrs.WebMercator()')
        poly = plot.Polyplot(subplot=sp, data=states)
        fp_poly = plot.Polyplot(subplot=sp, data=fp_states, facecolor='#dddddd', 
                                label='No reports')
        farm_threshold = 5
        chor = plot.Choropleth(subplot=sp, data=gcdf[gcdf.county_farms>=farm_threshold], hue='county_farms', lim=(farm_threshold,None))
        chor.legend(labelsize='tiny', cbaxis=(.87,.18,.02,.3),
                    title=r'\#Farms', title_fontsize='small')
        fp_poly.legend(bbox_to_anchor=(0,0), loc='lower left', title_fontsize='small', 
                       handlelength=1)
        if subtype == 'milk':
            subtype_title = 'dairy cattle'
        else:
            subtype_title = subtype
        sp.title(value=subtype_title)
        fig.savefig(f'false-pos_{outprefix}_{subtype}.pdf')

@cli.command()
def risk_maps():
    # load risk maps
    rs = pd.read_csv('../results/risk_scores_sp1_cr1.csv')

    pdf = rs[rs.subtype!='milk']
    pdf = pd.melt(pdf, id_vars=['county_code', 'subtype', 'time'],
                  var_name='ahead', value_name='risk')
    ddf = rs[rs.subtype=='milk']
    ddf = pd.melt(ddf, id_vars=['county_code', 'subtype', 'time'],
                  var_name='ahead', value_name='risk')
    ddf = utils.fit_central_valley(ddf)
    ddf = utils.combine_probs(ddf, id_vars=
                              ['county_code', 'subtype', 'time', 'ahead'])

    df = pd.concat([pdf, ddf])
    df = df[df.time==df.time.max()].copy()
    df.ahead = df.ahead.astype(int)

    # shapes
    regions, states = utils.load_shapes()
    dairy_regions = utils.fit_central_valley(regions)
    regions = pd.concat([regions, dairy_regions[dairy_regions.county_code==6999]
                         ]).reset_index(drop=True) 

    gdf = regions.merge(df, on='county_code', how='left')
    gdf = regions[['county_code', 'geometry']].merge(df, on=['county_code'], 
                                                     how='left')
    gdf.loc[gdf.risk.isnull(), 'risk'] = 10**-10 
    gdf.loc[gdf.risk==0, 'risk'] = 10**-10 

    gdf.risk = np.log10(gdf.risk)

    types = ['milk', 'turkeys', 'ckn-layers', 'ckn-broilers']
    months = 6
    fig = plot.Fig(x=5*len(types), y=3*months, constrained_layout=True)
    fig.grid(nrows=months, ncols=len(types), hspace=-.8, wspace=0)
    vmin = gdf.risk.min()
    vmax = gdf.risk.max()
    for month in range(1,months+1):
        for livestock in types:
            print(livestock,month)
            tdf = gdf[(gdf.subtype==livestock) & (gdf.ahead==month)]
            if livestock != 'milk':
                tdf = tdf[tdf.county_code!=6999]
            sp = plot.Subplot(fig=fig, projection='gcrs.WebMercator()')
            #poly = plot.Polyplot(subplot=sp, data=states)
            chor = plot.Choropleth(subplot=sp, data=tdf, hue='risk', 
                                   lim=(vmin,vmax), cmap='cbYlOrRd')
            if month == 1:
                sp.title(value=livestock, fontsize='huge')
            if livestock == types[0]:
                sp.ylabel(value=month, fontsize='huge')
            else:
                ylabel = ''
    chor.legend_data['labels'] = [rf'$10^{{{x}}}$' for x in 
                                  chor.legend_data['labels']]
    #chor.legend_data['ticks'][-1] = vmax
    #chor.legend_data['labels'][-1] = f'{vmax}'
    chor.legend(scope='figure', cbaxis=(1.03,1/8,0.01,.8), 
                labelsize='Large')
    fig.savefig('risk_maps.png', dpi=200)

@cli.command()
def select_county_risk():
    # just for reference
    counties = loader.load('usa_counties', contiguous_us=True)
    __, events, __, __ = risk.load_features()
    events = utils.fit_central_valley(events)
    events['ym'] = events.start_date.dt.strftime('%Y-%m')
    events = events.rename(columns={'type': 'subtype'})
    events = events[['county_code', 'subtype', 'ym']].drop_duplicates().reset_index(drop=True)

    preps = pd.read_csv('../results/poultry_outbreaks.csv').fillna(-1)
    xx = preps[preps.type=='backyard']

    select_counties = {}
    select_counties['milk'] = {'Weld, CO': 8123, 'Central Valley, CA': 6999, 
                       'Gooding, IA': 16047}
    select_counties['turkeys'] = {'Mercer, OH': 39107, 'Darke, OH': 39037,
                                  'Ottawa, MI': 26139}
    select_counties['backyard'] = {'Canyon, ID': 16027}
    
    # load both risks
    spdf = pd.read_csv('../results/risk_scores_sp1_cr0.csv')
    spdf = spdf[['county_code', 'time', 'subtype', '0']].rename(columns={'0': 'risk'})
    spdf['type'] = 'WB spillover'

    totdf = pd.read_csv('../results/risk_scores_sp1_cr1.csv')
    totdf = totdf[['county_code', 'time', 'subtype', '0', '1']]

    totdf1 = totdf[['county_code', 'time', 'subtype', '0']].rename(columns={'0': 'risk'})
    pcomp1 = r'$P_\text{comp}$ 1-month ahead'
    totdf1['type'] = pcomp1
    totdf2 = totdf[['county_code', 'time', 'subtype', '1']].rename(columns={'1': 'risk'})
    pcomp2 = r'$P_\text{comp}$ 2-month ahead'
    totdf2['type'] = pcomp2
    # add 1 to 'time' which is in 'YYYY-MM' format
    totdf2['time'] = (pd.to_datetime(totdf2['time']).dt.to_period('M') 
                      + 1).dt.strftime('%Y-%m')

    hue_order = [pcomp1, pcomp2, 'WB spillover']
    
    rdf = pd.concat([spdf, totdf1, totdf2], ignore_index=True)

    for subtype in ['milk', 'turkeys']:
        min_time = '2024-01'
        max_time = '2025-05'
        for county_name, county_code in select_counties[subtype].items():
            print(f'Processing {subtype} county: {county_name}')
            events_county = events[(events.county_code==county_code) & (events.subtype==subtype)].copy()
            rdf_county = rdf[(rdf.county_code==county_code) & (rdf.subtype==subtype) 
                             & (rdf.time >= min_time) & (rdf.time <= max_time)].copy()
            fig = plot.Fig(x=6, y=4)
            sp = plot.Subplot(fig=fig, xlim=(min_time, max_time),
                              ylim=(0, rdf_county.risk.max()), square_cells=True)
            pe = plot.Lineplot(subplot=sp, data=rdf_county, x='time', y='risk',
                               hue='type', hue_order=hue_order)
            sp.xlabel(value='Time')
            sp.ylabel(value='Incidence probability', width_ratio=.7)
            if subtype == 'milk':
                subtype_title = 'dairy'
            else:
                subtype_title = subtype
            sp.title(value=f'{subtype_title}: {county_name}')
            ticks, labels = sp.get_ticks('x')
            years = [x[:4] for x in labels]
            year_no_change = [i for i, x in enumerate(years) if x == years[i-1]]
            for i in range(len(labels)):
                if i in year_no_change:
                    labels[i] = labels[i][5:]
                else:
                    labels[i] = r'\parbox{1cm}{\centering ' + labels[i][5:] \
                        + r'\\' + labels[i][:4] + '}'

            sp.xticks(ticks=ticks, labels=labels, rotation=0, 
                      labeldict=dict(), labelsize='footnotesize')

            pe = plot.Scatterplot(subplot=sp, data=events_county, 
                                 x='ym', y=[.5]*events_county.shape[0],
                                 s=100, color=3, label='Outbreaks reported')
            # save legend data
            leg = pe.legend_data
            #pe.legend(fontsize='footnotesize')
            fig.savefig(f'risk_timeseries_{subtype}_{county_name.replace(", ","_").replace(" ","_")}.pdf'.lower())

    # create a separate legend figure
    plot.standalone_legend(leg['handles'], leg['labels'], 
                           'risk_timeseries_legend.pdf', ncols=4,
                           handlelength=1, columnspacing=.5)

def risk_persistence(rs, threshold=None):
    k = rs.time.drop_duplicates().shape[0]
    rs['persistence'] = rs.score >= threshold
    df = (rs.groupby('county_code')['persistence'].sum()/k).reset_index()
    df['risk_threshold'] = threshold
    return df

@cli.command()
def ww_h5():
    # model parameters
    delta = 30
    w_plus = 2
    w_minus = 2

    preps = pd.read_csv('../results/poultry_outbreaks.csv').fillna(-1)
    # preps.loc[preps.type=='all', 'type'] = 'all poultry'

    dreps = pd.read_csv('../results/dairy_outbreaks.csv').fillna(-1)
    dreps['type'] = 'dairy'

    reps = pd.concat([preps, dreps])
    reps['ym'] = reps.start_date.str[0:7]
    reps = reps.rename(columns={'type': 'subtype'})
    odf = reps.groupby(['county_code', 'subtype', 'delta', 'event0'])['ym'].agg(
            start='min', end='max', h5_instances='count').reset_index()
    odf = odf.rename(columns={'event0': 'event'})
    odf = odf[odf.delta==delta]
    odf.start = pd.to_datetime(odf.start)
    odf.end = pd.to_datetime(odf.end)

    # county neigbhors
    cn = pd.read_parquet('../intermediate_data/county_distances.parquet')
    cn = cn[['source', 'target', 'length', 'dist']][cn.length<=10].copy()

    # load geometries
    counties, states = utils.load_shapes()
    dcounties = utils.fit_central_valley(counties)

    # ww negatives: given negatives, are there correponding outbreaks?
    ### load ww monitoring
    ww = utils.wastewater()

    ### this part is for plotting surveillance sites
    wwcounties = counties.merge(ww.county_code.drop_duplicates(), how='right')
    wwcounties = wwcounties[~wwcounties.state_code.isnull()]

    ### select those which have never been positive
    wodf = ww.groupby('county_code').agg(
            {'time': 'min', 'present': 'max'}).reset_index()
    wodf.time = pd.to_datetime(wodf.time)

    ## odf['timediff'] = utils.timediff(odf.end, wodf.time.min())
    ## odf[odf.timediff>=-w_minus].copy()

    ### merge cn with h5
    wodf = wodf.merge(cn, left_on='county_code', right_on='source', how='left')

    ### merge ww and h5
    ### how='left' as the analysis is ww centric
    ### At this point,
    wwh5 = wodf.merge(odf, left_on='target', right_on='county_code', 
                      suffixes=('_ww', '_h5'), how='left')

    nwwh5 = wwh5[wwh5.present==0]
    nwwh5 = nwwh5[~nwwh5.event.isnull()]
    tot_neg_counties = nwwh5.county_code_ww.drop_duplicates().shape[0]

    ### Plot
    fig = plot.Fig(x=15, y=6.5, constrained_layout=True)
    fig.grid(nrows=2, ncols=3, hspace=-.5, wspace=0)
    i = 0
    for subtype in ['dairy', 'turkeys', 'ckn-layers', 'mixed', 'backyard', 'ckn-broilers']:
        nwwh5sub = nwwh5[nwwh5.subtype==subtype]
        if subtype == 'dairy':
            tdf = utils.fit_central_valley(nwwh5sub, county_code_col='target')
            nwwh5sub = tdf.groupby(['county_code_ww', 'target']).agg({
                'h5_instances': 'sum', 'length': 'min', 'dist': 'min'}).reset_index()

        tdf = nwwh5sub[nwwh5sub.dist<=MAX_MILES].groupby('county_code_ww').agg(
                            {'h5_instances': 'sum', 'length': 'min'}).reset_index()
        neg_counties = tdf.shape[0]
        cdf = counties.merge(tdf, left_on='county_code', right_on='county_code_ww')

        sp = plot.Subplot(fig=fig, projection='gcrs.WebMercator()')
        poly = plot.Polyplot(subplot=sp, data=states)
        chor = plot.Choropleth(subplot=sp, data=cdf, lim=(vmin,vmax), hue='h5_instances', cmap='cbYlOrRd')
        chor.legend(labelsize='tiny', title=r'\#H5 reports', 
                    title_fontsize='small', cbaxis=(.87,.15,.02,.3))
        surv = plot.Polyplot(subplot=sp, data=wwcounties, label='WW surv.', 
                             linewidth=1, edgecolor='black')
        surv.legend(handlelength=1, bbox_to_anchor=(0,0), loc='lower left') 
        sp.title(rf'{subtype}: {neg_counties}/{tot_neg_counties} ({100*neg_counties/tot_neg_counties: .1f}\%)')
        print(subtype, tdf.h5_instances.min(), tdf.h5_instances.max())
        i += 1
    fig.savefig('ww_neg_h5_maps.pdf')

    # ww positives: given positives, are there correponding outbreak?
    ### load ww coincident events
    ww = pd.read_csv('../results/ww_colocation_events.csv')
    set_trace()
    ww['ym'] = ww.time.str[0:7]
    wodf = ww.groupby(['county_code', 'delta', 'event'])['ym'].agg(
            start='min', end='max', ww_instances='count').reset_index()
    wodf = wodf[wodf.delta==delta]

    ### prepare before merging
    wodf.start = pd.to_datetime(wodf.start)
    wodf.end = pd.to_datetime(wodf.end)

    odf = odf[odf.delta==delta]
    odf.start = pd.to_datetime(odf.start)
    odf.end = pd.to_datetime(odf.end)
    odf['timediff'] = utils.timediff(odf.end, wodf.start.min())
    odf[odf.timediff>=-w_minus].copy()

    ### merge cn with h5
    wodf = wodf.merge(cn, left_on='county_code', right_on='source', how='left')

    # merge the two event clusters
    # how='left' as the analysis is ww centric
    wwh5 = wodf.merge(odf, left_on='target',right_on='county_code', 
                      suffixes=('_ww', '_h5'), how='left')

    ### Are there any ww events with no h5 match?
    tdf = wwh5[['event_ww', 'event_h5']].fillna(-1).groupby('event_h5').max()
    if not tdf.shape[0]:
        raise ValueError(
                'Found a ww incident that is no where near a h5 in space or time! Not really an error, but something has changed.')

    ### Time-based filtering
    wwh5 = wwh5[~wwh5.event_h5.isnull()]
    wwh5['h5s_wwe'] = utils.timediff(wwh5.start_h5, wwh5.end_ww)
    wwh5['wws_h5e'] = utils.timediff(wwh5.start_ww, wwh5.end_h5)
    pwwh5 = wwh5[~((wwh5.h5s_wwe>w_minus) | (wwh5.wws_h5e>w_plus))]
    pwwh5 = pwwh5[['event_ww', 'county_code_ww', 'subtype', 'ww_instances', 'h5_instances', 
                   'length', 'target', 'dist']].drop_duplicates()
    all_pos_counties = counties.merge(pwwh5.county_code_ww.drop_duplicates(),
                                      left_on='county_code',
                                      right_on='county_code_ww')
    tot_pos_counties = all_pos_counties.shape[0]

    ### Plot
    fig = plot.Fig(x=15, y=6.5, constrained_layout=True)
    fig.grid(nrows=2, ncols=3, hspace=-.5, wspace=0)
    for subtype in ['dairy', 'turkeys', 'ckn-layers', 'mixed', 'backyard', 'ckn-broilers']:
        pwwh5sub = pwwh5[pwwh5.subtype==subtype]
        if subtype == 'dairy':
            tdf = utils.fit_central_valley(pwwh5sub, county_code_col='target')
            pwwh5sub = tdf.groupby(['event_ww', 'county_code_ww', 'target']).agg({
                'h5_instances': 'sum', 'length': 'min', 'dist': 'min', 
                'ww_instances': 'first'}).reset_index()

        # Duplicates need to be removed as one county has been associated with two
        # WW events that are in turn associated with the same outbreak.
        tdf = pwwh5sub[pwwh5sub.dist<=MAX_MILES][
                ['county_code_ww', 'h5_instances', 'ww_instances']
                ].drop_duplicates().groupby('county_code_ww').agg(
                        {'h5_instances': 'sum', 'ww_instances': 'first'})
        pos_counties = tdf.shape[0]
        cdf = counties.merge(tdf, left_on='county_code', right_on='county_code_ww')
        cdf_ = all_pos_counties[~all_pos_counties.county_code.isin(cdf.county_code.tolist())]
        sp = plot.Subplot(fig=fig, projection='gcrs.WebMercator()')
        poly = plot.Polyplot(subplot=sp, data=states)
        chor = plot.Choropleth(subplot=sp, data=cdf, hue='h5_instances', cmap='cbYlGnBu')
        chor.legend(labelsize='tiny', title=r'\#H5 reports', 
                  title_fontsize='small', cbaxis=(.87,.15,.02,.3))
        noreports = plot.Polyplot(subplot=sp, data=cdf_, facecolor='red', 
                                  edgecolor='red', label='No reports')
        sp.title(rf'{subtype}: {pos_counties}/{tot_pos_counties} ({100*pos_counties/tot_pos_counties: .1f}\%)',
                 fontsize='normalsize')
        noreports.legend(handlelength=1, bbox_to_anchor=(0,.1), loc='lower left')
        surv = plot.Polyplot(subplot=sp, data=wwcounties, label='WW surv.', 
                             linewidth=1, edgecolor='black')
        surv.legend(handlelength=1, bbox_to_anchor=(0,0), loc='lower left') 
        print(subtype, tdf.h5_instances.min(), tdf.h5_instances.max())
        i += 1
    fig.savefig('ww_pos_h5_maps.pdf')

    ### Selected
    fig = plot.Fig(x=16, y=9, constrained_layout=True)
    fig.grid(nrows=2, ncols=2, hspace=-.1, wspace=.0)
    for subtype in ['dairy', 'turkeys', 'ckn-layers', 'ckn-broilers']:
        pwwh5sub = pwwh5[pwwh5.subtype==subtype]
        if subtype == 'dairy':
            tdf = utils.fit_central_valley(pwwh5sub, county_code_col='target')
            pwwh5sub = tdf.groupby(['event_ww', 'county_code_ww', 'target']).agg({
                'h5_instances': 'sum', 'length': 'min', 'dist': 'min', 
                'ww_instances': 'first'}).reset_index()

        # Duplicates need to be removed as one county has been associated with two
        # WW events that are in turn associated with the same outbreak.
        tdf = pwwh5sub[pwwh5sub.dist<=MAX_MILES][
                ['county_code_ww', 'h5_instances', 'ww_instances']
                ].drop_duplicates().groupby('county_code_ww').agg(
                        {'h5_instances': 'sum', 'ww_instances': 'first'})
        pos_counties = tdf.shape[0]
        cdf = counties.merge(tdf, left_on='county_code', right_on='county_code_ww')
        cdf_ = all_pos_counties[~all_pos_counties.county_code.isin(cdf.county_code.tolist())]
        sp = plot.Subplot(fig=fig, projection='gcrs.WebMercator()')
        poly = plot.Polyplot(subplot=sp, data=states)
        chor = plot.Choropleth(subplot=sp, data=cdf, hue='h5_instances', cmap='cbYlGnBu')
        chor.legend(labelsize='tiny', title=r'\#H5 reports', 
                  title_fontsize='small', cbaxis=(.9,.0,.02,.5))
        noreports = plot.Polyplot(subplot=sp, data=cdf_, facecolor='red', 
                                  edgecolor='red', label='No H5 reports')
        sp.title(rf'{subtype}: {pos_counties}/{tot_pos_counties} ({100*pos_counties/tot_pos_counties: .1f}\%)',
                 fontsize='large')
        noreports.legend(handlelength=1, bbox_to_anchor=(0,.1), loc='lower left')
        surv = plot.Polyplot(subplot=sp, data=wwcounties, label='WW surv.', 
                             linewidth=1, edgecolor='black')
        surv.legend(handlelength=1, bbox_to_anchor=(0,0), loc='lower left') 
        print(subtype, tdf.h5_instances.min(), tdf.h5_instances.max())
        i += 1
    fig.title(rf'(c) Wastewater sites matched to outbreaks $\le{MAX_MILES}$ miles', 
              fontsize='Large', y=1.05)
    fig.savefig('ww_pos_h5_maps_selected.pdf')

    # Results w.r.t. distances
    ### positives
    subtypes = ['dairy', 'turkeys', 'ckn-layers', 'mixed', 'backyard', 'ckn-broilers']
    pwwh5 = pwwh5.sort_values('dist')
    ll = []
    for subtype in subtypes:
        print(subtype)
        tdf = pwwh5[pwwh5.subtype==subtype].copy()
        if subtype == 'dairy':
            pwwh5sub = utils.fit_central_valley(tdf, county_code_col='target')
            tdf = pwwh5sub.groupby(['event_ww', 'county_code_ww', 'target']).agg({
                'h5_instances': 'sum', 'length': 'min', 'dist': 'min', 
                'ww_instances': 'first'}).reset_index()
        nunique = pd.Series({
            dist: tdf[tdf.dist<=dist]['county_code_ww'].nunique()
            for dist in tdf.dist.unique()}, name='hits').reset_index().rename(
                    columns={'index': 'dist'})
        nunique['subtype'] = subtype
        ll.append(nunique)

    pos = pd.concat(ll)
    pos['perc'] = pos.hits / tot_pos_counties * 100

    fig = plot.Fig(x=5, y=4)
    sp = plot.Subplot(fig=fig, square_cells=True, xlim=(0,500), ylim=(0,100))
    plot.Lineplot(subplot=sp, data=pos, x='dist', y='perc',
                  hue='subtype', hue_order=subtypes[::-1],
                  style='subtype', markers=False, dashes=True)
    lp = plot.Lineplot(subplot=sp, data=pos[pos.subtype=='dairy'], x='dist', y='perc',
                       color=5, linewidth=4)
    sp.xlabel(value='Distance (miles)')
    sp.ylabel(value=r'True positives (\%)')
    sp.title(value='(a) Wastewater positives matched with H5 outbreaks', fontsize='normalsize')
    lp.legend(bbox_to_anchor=(1,.5), loc='center left')

    fig.savefig('ww_pos_dist.pdf')

    ### negatives
    nwwh5 = nwwh5.sort_values('dist')
    ll = []
    for subtype in subtypes:
        print(subtype)
        nwwh5sub = nwwh5[nwwh5.subtype==subtype].copy()
        if subtype == 'dairy':
            tdf = utils.fit_central_valley(nwwh5sub, county_code_col='target')
            nwwh5sub = tdf.groupby(['county_code_ww', 'target']).agg({
                'h5_instances': 'sum', 'length': 'min', 'dist': 'min'}).reset_index()

        nunique = pd.Series({
            dist: nwwh5sub[nwwh5sub.dist<=dist]['county_code_ww'].nunique()
            for dist in tdf.dist.unique()}, name='hits').reset_index().rename(
                    columns={'index': 'dist'})
        nunique['subtype'] = subtype
        ll.append(nunique)
    neg = pd.concat(ll)
    neg['perc'] = neg.hits / tot_neg_counties * 100

    fig = plot.Fig(x=5, y=4)
    sp = plot.Subplot(fig=fig, square_cells=True, xlim=(0,500), ylim=(0,100))
    plot.Lineplot(subplot=sp, data=neg, x='dist', y='perc',
                  hue='subtype', hue_order=subtypes[::-1],
                  style='subtype', markers=False, dashes=True)
    lp = plot.Lineplot(subplot=sp, data=neg[neg.subtype=='dairy'], x='dist', y='perc',
                       color=5, linewidth=4)
    sp.xlabel(value='Distance (miles)')
    sp.ylabel(value=r'False negatives (\%)')
    sp.title(value='(b) Wastewater negatives matched with H5 outbreaks', fontsize='normalsize')
    lp.legend(bbox_to_anchor=(1,.5), loc='center left')
    fig.savefig('ww_neg_dist.pdf')
    return

    # Results w.r.t. hops
    ### positives
    subtypes = ['dairy', 'turkeys', 'ckn-layers', 'mixed', 'backyard', 'ckn-broilers']
    pwwh5 = pwwh5.rename(columns={'length': 'hops'})
    ll = []
    for hops in range(pwwh5.hops.max()+1):
        tdf = pwwh5[(pwwh5.hops<=hops) & (pwwh5.subtype.isin(subtypes))]
        ttdf = tdf.groupby('subtype').agg({
            'county_code_ww': 'nunique'}).reset_index()
        ttdf['hops'] = hops
        ll.append(ttdf)
    pos = pd.concat(ll)
    pos['perc'] = pos.county_code_ww / tot_pos_counties * 100

    fig = plot.Fig(x=5, y=4)
    sp = plot.Subplot(fig=fig, square_cells=True, xlim=(0,10), ylim=(0,100))
    plot.Lineplot(subplot=sp, data=pos, x='hops', y='perc',
                  hue='subtype', hue_order=subtypes[::-1],
                  style='subtype', markers=True, dashes=True)
    sp.xlabel(value='Hops')
    sp.ylabel(value=r'True positives (\%)')
    sp.title(value='Wastewater positives matched with H5 outbreaks', fontsize='normalsize')
    sp.legend()
    fig.savefig('ww_pos_hops.pdf')

    ### negatives
    subtypes = ['dairy', 'turkeys', 'ckn-layers', 'mixed', 'backyard', 'ckn-broilers']
    nwwh5 = nwwh5.rename(columns={'length': 'hops'})
    nwwh5.hops = nwwh5.hops.astype('int')
    ll = []
    for hops in range(nwwh5.hops.max()+1):
        tdf = nwwh5[(nwwh5.hops<=hops) & (nwwh5.subtype.isin(subtypes))]
        ttdf = tdf.groupby('subtype').agg({
            'county_code_ww': 'nunique'}).reset_index()
        ttdf['hops'] = hops
        ll.append(ttdf)
    neg = pd.concat(ll)
    neg['perc'] = neg.county_code_ww / tot_neg_counties * 100

    fig = plot.Fig(x=5, y=4)
    sp = plot.Subplot(fig=fig, square_cells=True, xlim=(0,10), ylim=(0,100))
    plot.Lineplot(subplot=sp, data=neg, x='hops', y='perc',
                  hue='subtype', hue_order=subtypes[::-1],
                  style='subtype', markers=True, dashes=True)
    sp.xlabel(value='Hops')
    sp.ylabel(value=r'False negatives (\%)')
    sp.title(value='Wastewater negatives matched with H5 outbreaks', fontsize='normalsize')
    sp.legend()
    fig.savefig('ww_neg_hops.pdf')

    ## ### Plot
    ## subtypes = ['dairy', 'turkeys', 'ckn-layers', 'ckn-broilers']
    ## tdf = pwwh5[pwwh5.subtype.isin(subtypes)]
    ## tdf = tdf.rename(columns={'length': 'hops'})
    ## sp = plot.Subplot(fig=fig)
    ## plot.Histplot(subplot=sp, data=tdf, x='hops', hue='subtype', multiple='stack',
    ##               hatch=True, discrete=True)

    ## fig = plot.Fig(x=5, y=4, constrained_layout=True)
    ## sp = plot.Subplot(fig=fig)
    ## plot.Histplot(subplot=sp, data=tdf, x='hops', hue='subtype', multiple='stack',
    ##               hatch=True, discrete=True)
    ## ## penguins = sns.load_dataset('penguins')
    ## ## plot.Histplot(subplot=sp, data=penguins, x='flipper_length_mm', hue='species', multiple='stack', hatch=True)

    ## sp.legend()
    ## sp.title('Pairwise distances between +ve wastewater sites and H5 events', 
    ##          fontsize='normalsize')
    ## sp.xlabel('Hops')
    ## fig.savefig('ww_pos_hops.pdf')

@cli.command()
def wwc():
    ww = pd.read_csv('../results/ww_colocation_events.csv')
    wcsize = ww.groupby('event')['present'].sum()
    fig = plot.Fig(x=5, y=4)
    sp = plot.Subplot(fig=fig, xlim=(0,None))
    pe = plot.Histplot(subplot=sp, data=wcsize)
    sp.title(rf'Wastewater chains length distribution (min $\ge{wcsize.min()}$)',
             fontsize='normalsize')
    sp.xlabel('Number of positive reports')
    sp.ylabel('Number of chains')
    sp.yticks(ticks=list(range(0,25,5)))
    fig.savefig('wc_chain_lengths.pdf')

@cli.command()
@click.option('--model', default='spillover')
def state_risk(model):
    if model == 'spillover':
        risk_file = '../results/risk_scores_sp1_cr0.csv'
    elif model == 'comprehensive':
        risk_file = '../results/risk_scores_sp1_cr1.csv'
    rs = pd.read_csv(risk_file)
    rs['state_code'] = rs.county_code // 1000
    rs = rs.drop('county_code', axis=1)

    df = utils.combine_probs(rs, 
                            id_vars=['state_code', 'time', 'subtype'])
    df.to_csv('state_' + model + '.csv', index=False)

if __name__ == '__main__':
    cli()

