DESC = '''
Colocation risk for the digital similar paper.
'''

from aadata import loader
from aaviz import plot
import click
from matplotlib.patches import Patch, Circle, Rectangle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from pdb import set_trace
import utils

@click.group()
def cli():
    pass

@cli.command()
def risk():
    # load data
    farms = pd.read_csv('../../livestock/results/farms_to_cells.csv.zip')
    bdf = pd.read_parquet('bird_features.parquet')
    bh5 = pd.read_parquet('../../data/birds_prevalence/bird_h5_prevalence.parquet')

    # process
    farms.loc[(farms.livestock=='cattle') & (farms.subtype=='all'), 
              'subtype'] = 'cattle'
    fdf = farms[['x', 'y', 'state_code', 'county_code', 'subtype', 'heads']
              ].groupby(['x', 'y', 'state_code', 'county_code', 'subtype'], 
                        as_index=False).sum()
    fdf = fdf[fdf.subtype.isin(['cattle', 'turkeys', 'ckn-layers', 'milk'])].copy()
    fdf = fdf.rename(columns={'subtype': 'livestock'})

    bdf = bdf[['x', 'y'] + ['birds'+str(x) for x in range(1,13)]].melt(
            id_vars=['x', 'y'], var_name='month', value_name='abundance')
    bdf.month = bdf.month.str[5:].astype('int')

    # merge
    bdf = bdf.merge(bh5, on=['x', 'y', 'month'])
    bdf = bdf.rename(columns={'value': 'prevalence'})
    bdf['adj_abundance'] = bdf.abundance * bdf.prevalence
    bdf = bdf.drop(['prevalence', 'abundance'], axis=1)

    fdf = fdf.merge(bdf, on=['x', 'y'])

    # compute county-level risk
    fdf['risk'] = fdf.heads * fdf.adj_abundance
    fdf['quarter'] = fdf.month // 4 + 1
    cdf = fdf[['state_code', 'county_code', 'livestock', 
               'quarter', 'risk']].groupby(
                       ['state_code', 'county_code', 'livestock', 'quarter'],
                       as_index=False).sum()

    cdf = cdf.groupby(['livestock', 'quarter'], group_keys=True, 
                      as_index=False).apply(percentile)

    cdf.to_csv('colocation_risk.csv', index=False)
    return

def percentile(df):
    percentiles = [0, 75, 90, 95, 100]
    percentiles_labels = ['Low', 'Medium', 'High', 'Very high']
    bins = np.unique(np.percentile(df.risk, percentiles))
    labels = percentiles_labels[-len(bins)+1:]
    df['riskp'] = pd.cut(df.risk, bins=bins, labels=labels,
                          include_lowest=True)
    return df

def load_maps_data():
    df = pd.read_csv('../results/colocation_risk.csv')
    df.county_code = df.state_code*1000 + df.county_code
    regions = loader.load('usa_county_shapes', contiguous_us=True)
    states = regions[['state_code', 'geometry']].dissolve(by='state_code')
    return df, regions, states

@cli.command()
def quarterly_maps():
    df, regions, states = load_maps_data()
    livestock_list = df.livestock.drop_duplicates().tolist()
    colors = plot.COLORS['cbYlOrRd']
    colormap = {
            "Very high": colors[6],
            "High": colors[4],
            "Medium": colors[2],
            'Low': colors[0]
            }
    qmap = {1: 'Jan-Mar', 2: 'Apr-Jun', 3: 'Jul-Sep', 4: 'Oct-Dec'}
    for livestock in livestock_list:
        print(livestock)
        fig, gs = plot.initiate_figure(x=5*4, y=4, 
                                       gs_nrows=1, gs_ncols=4,
                                       gs_wspace=-.1, gs_hspace=.015,
                                       color='tableau10',
                                       scilimits=[-2,2])

        for quarter in [1,2,3,4]:
            tdf = df[(df.livestock==livestock) & (df.quarter==quarter)]
            gdf = regions[['county_code', 'geometry']
                          ].merge(tdf, on='county_code', how='left')
            gdf.loc[gdf.riskp.isnull(), 'riskp'] = 'Low'

            ax = plot.subplot(fig=fig, grid=gs[0,quarter-1], func='gpd.plot',
                              data=states, 
                              pf_facecolor=None,
                              pf_edgecolor='black', pf_linewidth=.5)
            ax = plot.subplot(fig=fig, ax=ax, grid=gs[0,quarter-1], func='gpd.plot',
                              data=gdf, 
                              pf_color=gdf.riskp.map(colormap),
                              pf_edgecolor=None,
                              la_xlabel=qmap[quarter], fs_xlabel='large')
        legend_elements = [Patch(facecolor=colormap[key], label=key) 
                           for key in colormap]
        ax.legend(handles=legend_elements, 
                  loc="lower right", bbox_to_anchor=(0.25,-.17), fontsize=14, title_fontsize=10)
        plot.savefig(f'colocation_risk_map_{livestock}.pdf')

@cli.command()
def time_agg_risk():
    df, regions, states = load_maps_data()

    h5p = utils.h5n1_poultry(agg_by='county')
    h5p = h5p.rename(columns={'type': 'livestock'})
    h5d = utils.h5n1_dairy(agg_by='county')
    h5d['livestock'] = 'milk'

    regionsp = regions.merge(h5p, on='county_code')
    regionsd = regions.merge(h5d, on='county_code')
    regionsd = utils.fit_central_valley(regionsd)
    
    h5 = pd.concat([regionsp, regionsd])
    #h5.geometry = h5.geometry.centroid

    livestock_list = df.livestock.drop_duplicates().tolist()

    for livestock in livestock_list:
        print(livestock)
        fig, gs = plot.initiate_figure(x=10, y=6, 
                                       gs_nrows=1, gs_ncols=4,
                                       gs_wspace=-.1, gs_hspace=.015,
                                       color='tableau10',
                                       scilimits=[-2,2])
        tdf = df[(df.livestock==livestock)].copy()
        h5s = h5[h5.livestock==livestock].copy()
        tdf['weight'] = tdf.riskp.map({'Very high': 1000, 'High': 100, 'Medium': 10, 'Low': 1})
        ttdf = tdf.groupby(['county_code', 'livestock'], 
                           as_index=False)['weight'].sum()

        gdf = regions[['county_code', 'geometry']
                      ].merge(ttdf, on='county_code', how='left')
        gdf.loc[gdf.weight.isnull(), 'weight'] = 1
        gdf.weight = np.log10(gdf.weight)
        
        ## ax = plot.subplot(fig=fig, grid=gs[0,0], 
        ##                   func='gpd.boundary.plot',
        ##                   pf_facecolor='white', pf_edgecolor='grey',
        ##                   pf_linewidth=.1, data=states) 
        ax = fig.add_subplot(gs[0,0])
        gdf.plot(ax=ax, column='weight', cmap='YlOrRd')
        ax.set_title(livestock, fontsize=7)
        ax.set_aspect('auto')
        plt.axis('tight')
        cax = fig.add_axes([0.301, 0.410, 0.005, 0.1])  # [left, bottom, width, height] in figure coordinates
        cbar = fig.colorbar(ax.collections[0], cax=cax, shrink=.1, 
                            ticks=[np.log10(4000),np.log10(400),np.log10(40),np.log10(4)])
        cbar.ax.tick_params(width=.2, length=.5)
        cbar.outline.set_visible(False)
        cbar.ax.set_yticklabels([r'Very high', r'High', r'Medium', r'Low'], fontsize=5)
        states.boundary.plot(ax=ax, edgecolor='grey', linewidth=.1)
        if livestock!='cattle':
            h5s.boundary.plot(ax=ax, edgecolor='blue', linewidth=.5)
        ax.set_axis_off()
        ## ax = plot.subplot(fig=fig, ax=ax, grid=gs[0,0], func='gpd.plot',
        ##                   pf_cmap='YlOrRd',
        ##                   data=gdf, pf_column='weight')
        ## ## if livestock!='cattle':
        ## ##     h5s.boundary.plot(ax=ax, edgecolor='blue', linewidth=.3)
        if livestock!='cattle':
            rect = Rectangle((2,3), 4,2, color='blue', label='Incidence')
            ax.legend(handles=[rect],
                      loc="lower right", bbox_to_anchor=(0.4,.1), 
                      fontsize=5, title_fontsize=10, handleheight=.1)
        plot.savefig(f'colocation_time_agg_risk_map_{livestock}.pdf')

@cli.command()
def peak_risk():
    df, regions, states = load_maps_data()
    livestock_list = df.livestock.drop_duplicates().tolist()
    pdf = df.loc[df.groupby(['state_code', 'county_code', 'livestock']
                            )['risk'].idxmax()].reset_index(drop=True) 
    colors = plot.COLORS['cbYlOrRd']
    colormap = {
            1: plot.get_style('color', 1),
            2: plot.get_style('color', 2),
            3: plot.get_style('color', 3),
            4: plot.get_style('color', 4)
            }
    qmap = {1: 'Jan-Mar', 2: 'Apr-Jun', 3: 'Jul-Sep', 4: 'Oct-Dec'}
    for livestock in livestock_list:
        print(livestock)
        fig, gs = plot.initiate_figure(x=10, y=6, 
                                       gs_nrows=1, gs_ncols=1,
                                       gs_wspace=-.1, gs_hspace=.015,
                                       color='tableau10',
                                       scilimits=[-2,2])
        tdf = pdf[(pdf.livestock==livestock)].copy()

        gdf = regions[['state_code', 'county_code', 'geometry']
                      ].merge(tdf, on=['state_code', 'county_code'])
        
        ## ax = plot.subplot(fig=fig, grid=gs[0,0], 
        ##                   func='gpd.boundary.plot',
        ##                   pf_facecolor='white', pf_edgecolor='grey',
        ##                   pf_linewidth=.1, data=states) 
        ax = plot.subplot(fig=fig, grid=gs[0,0], func='gpd.plot',
                          data=gdf, 
                          pf_color=gdf.quarter.map(colormap))
        ax.set_title(livestock, fontsize=25)
        legend_elements = [Patch(facecolor=colormap[key], label=qmap[key]) 
                           for key in colormap]
        ax.legend(handles=legend_elements, 
                  loc="lower right", bbox_to_anchor=(.95,.05), fontsize=14, title_fontsize=10)
        plot.savefig(f'colocation_time_peak_risk_map_{livestock}.pdf')

@cli.command()
def recall():
    df, regions, states = load_maps_data()

    h5p = utils.h5n1_poultry(agg_by='quarter')
    h5p = h5p.rename(columns={'type': 'livestock'})
    h5d = utils.h5n1_dairy(agg_by='quarter')
    h5d['livestock'] = 'milk'

    h5 = pd.concat([h5p, h5d])

    df = df.merge(h5, on=['county_code', 'livestock', 'quarter'], how='right').fillna('Low')

    livestock_list = ['milk', 'turkeys', 'ckn-layers']

    recall = df.groupby(['livestock', 'riskp'])['reports'].sum() / \
            df.groupby('livestock')['reports'].sum() * 100
    recall = recall.reset_index()
    order = ['Low', 'Medium', 'High', 'Very high']
    recall.riskp = pd.Categorical(recall.riskp, categories=order, ordered=True)
    recall = recall.sort_values('riskp')

    for livestock in livestock_list:
        print(livestock)
        ax = plot.oneplot(fg_x=3, fg_y=3, data=recall[recall.livestock==livestock],
                          fg_color='tableau10',
                          func='sns.barplot',
                          pf_x='riskp', pf_y='reports',
                          pf_color=plot.get_style('color', 3),
                          xt_rotation=40,
                          la_xlabel='', la_ylabel=r'\% reports', la_title=livestock,
                          fs_ylabel='large', fs_title='Large')
        plot.savefig(f'colocation_recall_{livestock}.pdf')

if __name__ == '__main__':
    cli()
