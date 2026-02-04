import geopandas as gpd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdb import set_trace

import aa_plot as plot
import utils
import seaborn as sns

if __name__ == '__main__':
    bdf = utils.h5n1_birds(agg_by='quarter')
    pdf = utils.h5n1_poultry(agg_by='quarter', commercial=False)
    pdf = pdf.drop('type', axis=1)
    pdf = pdf.groupby(['state_code', 'county_code', 'year', 'quarter'], 
                      as_index=False).sum()

    pbdf = bdf.merge(pdf, on=['state_code', 'county_code', 'year', 
                              'quarter'], how='outer', indicator=True)
    pbdf._merge = pbdf._merge.map({'left_only': 'birds', 
                                   'right_only': 'poultry',
                                   'both': 'both'})
    pbdf._merge = pbdf._merge.astype(str)

    fig, gs = plot.initiate_figure(x=5*4, y=4*3, 
                                   gs_nrows=3, gs_ncols=4,
                                   gs_wspace=-.1, gs_hspace=-.2,
                                   color='tableau10')

    colors = plot.COLORS['tableau10']
    colormap = {
            "birds": colors[0],
            "poultry": colors[1],
            "both": colors[2],
            }
    legend_elements = [Patch(facecolor=colormap[key], label=key) 
                       for key in colormap]

    # load shapes
    regions = gpd.read_file('../scripts/usa/cb_2018_us_county_20m.shp')
    regions.columns = regions.columns.str.lower()
    regions.name = regions.name.str.lower()
    regions = regions.astype({'statefp': 'int', 'countyfp': 'int'})
    regions = regions[~regions.statefp.isin([2, 15, 72, 66, 69, 78, 60])]

    states = regions[['statefp', 'geometry']].dissolve(by='statefp')
    rpbdf = regions.merge(pbdf, left_on=['statefp', 'countyfp'],
                          right_on=['state_code', 'county_code']).fillna(0)

    i = 0
    for year in [2022, 2023, 2024]:
        j = 0
        for quarter in [1,2,3,4]:
            if j:
                ylabel = ''
            else:
                ylabel = f'{year}'
            if i == 2:
                xlabel = f'{quarter}'
            else:
                xlabel = ''
            tdf = rpbdf[(rpbdf.year==year) & (rpbdf.quarter==quarter)]
            bi = tdf.incidences.sum().astype(int)
            pi = tdf.reports.sum().astype(int)
            ax = plot.subplot(fig=fig, grid=gs[i,j], 
                              func='gpd.boundary.plot',
                              pf_facecolor='white', pf_edgecolor='grey',
                              pf_linewidth=.1, data=states) 
            ax = plot.subplot(fig=fig, ax=ax, grid=gs[i,j], func='gpd.plot',
                              data=tdf, 
                              pf_color=tdf._merge.map(colormap),
                              pf_markersize=2,
                              la_ylabel=ylabel, fs_ylabel='large',
                              la_title=f'b: {bi}, p: {pi}',
                              la_xlabel=xlabel, fs_xlabel='large')
            j += 1
        i += 1
    ax.legend(handles=legend_elements, 
              loc="lower left", bbox_to_anchor=(.7,.9), 
              fontsize=15, title_fontsize=12)
    plot.savefig('birds_poultry_corr.pdf')

    bdf_type = utils.h5n1_bird_types(agg_by_quarter=True)
    #replace 'bird species' House Sparrow' with 'House sparrow'
    bdf_type['bird species'] = bdf_type['bird species'].replace('House Sparrow', 'House sparrow')
    #group and sum incidences by 'bird species'
    top_birds = bdf_type.groupby(['bird species'])[['incidences']].sum().reset_index().sort_values(by='incidences', ascending=False)
    #keep the top birds assign them to color in a dictionary
    top_birds = top_birds.head(len(plot.COLORS['not_red'])-1)
    top_birds = top_birds['bird species'].to_list()
    
    bird_colormap = dict(zip(top_birds, plot.COLORS['not_red'][1:]))
    #bird_colormap other to first in tableau20
    bird_colormap['other'] = plot.COLORS['not_red'][0]

    pbdf_type = bdf_type.merge(pdf, on=['state_code', 'county_code', 'year', 
                              'quarter'], how='outer', indicator=True)
    pbdf_type._merge = pbdf_type._merge.map({'left_only': 'birds', 
                                   'right_only': 'poultry',
                                   'both': 'both'})
    pbdf_type._merge = pbdf_type._merge.astype(str)

    #also get the top_birds by year that have the most incidences with _merge == 'both'
    #top_both_birds = pbdf_type[pbdf_type._merge == 'both'].groupby(['year', 'bird species'])[['incidences']].sum().reset_index()
    pbdf_type.loc[:,'fips'] = pbdf_type['state_code'].astype(str) + pbdf_type['county_code'].astype(str)
    top_both_birds = pbdf_type[pbdf_type._merge == 'both'].groupby(['year', 'bird species'])['fips'].nunique().reset_index()
    top_both_birds = top_both_birds.rename(columns={'county_code': 'num_counties'})

    #similar but get the top 18 birds only from 2024
    top_2024_birds = pbdf_type[(pbdf_type._merge == 'both') & (pbdf_type.year == 2024)].groupby(['bird species'])['fips'].nunique().reset_index(name='num_counties')
    top_2024_birds = top_2024_birds.nlargest(18, 'num_counties')
    #keep the top six per year
    #top_both_birds = top_both_birds.groupby('year').apply(lambda x: x.nlargest(6, 'incidences')).reset_index(drop=True)

    #keep the top birds assign them to color in a dictionary
    top_both_birds = top_both_birds.head(len(plot.COLORS['not_red'])-1)
    top_both_birds = top_both_birds['bird species'].to_list()
    both_colormap = dict(zip(top_both_birds, plot.COLORS['not_red'][1:]))
    #bird_colormap other to first in tableau20
    both_colormap['other'] = plot.COLORS['not_red'][0]

    #keep the top birds assign them to color in a dictionary
    top_2024_birds = top_2024_birds.head(len(plot.COLORS['not_red'])-1)
    top_2024_birds = top_2024_birds['bird species'].to_list()
    top_2024_colormap = dict(zip(top_2024_birds, plot.COLORS['not_red'][1:]))
    #bird_colormap other to first in tableau20
    top_2024_colormap['other'] = plot.COLORS['not_red'][0]

    fig, gs = plot.initiate_figure(x=5*4, y=4*3, 
                                   gs_nrows=3, gs_ncols=4,
                                   gs_wspace=-.1, gs_hspace=-.2,
                                   color='tableau20')
    cm_reduced = {
            "poultry": colors[2],
            "both": colors[2],
    }
    alt_legend_elements = [Patch(facecolor=both_colormap[key], label=key) 
                       for key in both_colormap]
    
    top2024_legend_elements = [Patch(facecolor=top_2024_colormap[key], label=key) 
                       for key in top_2024_colormap]
    
    rpbdf_type = regions.merge(pbdf_type, left_on=['statefp', 'countyfp'],
                          right_on=['state_code', 'county_code']).fillna(0)

    i = 0
    #rpbdf select rows with NaN for bird species
    
    for year in [2022, 2023, 2024]:
        j = 0
        for quarter in [1,2,3,4]:
            if j:
                ylabel = ''
            else:
                ylabel = f'{year}'
            if i == 2:
                xlabel = f'{quarter}'
            else:
                xlabel = ''
            tdf_i = rpbdf[(rpbdf.year==year) & (rpbdf.quarter==quarter)]

            #select tdf_i rows that are duplicates in 'countyfp'
            #tdf_i = tdf_i[tdf_i.duplicated(subset='countyfp', keep=False)].sort_values(by='countyfp')
            bi = tdf_i.incidences.sum().astype(int)
            pi = tdf_i.reports.sum().astype(int)
            tdf = rpbdf_type[(rpbdf_type.year==year) & (rpbdf_type.quarter==quarter)].copy()
            #only keep the largest 'bird species' per countyfp
            tdf.loc[:, 'fips'] = tdf['statefp'].astype(str) + tdf['countyfp'].astype(str)
            #count the number of unique 'bird species' per fips
            tdf.loc[:, 'num_species'] = tdf.groupby('fips')['bird species'].transform('nunique')
            #calculate the total number of incidences per fips
            tdf.loc[:, 'total_incidences'] = tdf.groupby('fips')['incidences'].transform('sum')
            #tdf = tdf.sort_values(by='incidences', ascending=False).drop_duplicates(subset='fips')
            #combine statefp and countyfp into fips column
            #tdf['fips'] = tdf['statefp'].astype(str) + tdf['countyfp'].astype(str)
            #tdf = tdf[tdf.duplicated(subset='countyfp', keep=False)].sort_values(by='countyfp')
            #pi = tdf.drop_duplicates(subset=['year', 'quarter', 'countyfp']).reports.sum().astype(int)
            ax = plot.subplot(fig=fig, grid=gs[i,j], 
                              func='gpd.boundary.plot',
                              pf_facecolor='white', pf_edgecolor='grey',
                              pf_linewidth=.1, data=states) 
 
            ax = plot.subplot(fig=fig, ax=ax, grid=gs[i,j], func='gpd.plot',
                    data=tdf, 
                    pf_color=tdf['bird species'].map(lambda x: top_2024_colormap.get(x, top_2024_colormap['other'])),
                    pf_markersize=2,
                    pf_alpha=tdf['total_incidences'].apply(lambda x: min(x / float(3), 1)),
                    la_ylabel=ylabel, fs_ylabel='large',
                    la_title=f'b: {bi}, p: {pi}',
                    la_xlabel=xlabel, fs_xlabel='large')
            
            ax = plot.subplot(fig=fig, ax=ax, grid=gs[i,j], func='gpd.boundary.plot',
                              data=tdf_i, 
                              pf_edgecolor=tdf_i._merge.map(lambda x: cm_reduced.get(x, 'none')),
                              pf_linewidth=tdf_i._merge.map(lambda x: 0.5 if x in cm_reduced else 0),
                              pf_facecolor='none',
                              la_ylabel=ylabel, fs_ylabel='large',
                              la_title=f'b: {bi}, p: {pi}',
                              la_xlabel=xlabel, fs_xlabel='large')

            #write the total incidences and num_species on the plot as f"{total_incidences},{num_species}" in the middle of the geometry
            #for idx, row in tdf.iterrows():
            #    ax.annotate(f"{row['total_incidences']},{row['num_species']}", 
            #                xy=row['geometry'].centroid.coords[0], 
            #                ha='center', fontsize=1, color='white')

            
            j += 1
        i += 1
    ax.legend(handles=top2024_legend_elements, 
              loc="lower left", bbox_to_anchor=(.7,.9), 
              fontsize=8, title_fontsize=12)
    plot.savefig('bird_type_poultry_corr_v3.pdf')

    #create an 4 x 3 grid of plots similar to above but make a bar plot of the 'bird species' for each year/quarter from the tdf rows with _merge == 'both'

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(24, 16))
    fig.subplots_adjust(wspace=0.4, hspace=0.6)

    i = 0
    for year in [2022, 2023, 2024]:
        j = 0
        for quarter in [1, 2, 3, 4]:
            ax = axes[i, j]
            if j:
                ylabel = ''
            else:
                ylabel = f'{year}'
            if i == 2:
                xlabel = f'{quarter}'
            else:
                xlabel = ''
            tdf_i = rpbdf[(rpbdf.year==year) & (rpbdf.quarter==quarter)]
            bi = tdf_i.incidences.sum().astype(int)
            pi = tdf_i.reports.sum().astype(int)
            tdf = rpbdf_type[(rpbdf_type.year == year) & (rpbdf_type.quarter == quarter) & (rpbdf_type._merge == 'both')]
            #pi = tdf.drop_duplicates(subset=['year', 'quarter', 'countyfp']).reports.sum().astype(int)
            sns.barplot(ax=ax, data=tdf, x='bird species', y='incidences', hue='bird species', palette='tab20')
            ax.set_xlabel('bird species', fontsize='large')
            ax.set_ylabel('incidences', fontsize='large')
            ax.set_title(f'b: {bi}, p: {pi}', fontsize='large')
            #move the title down
            ax.title.set_position([.5, 1.05])
            ax.set_xlabel(xlabel, fontsize='large')
            ax.set_ylabel(ylabel, fontsize='large')
            #rotate x-axis labels to 45 degrees
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            j += 1  
        i += 1

    plt.savefig('bird_type_poultry_barplot.pdf')

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(24, 16))
    fig.subplots_adjust(wspace=0.4, hspace=0.6)

    i = 0
    for year in [2022, 2023, 2024]:
        j = 0
        for quarter in [1, 2, 3, 4]:
            ax = axes[i, j]
            if j:
                ylabel = ''
            else:
                ylabel = f'{year}'
            if i == 2:
                xlabel = f'{quarter}'
            else:
                xlabel = ''
            tdf_i = rpbdf[(rpbdf.year==year) & (rpbdf.quarter==quarter)]
            bi = tdf_i.incidences.sum().astype(int)
            pi = tdf_i.reports.sum().astype(int)
            tdf = rpbdf_type[(rpbdf_type.year == year) & (rpbdf_type.quarter == quarter)]
            #pi = tdf.drop_duplicates(subset=['year', 'quarter', 'countyfp']).reports.sum().astype(int)
            cur_birds =tdf.groupby(['bird species'])[['incidences']].sum().reset_index()
            cur_birds = cur_birds.sort_values(by='incidences', ascending=False).head(10)
            #filter tdf to only include the top 20 bird species
            tdf = tdf[tdf['bird species'].isin(cur_birds['bird species'])]
            #sort tdf according to total incidences of 'bird species' in cur_birds
            tdf = tdf.merge(cur_birds, on='bird species', how='left')
            tdf = tdf.rename(columns={'incidences_x': 'incidences', 'incidences_y': 'total_incidences'})
            tdf = tdf.sort_values(by='total_incidences', ascending=False)
            #combine total incidences with species name
            tdf['bs_total'] = tdf['bird species'] + ' (' + tdf['total_incidences'].astype(int).astype(str) + ')'
            #sort table by bs_total alphabetical
            tdf = tdf.sort_values(by='bs_total')
            #keep plot bars in same order as tdf
            #g = sns.barplot(ax=ax, data=cur_birds, x='bird species', y='incidences', hue='bird species', palette='tab20', order=tdf['bird species'])
            #g = sns.barplot(ax=ax, data=tdf, x='bird species', y='incidences', hue='bird species', palette='tab20', order=tdf['bird species'])
            #g = sns.violinplot(ax=ax, data=tdf, x='bird species', y='incidences', hue='bird species')
            g = sns.violinplot(ax=ax, data=tdf, x='bs_total', y='incidences', color="0.8")
            g = sns.stripplot(ax=ax, data=tdf, x='bs_total', y='incidences', jitter=True)
            g.legend([],[], frameon=False)
            ax.set_xlabel('bs_total', fontsize='large')
            ax.set_ylabel('incidences', fontsize='large')
            ax.set_title(f'b: {bi}, p: {pi}', fontsize='large')
            #move the title down
            ax.title.set_position([.5, 1.05])
            ax.set_xlabel(xlabel, fontsize='large')
            ax.set_ylabel(ylabel, fontsize='large')
            #rotate x-axis labels to 45 degrees
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            j += 1  
        i += 1

    plt.savefig('top_10_bird_incidences_distribution.pdf')

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(24, 16))
    fig.subplots_adjust(wspace=0.4, hspace=0.6)

    i = 0
    for year in [2022, 2023, 2024]:
        j = 0
        for quarter in [1, 2, 3, 4]:
            ax = axes[i, j]
            if j:
                ylabel = ''
            else:
                ylabel = f'{year}'
            if i == 2:
                xlabel = f'{quarter}'
            else:
                xlabel = ''
            tdf_i = rpbdf[(rpbdf.year==year) & (rpbdf.quarter==quarter)]
            bi = tdf_i.incidences.sum().astype(int)
            pi = tdf_i.reports.sum().astype(int)
            tdf = rpbdf_type[(rpbdf_type.year == year) & (rpbdf_type.quarter == quarter) &(rpbdf_type._merge == 'both')]
            #pi = tdf.drop_duplicates(subset=['year', 'quarter', 'countyfp']).reports.sum().astype(int)
            cur_birds =tdf.groupby(['bird species'])[['incidences']].sum().reset_index()
            cur_birds = cur_birds.sort_values(by='incidences', ascending=False).head(10)
            #filter tdf to only include the top 20 bird species
            tdf = tdf[tdf['bird species'].isin(cur_birds['bird species'])]
            #sort tdf according to total incidences of 'bird species' in cur_birds
            tdf = tdf.merge(cur_birds, on='bird species', how='left')
            tdf = tdf.rename(columns={'incidences_x': 'incidences', 'incidences_y': 'total_incidences'})
            tdf = tdf.sort_values(by='total_incidences', ascending=False)
            #combine total incidences with species name
            tdf['bs_total'] = tdf['bird species'] + ' (' + tdf['total_incidences'].astype(int).astype(str) + ')'
            #sort table by bs_total alphabetical
            tdf = tdf.sort_values(by='bs_total')
            #keep plot bars in same order as tdf
            #g = sns.barplot(ax=ax, data=cur_birds, x='bird species', y='incidences', hue='bird species', palette='tab20', order=tdf['bird species'])
            #g = sns.barplot(ax=ax, data=tdf, x='bird species', y='incidences', hue='bird species', palette='tab20', order=tdf['bird species'])
            #g = sns.violinplot(ax=ax, data=tdf, x='bird species', y='incidences', hue='bird species')
            g = sns.violinplot(ax=ax, data=tdf, x='bs_total', y='incidences', color="0.8")
            g = sns.stripplot(ax=ax, data=tdf, x='bs_total', y='incidences', jitter=True)
            g.legend([],[], frameon=False)
            ax.set_xlabel('bs_total', fontsize='large')
            ax.set_ylabel('incidences', fontsize='large')
            ax.set_title(f'b: {bi}, p: {pi}', fontsize='large')
            #move the title down
            ax.title.set_position([.5, 1.05])
            ax.set_xlabel(xlabel, fontsize='large')
            ax.set_ylabel(ylabel, fontsize='large')
            #rotate x-axis labels to 45 degrees
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            j += 1  
        i += 1

    plt.savefig('both_top_10_bird_distribution.pdf')