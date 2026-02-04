DESC='''
Dairy experiments analysis.

AA
'''

import argparse
import geopandas as gpd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdb import set_trace
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from aadata import loader
from aaviz import plot
from aautils.display import pf
import risk_dairy
import parlist

PARLIST = parlist.DAIRY_PARLIST

def main():

    # parser
    parser=argparse.ArgumentParser(description=DESC, 
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('analysis', help='type of analysis')
    args = parser.parse_args()

    # load data
    df = pd.read_parquet('../results/dairy_risk_processed.parquet')

    kcols = [x for x in df.columns if 'kst' in x]
    features = df[kcols]
    features.columns = [x.split('_')[1] for x in features.columns]

    ## ind_vars = pd.read_parquet('../intermediate_data/risk_features.parquet')
    ## set_trace()

    # First do elbow analysis
    if args.analysis == 'elbow':
        elbow_analysis(features)

    # Generating cluster labels for chosen clustering
    if args.analysis != 'elbow':
        cluster_labels = chosen_clustering(features, 8)
        df.loc[:, 'cluster'] = cluster_labels

    # Plot scores
    if args.analysis == 'scores':
        scores(df, features)

    # Plot risk
    if args.analysis == 'risk_map':
        risk_map(df, features, 4)

@pf
def elbow_analysis(df):
    num_clusters_list = list(range(3,15))
    inertia = []
    # elbow analysis
    for num_clusters in num_clusters_list:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)

    fig, gs = plot.initiate_figure(x=5, y=4, 
                                   gs_nrows=1, gs_ncols=1,
                                   gs_wspace=.5, gs_hspace=.4,
                                   color='tableau10')
    res = pd.DataFrame(zip(num_clusters_list,inertia), columns=['clusters', 
                                                                'inertia'])
    ax = plot.subplot(fig=fig, grid=gs[0,0], func='sns.scatterplot', 
                      data=res,
                      pf_x='clusters', pf_y='inertia',
                      la_xlabel=r'\#clusters', la_ylabel='Inertia',
                      la_title='Elbow analysis of KMeans clustering'
                      )
    plot.savefig('dairy_elbow_analysis.pdf')

@pf
def chosen_clustering(features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels

@pf
def scores(sdf, features):
    num_clusters = sdf.cluster.drop_duplicates().shape[0]

    # total score
    fig, gs = plot.initiate_figure(
            x=3, y=6, gs_nrows=num_clusters, gs_ncols=1,
            gs_wspace=.6, gs_hspace=.6, color='tableau10')

    kdf = pd.DataFrame(features.sum(axis=1)).join(sdf[['ksst', 'cluster']])
    kdf['tot'] = kdf[0] / kdf.ksst

    for cl in range(num_clusters):
        if cl == num_clusters-1:
            title = '', #r'$\textrm{score}_\textrm{tot}(\underline{\alpha})$'
        else:
            title = ''
        ## if cl == 0:
        ##     xlabel = '(b)'
        ## else:
        ##     xlabel = ''
        ax = plot.subplot(fig=fig, 
                          grid=gs[num_clusters-cl-1,0], 
                          func='sns.boxplot', 
                          data=kdf[kdf.cluster==cl],
                          pf_y='tot',
                          pf_color=plot.get_style('color', cl),
                          sp_ylim=(0,1), sp_sharex=0,
                          la_ylabel=cl, la_xlabel='',
                          la_title=title
                          )
    fig.supylabel('Cluster', fontsize=20, x=-.1)
    plot.savefig('dairy_cluster_total_score.pdf')

    # state scores
    fig, gs = plot.initiate_figure(
            x=8, y=8, gs_nrows=num_clusters, gs_ncols=1,
            gs_wspace=.6, gs_hspace=.6, color='tableau10')

    scols = [x for x in sdf.columns if 'score' in x]
    tdf = sdf[scols + ['cluster']].copy()
    tdf.columns = [x.split('_')[1] for x in list(tdf.columns) if 'score' in x
                   ] + ['cluster']
    ttdf = tdf.melt(id_vars='cluster', var_name='state',
                    value_name='score')
    ttdf.columns = ['cluster', 'state', 'score']
    states = loader.load('usa_states')
    smap = states[['fips', 'iso']].set_index('fips').squeeze()
    ttdf.state = ttdf.state.astype('int').map(smap)

    for cl in range(num_clusters):
        if cl == num_clusters-1:
            title = '', #'Scores for each state'
        else:
            title = ''
        ## if cl == 0:
        ##     xlabel = '(c)'
        ## else:
        ##     xlabel = ''

        ax = plot.subplot(fig=fig, 
                          grid=gs[num_clusters-cl-1,0], 
                          func='sns.boxplot', 
                          data=ttdf[ttdf.cluster==cl],
                          pf_x='state', pf_y='score',
                          pf_color=plot.get_style('color', cl),
                          sp_ylim=(0,1), sp_sharex=0,
                          la_ylabel=cl, la_xlabel='',
                          la_title=title
                          )
    fig.supylabel('Cluster', fontsize=20, x=.04)
    plot.savefig('dairy_cluster_state_scores.pdf')

    # parameters
    fig, gs = plot.initiate_figure(
            x=6, y=8, gs_nrows=num_clusters, gs_ncols=1,
            gs_wspace=.6, gs_hspace=.6, color='tableau10')

    pars = sdf.cluster.reset_index().melt(id_vars='cluster', var_name='par',
                                         value_name='value')
    pars = pars[pars.par!='a_ksst']
    temp_par = PARLIST.copy()
    temp_par.remove('a_ksst')
    pmap = pd.Series(
            index=temp_par,
            data=[
                '$\\alpha^P_\\textrm{milk}$',
                '$\\alpha^W_\\textrm{milk}$',
                '$\\alpha^P_\\textrm{cattle}$',
                '$\\alpha^W_\\textrm{cattle}$',
                '$\\alpha^P_\\textrm{poultry}$',
                '$\\alpha^P_\\textrm{birds}$',
                '$\\alpha_d$'
                  ])
    pars.par = pars.par.map(pmap)

    for cl in range(num_clusters):
        if cl == num_clusters-1:
            title = '', #'Parameters'
        else:
            title = ''
        ## if cl == 0:
        ##     xlabel = '(d)'
        ## else:
        ##     xlabel = ''

        ax = plot.subplot(fig=fig, 
                          grid=gs[num_clusters-cl-1,0], 
                          func='sns.boxplot', 
                          data=pars[pars.cluster==cl],
                          pf_x='par', pf_y='value',
                          pf_color=plot.get_style('color', cl),
                          sp_yscale='log', sp_sharex=0,
                          la_ylabel=cl, la_xlabel='',
                          la_title=title
                          )
    fig.supylabel('Cluster', fontsize=20, x=-.05)
    plot.savefig('dairy_cluster_params.pdf')

@pf
def risk_map(sdf, features, chosen_cluster):
    # heatmaps
    regions = loader.load('usa_county_shapes')
    regions = regions.astype({'statefp': 'int', 'countyfp': 'int'})
    regions = regions[~regions.statefp.isin([2, 15, 72, 66, 69, 78, 60])]
    features = pd.read_parquet('features.parquet')

    ### get relevant parameters
    parsdf = sdf.cluster.reset_index()
    par = parsdf[parsdf.cluster==chosen_cluster].median().to_dict()
    del par['cluster']

    ### compute risk
    rdf = risk_dairy.risk(features, par)
    riskcols = ['risk'+str(q) for q in [1,2,3,4]]
    risk = rdf.groupby(['state_code', 'county_code'])[riskcols].sum()
    risk.to_csv('dairy_risk_selected_model.csv.zip')

    milk = features.groupby(['state_code', 'county_code'])['milk_f'].sum() 
    risk = risk.join(milk)

    regions = regions.merge(risk, left_on=['statefp', 'countyfp'], 
                            right_on=['state_code', 'county_code'], 
                            how='left').fillna(0)
    for q in [1,2,3,4]:
        regions['rank'+str(q)] = pd.qcut(regions['risk'+str(q)], 
                                         q=[0, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0], 
                                         labels=['$<$ 50\\%', '50-75\\%', 
                                                 '75-90\\%', '90-95\\%', 
                                                 '95-99\\%', 'Top 1\\%'])
    colors = plot.COLORS['cbYlOrRd']
    colormap = {
            "$<$ 50\\%": colors[0],
            "50-75\\%": colors[2],
            "75-90\\%": colors[3],
            "90-95\\%": colors[4],
            "95-99\\%": colors[6],
            "Top 1\\%": colors[7],
            }

    fig, gs = plot.initiate_figure(x=8, y=4, 
                                   gs_nrows=1, gs_ncols=1,
                                   gs_wspace=.001, gs_hspace=.015,
                                   color='tableau10',
                                   scilimits=[-2,2])

    legend_elements = [Patch(facecolor=colormap[key], label=key) 
                       for key in colormap]
    for q in [1]:
        ax = plot.subplot(fig=fig, grid=gs[0,0], func='gpd.plot',
                          data=regions,
                          pf_color=regions[f'rank{q}'].map(colormap),
                          pf_markersize=2,
                          pf_legend=True, pf_legend_kwds={'shrink': 0.28},
                          la_ylabel='',
                          la_title='', #'Counties ranked by risk $R(f,t)$', 
                          la_xlabel='', fs_xlabel='Large')
        ax.legend(handles=legend_elements, 
                  loc="lower right", fontsize=10, title_fontsize=12)
    plot.savefig('dairy_risk_map.pdf')

    ## pie chart
    known_counties = pd.DataFrame.from_records([(8,123), (26,117), (26,73), (26,57), 
                                                (26,67), (26,139), (26,37), (26,155), 
                                                (26,5), (26,15), (26,65), (26,159), 
                                                (26,25)], columns=['state', 'county'])
    known_counties = known_counties.merge(regions, left_on=['state', 'county'],
                                          right_on=['statefp', 'countyfp'])
    sizes = known_counties.rank1.value_counts()
    cdf = pd.DataFrame(zip(colormap.keys(),colormap.values())).set_index(0).join(sizes)
    cdf = cdf[cdf['count']!=0]
   
    fig, gs = plot.initiate_figure(x=4, y=4, 
                                   gs_nrows=1, gs_ncols=1,
                                   gs_wspace=.001, gs_hspace=.015,
                                   color='tableau10',
                                   scilimits=[-2,2])
    ax = plot.subplot(fig=fig, grid=gs[0,0], func='pie',
                      data=cdf['count'],
                      pf_labels=cdf.index.tolist(),
                      pf_colors=cdf[1].tolist(),
                      pf_inside_display='count',
                      pf_startangle=140,
                      la_title='', #r'\parbox{7cm}{\center Ranking of counties with reported incidences}',
                      la_xlabel='', fs_xlabel='Large')
    plot.set_legend_invisible(ax)

    plot.savefig('dairy_known_counties.pdf')

    # top states
    scols = [x for x in sdf.columns if 'score' in x]
    tdf = sdf[scols + ['cluster']].copy()
    tdf.columns = [x.split('_')[1] for x in list(tdf.columns) if 'score' in x
                   ] + ['cluster']
    ttdf = tdf.melt(id_vars='cluster', var_name='state', value_name='score')
    ttdf.columns = ['cluster', 'state', 'score']

    states = loader.load('usa_states')
    smap = states[['fips', 'iso']].set_index('fips').squeeze()
    ttdf.state = ttdf.state.astype('int').map(smap)
    state_risk = regions.groupby('statefp', as_index=False)[
            ['risk1', 'risk2', 'risk3', 'risk4']].sum()
    ranks = {}
    for q in [1,2,3,4]:
        ranks[q] = state_risk.sort_values(f'risk{q}', ascending=False
                                          ).statefp.tolist()
    rndf = pd.DataFrame(ranks)

    reporting_states = ttdf.state.drop_duplicates().tolist()
    ## yy = r'\color{red}{' + xx + '}'
    ## known_smap = pd.Series(yy.values, index=xx.values)

    for c in rndf.columns:
        rndf[c] = rndf[c].map(smap)
        ind = rndf[c].isin(reporting_states)
        rndf.loc[ind, c] = r'\color{red}{' + rndf[ind][c] + '}'

    rndf.head(15).style.hide(axis='index').to_latex('dairy_top_states.tex')
    return

if __name__ == '__main__':
    main()

# No longer used
def tsne_embed(features):
    # tsne
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, 
                n_iter=1000, random_state=42)
    tsne_embedding = tsne.fit_transform(features)
    fig, gs = plot.initiate_figure(x=5, y=4, 
                                   gs_nrows=1, gs_ncols=1,
                                   gs_wspace=.5, gs_hspace=.4,
                                   color='tableau10')
    ax = plot.subplot(fig=fig, grid=gs[0,0], func='sns.scatterplot', 
                      data=df,
                      pf_palette=plot.COLORS['tableau10'],
                      pf_x='x', pf_y='y', pf_hue='cluster',
                      la_ylabel='', la_xlabel='(a)',
                      la_title='TSNE representation of instances'
                      )
    plot.savefig('dairy_tsne_clusters.pdf')
    ## df.loc[:, 'x'] = tsne_embedding[:,0]
    ## df.loc[:, 'y'] = tsne_embedding[:,1]
    return tsne_embedding

# replace fips
# add score
# add parameters plot
