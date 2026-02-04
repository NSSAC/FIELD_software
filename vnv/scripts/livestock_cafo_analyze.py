DESC = '''
Analyze the matching corresponding to CAFO maps.

By: AA
'''

from aaviz import plot
import pandas as pd
from pdb import set_trace
from re import sub
import seaborn as sns

def plot_instance(input):

    # load statistics
    match = pd.read_csv(input)
    ## farms = pd.read_csv('../../livestock/results/farms_to_cells.csv.zip')
    ## agcensus = pd.read_csv('../../livestock/results/agcensus_processed2_filled_counts.csv.zip')

    ## mdf = match[(match.livestock=='hogs') & (match.statefp==48) & (match.countyfp==195)]
    ## fdf = farms[(farms.livestock=='hogs') & (farms.state_code==48) & (farms.county_code==195)]
    ## adf = agcensus[(agcensus.livestock=='hogs') & (agcensus.state_code==48) & (agcensus.county_code==195)]

    cattle, chickens, hogs = sub('cafo_match_', '', sub('.csv.zip', '', input
                                                        )).split('_')

    # initiate figure
    fig, gs = plot.initiate_figure(x=10, y=4, 
                                   gs_nrows=1, gs_ncols=2,
                                   gs_wspace=.3, gs_hspace=.4,
                                   suptitle=f'cattle$\\ge${cattle}, chickens$\\ge${chickens}, hogs$\\ge${hogs}',
                                   st_fontsize='large', st_y=.95,

                                   color='tableau10')
    livestock = ['cattle', 'hogs', 'chickens']

    # distance
    ylabel = r'\parbox{6cm}{\centering Cummulative distribution of distances}'
    hue_order = ['cattle', 'chickens', 'hogs']
    ax = plot.subplot(fig=fig, grid=gs[0,1], func='sns.ecdfplot', 
                      sp_sharey=0,
                      data=match[(match.distance!=-1)], 
                      pf_hue='livestock',
                      pf_hue_order=hue_order,
                      pf_x='distance', 
                      pf_stat='percent',
                      sp_xlim=(0,'default'),
                      la_ylabel=ylabel,
                      lg_title=False,
                      la_title='',
                      la_xlabel='Distance (miles)')
        
    match['matched'] = 'matched'
    match.loc[match.distance==-1, 'matched'] = 'unmatched'
    match = match.sort_values('livestock')
    ax = plot.subplot(fig=fig, grid=gs[0,0], func='sns.histplot', 
                      data=match,
                      pf_x='livestock', 
                      pf_hue='matched', pf_hue_order=['unmatched', 'matched'],
                      pf_palette=[plot.get_style('color',i) for i in [2,3]],
                      pf_multiple='stack',
                      la_ylabel='\\#CAFO locations',
                      lg_title=False,
                      la_xlabel='')
    
    output = sub('_match', '', sub('.csv.zip', '.pdf', input))
    plot.savefig('livestock_' + output)


if __name__ == '__main__':
    for f in ['cafo_match_100_10000_10.csv.zip', 'cafo_match_300_20000_600.csv.zip']:
        plot_instance(f)
