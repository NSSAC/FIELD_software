DESC = '''
Relationship between population and BLS.

AA
'''

from kbdata import loader
from kbviz import plot
import click
import geopandas as gpd
import numpy as np
import pandas as pd
from pdb import set_trace

from scipy.stats import pearsonr, spearmanr

THRESHOLD1 = {
        'cattle': 10,
        'poultry': 50,
        'hogs': 24,
        'sheep': 10
        }

def modulate_counts(farms, threshold):
    farms['fips'] = farms.state_code.astype(str).str.zfill(2) + \
            farms.county_code.astype(str).str.zfill(3)
    farms = farms[
            ((farms.livestock=='cattle') & (farms.heads<=threshold['cattle'])) |
            ((farms.livestock=='poultry') & (farms.heads<=threshold['poultry'])) |
            ((farms.livestock=='hogs') & (farms.heads<=threshold['hogs'])) |
            ((farms.livestock=='sheep') & (farms.heads<=threshold['sheep']))
            ]
    fdf = farms.groupby('fips', as_index=False)['fid'].count()
    return fdf

@click.group()
def cli():
    pass

# 
@cli.command()
def total_pop():
    pop = pd.read_csv('../../population/results/population.csv.zip')
    agpop = pop[pop.ag_naics==112]
    totpop = pop["count"].sum()
    totagpop = agpop["count"].sum()
    print(f'Total pop. {totpop}')
    print(f'Ag. pop. {totagpop}')
    print(f'perc. {totagpop/totpop}')

# population comparison
@cli.command()
def bls_compare():
    df = pd.read_csv('../../population/results/population.csv.zip')
    bls = pd.read_csv('../../data/bls/2021/112.csv')
    farms = pd.read_csv('../../livestock/results/farms_to_cells.csv.zip')

    pop = df[(df.ag_naics==112) | (df.ag_socp=='4520XX')]

    #pop = pop[pop.state.isin([6, 36, 53])]

    pdf = pop.groupby(['state', 'county'], as_index=False)['count'].sum()

    pdf['fips'] = pdf.state.astype(str).str.zfill(2) + \
            pdf.county.astype(str).str.zfill(3)

    pdf = pdf.merge(bls[['area_fips', 'annual_avg_emplvl']], left_on='fips',
                    right_on='area_fips')
    pdf['rdiff'] = (pdf['count'] - pdf.annual_avg_emplvl) #/pdf['count']
    pdf = pdf.sort_values('rdiff').reset_index(drop=True)

    farms = farms[farms.subtype=='all']

    fdf = farms[farms.heads>=100][['state_code', 'county_code', 'fid']
                ].drop_duplicates().groupby(
                        ['state_code', 'county_code'],
                        as_index=False).count()
    
    pdf = pdf.merge(fdf, left_on=['state', 'county'],
                    right_on=['state_code', 'county_code'], how='left')

    fig = plot.Fig(x=5, y=4)
    sp = plot.Subplot(fig=fig, ylim=(1, None), 
                      xlim=(10, None), xscale='log', yscale='log')
    pe = plot.Scatterplot(subplot=sp, data=pdf, 
                          x='count', y='rdiff')
    sp.xlabel('Number of livestock workers in the DS')
    sp.ylabel('Pop. in BLS')

    fig.savefig('pop_bls_diff.pdf')

    fig = plot.Fig(x=5, y=4)
    sp = plot.Subplot(fig=fig, ylim=(1, None), 
                      xlim=(1, None), xscale='log', yscale='log')
    pe = plot.Scatterplot(subplot=sp, data=pdf, 
                          x='count', y='fid', color=1)
    sp.xlabel('Number of farms')
    sp.ylabel('Number of livestock workers')

    fig.savefig('pop_workers_farms.pdf')

    neg_diff = (pdf.rdiff<0).sum()
    tot_diff = pdf.shape[0]
    print('tot. counties:', tot_diff)
    print('-ve counties:', neg_diff, 'perc', neg_diff/tot_diff*100)

    return

    pcorr = pdf.groupby('state', as_index=False).apply(
            lambda x: pearsonr(x['count'], x.annual_avg_emplvl).statistic)
    pcorr = pcorr.rename(columns={None: 'pc'})
    scorr = pdf.groupby('state', as_index=False).apply(
            lambda x: spearmanr(x['count'], x.annual_avg_emplvl).statistic)
    scorr = scorr.rename(columns={None: 'sc'})
    frac = pdf.groupby('state', as_index=False)[['count', 'annual_avg_emplvl']
                                                ].sum()
    df = pcorr.merge(frac, on='state')
    df = df.merge(scorr, on='state')
    df['ratio'] = df['count'] / df.annual_avg_emplvl

@cli.command()
def seasonality():
    lbls = []
    for q in range(1,5):
        tdf = pd.read_csv(f'../../data/bls/2023/112_{q}.csv',
                          usecols=['area_fips', 'month1_emplvl', 'month2_emplvl', 
                                   'month3_emplvl'])
        tdf['employment'] = tdf[['month1_emplvl', 'month2_emplvl', 
                                 'month3_emplvl']].mean(axis=1)
        tdf['quarter'] = q
        lbls.append(tdf[['area_fips', 'quarter', 'employment']])
    bls = pd.concat(lbls)

    bls = bls[bls.area_fips.str[0].str.isdigit()]
    bls = bls[bls.area_fips.str[0].astype(int)<=5]
    bls = bls[bls.area_fips.str[2:]!='000']

    cdf = bls.groupby('area_fips').employment.describe()

    cdf['cv'] = (cdf['std']/cdf['mean']*100)
    cdf = cdf[~cdf.cv.isnull()]

    ax = plot.oneplot(fg_x=5, fg_y=4, 
                      func='sns.scatterplot', data=cdf, 
                      pf_x='mean', pf_y='cv',
                      pf_color=plot.get_style('color',2),
                      sp_ylim=(0, 'default'), sp_xlim=(0, 'default'),
                      la_xlabel='Avg. number of livestock workers', 
                      la_ylabel='Coeff. of var.')
    plot.savefig('pop_seasonality.pdf')


if __name__ == '__main__':
    cli()
