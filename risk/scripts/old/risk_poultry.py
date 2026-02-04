DESC = '''
Compute risk for milk cattle and score w.r.t. incidence reports.

AA
'''

import argparse
import bamboolib as bam
from itertools import product
import numpy as np
import pandas as pd
from pdb import set_trace
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from parlist import POULTRY_PARLIST as PARLIST
import utils

COMMERCIAL = True
COMMERCIAL_THRESHOLD = 10 
BIRDS_NEIGHBOR_COUNTY_SCALE = 1

def main(pars, year=None, type=None, criteria=['expected_inf_farms', 
                                               'spillover_risk', 
                                               'farm_county_quants']):

    df, h5n1_poultry, h5n1_birds_ind = load_features()
    df, h5n1_poultry = process_features(df, h5n1_poultry, h5n1_birds_ind, 2, 
                                        year, type)

    #tdf = risk_agg_score(df, pars)
    df = risk(df, pars)

    odf = evaluate(df, h5n1_poultry, pars, criteria=criteria)
    return odf

def load_features():
    features = pd.read_parquet('poultry_features.parquet')
    h5n1_poultry = utils.h5n1_poultry(agg_by='month', commercial=COMMERCIAL)
    h5n1_birds_ind = pd.read_parquet('birds_h5n1_ind_buffer.parquet')
    return features, h5n1_poultry, h5n1_birds_ind

def process_features(df, h5n1_poultry, h5n1_birds_ind, dist, year, type):
    #print('Processing features and ground truth...')
    dist = remove_trailing_zeros(dist)
    # Choosing the weighted sum of population based on distance
    retain_cols = []
    for col in df.columns:
        if '_W' not in col:
            retain_cols.append(col)
        elif f'_W{dist}' in col:
            retain_cols.append(col)

    rename_cols = {x: x[:-1] for x in retain_cols if '_W' in x} 
    df = df[retain_cols].rename(columns=rename_cols)

    df = df[df.type==type]

    h5n1_birds_ind.loc[h5n1_birds_ind.dist==0, 'ind'] = 1
    h5n1_birds_ind.loc[h5n1_birds_ind.dist==1, 'ind'] = BIRDS_NEIGHBOR_COUNTY_SCALE
    h5n1_birds_ind = h5n1_birds_ind[h5n1_birds_ind.year==year]
    h5n1_birds_ind.loc[:, 'month'] = 'binf' + h5n1_birds_ind.month.astype(str)
    hdf = h5n1_birds_ind.pivot(index=['year', 'state_code', 'county_code', 
                                      'dist'], columns='month', values='ind',
                               ).fillna(0).reset_index() 
    hdf = hdf.drop('year', axis=1)
    df = df.merge(hdf, on=['state_code', 'county_code'])
                           
    #df = bird_incidence_weighted(df, h5n1_birds_ind, year)
    h5n1_poultry = h5n1_poultry[(h5n1_poultry.year==year) &
                                (h5n1_poultry.type==type)]

    if COMMERCIAL:
        # criterion for commercial farms
        df = df[df.heads>=COMMERCIAL_THRESHOLD].copy()  
    else:
        df = df[df.heads<COMMERCIAL_THRESHOLD].copy()  
    return df, h5n1_poultry

def remove_trailing_zeros(num):
    # Convert to an integer if there's no fractional part
    if num == int(num):
        return int(num)
    return num

def risk(df, par):
    # risk
    for q in range(1,13):
        df[f'risk_prob{q}'] = 1-np.exp(-(
            par['birds_W'] * df[f'birds{q}_W'] * df[f'binf{q}']))
    # Note that this risk probability is conditioned upon the event that the farm
    # is not infected in the previous time steps
    return df

def evaluate(df, h5n1_poultry, par, 
             criteria=['expected_inf_farms', 'spillover_risk', 
                       'farm_county_quants']):
    eval = {}

    if 'expected_inf_farms' in criteria or  'spillover_risk' in criteria:
        # Compare total poultry farms infected in the year to find the right strength
        ### Probability that a farm will be infected once in a year
        df['inf'] = (1-df.risk_prob1)
        for t in range(2,13):
            df.inf = df.inf * (1-df[f'risk_prob{t}'])
        df.inf = 1 - df.inf
        ## tds = df.risk_prob1
        ## for t in range(2,13):
        ##     tds = tds * df[f'risk_prob{t}'])

    if 'expected_inf_farms' in criteria:
        eval['expected_inf_farms'] = (1-df.inf).sum()
        eval['err_expected_inf_farms'] = np.abs(
                eval['expected_inf_farms']-h5n1_poultry.reports.sum())

    if 'spillover_risk' in criteria:
        df['state_code'] = df.state_code
        df['county_code'] = df.county_code
        #df.to_csv('tp.csv')

    if 'farm_county_quants' in criteria:
        # pdf = pdf[pdf.year==2022].copy()
        nbins = 4
        tot = df.fid.sum()
        cdf = df.groupby(['state_code', 'county_code'], as_index=False)[
                ['fid', 'inf']].sum()
        cdf = cdf.sort_values('fid')
        cdf['cs'] = cdf.fid.cumsum()
        cdf['county'] = cdf.state_code*1000 + df.county_code
        cpdf = h5n1_poultry.groupby(['state_code', 'county_code', 'type'], 
                                    as_index=False)['reports'].sum()
        cpdf = cpdf[cpdf.type==df.subtype.head(1).values[0]]
        cdf = cdf.merge(cpdf, on=['state_code', 'county_code'], 
                        how='left').fillna(0)
        for i in range(nbins):
            cdf.loc[cdf.cs>tot/nbins*i, 'bin'] = i + 1
        odf = cdf.groupby('bin', as_index=False).agg({'fid': 'sum', 
                                                     'reports': 'sum',
                                                     'inf': 'sum',
                                                     'county': 'count'})

        set_trace()
        #eval['expected_inf_farms'] = tds.sum()

    ## # monthly comparison
    ## mrisk = df[[f'risk_prob{m}' for m in range(1,13)]].sum().reset_index().rename(
    ##         columns={'index': 'month', 0: 'risk'})
    ## mrisk.month = mrisk.month.str.replace('risk_prob', '').astype(int)
    ## mrisk = mrisk.sort_values('month')
    ## mbinf = df[[f'binf{m}' for m in range(1,13)]].sum().reset_index().rename(
    ##         columns={'index': 'month', 0: 'binf'})
    ## mbinf.month = mbinf.month.str.replace('binf', '').astype(int)
    ## mbinf = mbinf.sort_values('month')
    ## mreports = h5n1_poultry.groupby('month', as_index=False)[
    ##         'reports'].sum().sort_values('month')
    ## mc = mrisk.merge(mreports, on='month', how='left').fillna(0)
    ## mc = mc.merge(mbinf, on='month', how='left').fillna(0)
    ## set_trace()

    odf = pd.DataFrame.from_records(par | eval, index=[0])
    return odf

if __name__ == '__main__':
    # parser
    parser=argparse.ArgumentParser(description=DESC, 
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--birds_W', type=float, required=True)
    #parser.add_argument('--birds_W_m_1', type=float, required=True)
    #parser.add_argument('--dist', type=float, required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    pars = {}
    for k in PARLIST:
        pars[k] = getattr(args, k)

    odf = main(pars, year=args.year, type=args.type)
    odf.to_parquet(args.output)

