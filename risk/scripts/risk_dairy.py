DESC = '''
Compute risk for milk cattle and score w.r.t. incidence reports.

AA
'''

import argparse
from itertools import product
import numpy as np
import pandas as pd
from pdb import set_trace

RISK_UPPER_THRESHOLD = 0.1
RISK_LOWER_THRESHOLD = 0.01

PARLIST = ['milk', 'milk_W', 'cattle', 'cattle_W', 'poultry', 'birds', 'dist', 
           'a_ksst']

def main():

    # parser
    parser=argparse.ArgumentParser(description=DESC, 
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--milk', type=float, required=True)
    parser.add_argument('--milk_W', type=float, required=True)
    parser.add_argument('--cattle', type=float, required=True)
    parser.add_argument('--cattle_W', type=float, required=True)
    parser.add_argument('--poultry', type=float, required=True)
    parser.add_argument('--birds', type=float, required=True)
    parser.add_argument('--dist', type=float, required=True)
    parser.add_argument('--a_ksst', type=float, required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    # load input
    df = pd.read_parquet('features.parquet')
    h5n1 = pd.read_parquet('h5n1.parquet')

    pars = {}
    for k in PARLIST:
        pars[k] = getattr(args, k)

    odf = risk_agg_score(df, pars, h5n1).drop('level_1', axis=1)

    odf.to_parquet(args.output)
    return

def risk_agg_score(df, par, h5n1):

    df = risk(df, par)
            
    cols = ['risk' + str(i) for i in range(1,5)]
    sum_risk = df.groupby('state_code', as_index=False)[cols].sum()
    sum_risk.columns = ['state_code',1,2,3,4]
    srdf = sum_risk.melt(id_vars='state_code', var_name='quarter', 
                         value_name='risk')

    sum_infected = df.groupby('state_code', as_index=False)[cols].apply(
            lambda x: (x>=1).sum())
    sum_infected.columns = ['state_code',1,2,3,4]
    sidf = sum_infected.melt(id_vars='state_code', var_name='quarter', 
                             value_name='infected_farms')

    sum_kst = df.groupby('state_code')[cols + ['milk_f']].apply(
            kst, h5n1, par).reset_index()

    odf = srdf.merge(sidf, on=['state_code', 'quarter'])
    odf = odf.merge(sum_kst, on=['state_code', 'quarter'])

    for k,v in par.items():
        odf[k] = v
    return odf

def risk(df, par):
    # risk
    if df.cattle.min() < 0:
        raise ValueError('"cattle" < "all"')
    distpow = int(par['dist'])

    for t in [1, 2, 3, 4]:
        df['risk'+str(t)] = df['milk_f'] * (1-np.exp(-(
            par['milk'] * df.milk +
            par['milk_W'] * df[f'milk_W{distpow}'] +
            par['cattle'] * df.cattle +
            par['cattle_W'] * df[f'cattle_W{distpow}'] +
            par['poultry'] * df.poultry +
            par['birds'] * df['birds'+str(t)])))
    return df

def kst(df, h5n1, par):
    hdf = h5n1[h5n1.fips==df.name]

    if not hdf.shape[0]:
        odf = pd.DataFrame(columns=['quarter', 'kst', 'ksst'])
        odf.name = df.name
        return odf

    kst_list = []
    ksst_list = []
    q_list = []
    for q in hdf.quarter.tolist():
        ksst = int(np.ceil(hdf[hdf.quarter==q].Confirmed.values[0]*par['a_ksst']))
        col = 'risk' + str(q)
        tdf = df[[col, 'milk_f']][df[col]>=1].sort_values(
                by=col, ascending=False).head(ksst)
        tdf['frac'] = tdf[col] / tdf.milk_f
        kst = tdf[(tdf.frac<=RISK_UPPER_THRESHOLD) &
                  (tdf.frac>=RISK_LOWER_THRESHOLD)].shape[0]
        q_list.append(q)
        kst_list.append(kst)
        ksst_list.append(ksst)

    odf = pd.DataFrame({'quarter': q_list, 'kst': kst_list, 
                        'ksst': ksst_list})

    odf.name = df.name
    return odf

if __name__ == '__main__':
    main()

