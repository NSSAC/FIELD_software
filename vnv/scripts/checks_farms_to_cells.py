DESC='''
Verifying basic stuff from farms to cells assignment. Still not clear if it will 
remain in this form. No elaborate plots here.

By: AA
'''

import pandas as pd
from pdb import set_trace

def farm_counts(farms, agcensus):
    tdf = agcensus[(agcensus.unit=='operations') &
                   (agcensus.county_code!=-1) &
                   (agcensus.category=='county_total') &
                   (agcensus.subtype=='all')]
    ag_count = tdf.groupby(['state_code', 'county_code', 'livestock']
                           )['value'].sum().reset_index()
    fc_count = farms[(farms.subtype=='all')].groupby(
            ['state_code', 'county_code', 'livestock']
            )['fid'].count().reset_index()
    comp = fc_count.merge(ag_count, on=['state_code', 'county_code', 'livestock'],
                          how='left')

    #(comp.value - comp.fid)

    if (comp.fid!=comp.value).sum():
        set_trace()
        raise ValueError('Farm mismatch.')
    else:
        print('Farm counts match')

def optimality(stats):
    print(stats.gf_sol.value_counts())

if __name__ == '__main__':
    stats = pd.read_csv('../../livestock/results/stats_farms_to_cells.csv.zip')
    farms = pd.read_csv('../../livestock/results/farms_to_cells.csv.zip')
    agcensus = pd.read_csv(
            '../../livestock/results/agcensus_processed2_filled_counts.csv.zip')
    farm_counts(farms, agcensus)
    optimality(stats)

