DESC = '''
Generate model instances for dairy risk.

AA
'''

from itertools import product
import numpy as np
import os
import pandas as pd
from pdb import set_trace
import stat

PARLIST = ['milk', 'milk_W', 'cattle', 'cattle_W', 'poultry', 
           'birds', 'dist', 'a_ksst']

NUM_BATCHES = 500

try:
    old_res = pd.read_parquet('../results/dairy_risk.parquet')
    old_res = old_res[PARLIST].drop_duplicates()
except:
    print('Did not find old results.')
    old_res = pd.DataFrame(columns=PARLIST)

def main():

    # load features and farms
    features = pd.read_parquet('../intermediate_data/risk_features.parquet')
    features = features[['x', 'y', 'county_code', 'state_code',
                         'milk', 'milk_W1', 'milk_W2',
                         'cattle', 'cattle_W1', 'cattle_W2',
                         'poultry', 'birds1', 'birds2', 'birds3', 'birds4']]

    farms = pd.read_csv('../../livestock/results/farms_to_cells.csv.zip')
    cattle = farms[farms.livestock=='cattle'].pivot(index=['x', 'y', 'fid'], 
                                                    columns='subtype',
                                                    values='heads'
                                                    ).reset_index().fillna(0)
    h5n1 = pd.read_csv('../../data/h5n1/dairy.csv').groupby(
            ['fips', 'quarter'], as_index=False)['Confirmed'].count()
    h5n1.to_parquet('h5n1.parquet')

    # This is dairy-specific farm-level risk computation. Hence, this step.
    milk = cattle[cattle.milk!=0]
    df = milk.merge(features, on=['x', 'y'], suffixes=('_f', ''), how='left')
    df.cattle = df.cattle - df.milk_f
    df.milk = df.milk - df.milk_f
    df.to_parquet('features.parquet')

    params = [
            [0] + list(np.arange(1e-6,10e-6,1e-6)),    # milk
            [0] + list(np.arange(1e-6,1e-4,5e-5)),     # milk_W
            [0] + list(np.arange(1e-6,10e-6,1e-6)),    # cattle
            [0] + list(np.arange(1e-6,1e-4,5e-5)),     # cattle_W
            [0] + list(np.arange(1e-8,1e-7,2e-8)),     # poultry
            [0] + list(np.arange(1e-6,1e-5,2e-6)),     # birds
            [1,2],                                     # distpow
            [2]                                         # ksst
            ]

    pars = pd.DataFrame.from_records(list(product(*params)), columns=PARLIST)
    tot_considered_instances = pars.shape[0]
    pars = pars.merge(old_res, how='left', indicator=True)
    pars = pars[pars._merge=='left_only']
    tot_instances = pars.shape[0]
    print(f'Number of instances: {tot_instances}/{tot_considered_instances}')

    pars = pars.drop('_merge', axis=1)

    old_fid = 0
    f = open(f'runlist0','w')
    i = 0
    for par in pars[PARLIST].values:
        fid = i//NUM_BATCHES
        if fid != old_fid:
            f.close()
            f = open(f'runlist{fid}','w')
            old_fid = fid
            print('Batch', fid)
        f.write('python ../scripts/risk_dairy.py ' + 
                ' '.join(['--' + k + ' ' + str(v) for k,v in 
                    zip(PARLIST, par)]) + f' -o {i}.parquet' + '\n')
        i += 1
    f.close()

    with open('run', 'w') as f:
        f.write(f'''sbatch \
--array=0-{NUM_BATCHES} \
../scripts/run_arr.sbatch\n''')

    os.chmod('run', os.stat('run').st_mode | stat.S_IXUSR)

    print('File to execute: run')
    print('List of instances: runlist*')

if __name__ == '__main__':
    main()
