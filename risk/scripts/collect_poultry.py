DESC = '''
Collect all results for risk computation.

AA
'''

from glob import glob
import os
import pandas as pd
from pdb import set_trace
from shutil import move

from parlist import POULTRY_PARLIST as PARLIST

TRASH_FOLDER = './trash_poultry/'

try:
    old_res = pd.read_parquet('../results/poultry_models_eval.parquet')
    old_instances = old_res[PARLIST].drop_duplicates().shape[0]
except:
    print('Did not find poultry_models_eval.parquet')
    old_instances = 0

os.makedirs(TRASH_FOLDER, exist_ok=True)

dfl = []

def force_move(file, dest):
    try:
        os.remove(f'{dest}/{file}')
    except:
        pass
    move(file, dest)


print(f'Old instances: {old_instances}')
i = 0
for f in glob('[0-9]*parquet'):
    dfl.append(pd.read_parquet(f))
    i += 1
    if not i % 100:
        print(i)
print(f'Final instances: {i}')

res = pd.concat(dfl).reset_index(drop=True)
print('concatenating done')

if old_instances:
    res = pd.concat([res, old_res]).reset_index(drop=True)

res = res.drop_duplicates()

if res[PARLIST].duplicated().any():
    raise ValueError('Found some duplicated instances. Something in risk computation has changed.')

print(f'Total models: {res.shape[0]}')
res.to_parquet('poultry_models_eval.parquet')

## print('Removing files ...')
## for f in glob('[0-9]*parquet'):
##     force_move(f, TRASH_FOLDER)
## 
## for f in glob('log*out'):
##     force_move(f, TRASH_FOLDER)

