DESC = '''
Collect all results for dairy risk computation.

AA
'''

from glob import glob
import os
import pandas as pd
from pdb import set_trace
from shutil import move

TRASH_FOLDER = './trash_dairy/'
PARLIST = ['milk', 'milk_W', 'cattle', 'cattle_W', 'poultry', 
           'birds', 'dist', 'a_ksst']
INSTANCE_LIST = PARLIST + ['state_code', 'quarter']

try:
    old_res = pd.read_parquet('../results/dairy_risk.parquet')
    old_instances = old_res[PARLIST].drop_duplicates().shape[0]
except:
    print('Did not find dairy_risk.parquet')
    old_instances = pd.DataFrame(columns=PARLIST)

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

if old_instances.shape[0]:
    res = pd.concat([res, old_res]).reset_index(drop=True)

res = res.drop_duplicates()

if res[INSTANCE_LIST].duplicated().any():
    set_trace()
    raise ValueError('Found some duplicated instances. Something in risk computation has changed.')

res.to_parquet('dairy_risk.parquet')

for f in glob('[0-9]*parquet'):
    force_move(f, TRASH_FOLDER)

for f in glob('log*out'):
    force_move(f, TRASH_FOLDER)

