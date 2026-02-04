DESC='''
Stored as json files by species--date. Clumping them together by date and storing as 
parquet files.

AA
'''

from glob import glob
from os.path import basename
import pandas as pd
from pdb import set_trace
from re import sub

files = list(glob('../../../../data/ldt/county_glw_output/**/*json'))
tot = len(files) 

dfl = []
## # Uncomment this to rerun
## for i,f in enumerate(files):
##     fl = basename(f)
## 
##     try:
##         df = pd.read_json(f)
##     except:
##         print(fl, 'failed, skipping ...')
##     df = df[df.abundance>0]
## 
##     if not df.shape[0]:
##         print(f'{fl}: No lines to read, skipping ...')
##         continue
## 
##     df = df.rename(columns={'glw_x': 'x', 'glw_y': 'y'})
##     date = sub('.*_', '', sub('.json', '', fl)) 
##     species = sub('_.*', '', fl)
##     df['species'] = species
##     df['date'] = date
## 
##     if not i % 100:
##         print(f'{i}/{tot}')
##     
##     dfl.append(df)
## 
## df = pd.concat(dfl, ignore_index=True)
## 
## df.write_parquet('birds.parquet')
## df.to_csv('birds.csv.zip', index=False)

df = pd.read_parquet('../../../../data/ldt/birds.parquet')

dff = df[['x', 'y', 'date', 'abundance']].groupby(
        ['x', 'y', 'date']).sum().reset_index()
dff.to_parquet('total_birds.parquet')
