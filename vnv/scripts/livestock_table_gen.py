
import pandas as pd
from pdb import set_trace

df = pd.read_csv('../../livestock/results/agcensus_filled_gaps.csv.zip')
df = df.drop(df[(df.livestock!='cattle') & (df.subtype=='all')].index)
df.loc[(df.livestock!='poultry') & (df.subtype=='dummy'), 'subtype'
        ] = 'all'
poultry = df.livestock.drop_duplicates().tolist()

for ll in ['cattle', 'hogs', 'sheep']:
    poultry.remove(ll)
pind = df.livestock.isin(poultry)
df.loc[pind, 'subtype'] = df[pind].livestock
df.loc[pind, 'livestock'] = 'poultry'

fdf = pd.read_csv('../../livestock/results/farms_to_cells.csv.zip')

print('-----\nSummary\n-----')
df1 = df[(df.category=='state_total') & (df.unit=='heads')][
        ['livestock', 'subtype', 'value_original']].groupby(
                ['livestock', 'subtype']).sum()
df2 = df[(df.category=='county_by_farmsize') & (df.unit=='heads')][
        ['livestock', 'subtype', 'value_original', 'value']].groupby(
                ['livestock', 'subtype']).sum()
df1 = df1.join(df2, lsuffix='-state', rsuffix='-county')
df3 = df[(df.category=='state_total') & (df.unit=='operations')][
        ['livestock', 'subtype', 'value_original', 'state']
        ].drop_duplicates().groupby(['livestock', 'subtype']).sum()
df4 = df[(df.category=='county_by_farmsize') & (df.unit=='operations')][
        ['livestock', 'subtype', 'value']].groupby(
                ['livestock', 'subtype']).sum()

df1 = df1.join(df3, how='right').fillna(-1)
df1 = df1.join(df4, how='right', rsuffix='-ipf').fillna(-1)
df1 = df1.astype({'value_original-state': 'int', 
                  'value_original-county': 'int', 
                  'value_original': 'int',
                  'value': 'int'})

fdf1 = fdf[['livestock', 'subtype', 'heads', 'fid']].groupby(
        ['livestock', 'subtype']).agg({'heads': 'sum', 'fid': 'count'})

df1 = df1.join(fdf1)
df1 = df1[['value_original-state', 'value_original-county', 'value', 
           'heads', 'value_original', 'fid']]

df1 = df1.rename(columns={
        'value_original-state': ('heads', 'state tot.'), 
        'value_original-county': ('heads', 'county tot.'), 
        'value': ('heads', 'filled gaps'), 
        'heads': ('heads', 'final'),
        'value_original': ('farms', 'state'), 
        'fid': ('farms', 'processed')})
df1.columns = pd.MultiIndex.from_tuples(df1.columns)
df1 = df1.replace(-1, '--')
print(df1)
df1.to_latex('table_heads_farms.tex')
