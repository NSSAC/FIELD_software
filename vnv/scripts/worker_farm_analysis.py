from glob import glob
from kbviz import plot
import numpy as np
import pandas as pd
from pdb import set_trace
from re import sub


farms = pd.read_csv('../../population/results/farms_with_bounds.csv')
## network = pd.read_parquet('../work/cell_farm_bipartite_edges.parquet')

dlist = glob('../../population/results/worker_farm_mw*.csv')
dflist = []
for fname in dlist:
    data = pd.read_csv(fname)
    miles = sub('.*mi', '', sub('.csv', '', fname))
    min_workers = sub('.*_mw', '', sub('_mi.*', '', fname))
    print('miles:', miles, 'min_workers:', min_workers)
    data['miles'] = miles
    data['min_workers'] = min_workers
    dflist.append(data)

df = pd.concat(dflist)

df = df.astype({'miles': float, 'min_workers': int})

# Checking and verifying the number of farms assigned
total_farms = farms.shape[0]
assigned_farms = df.groupby(['min_workers', 'miles']).farm.nunique().reset_index()
assigned_farms['perc'] = assigned_farms['farm'] / total_farms * 100

fig = plot.Fig(x=5, y=4)
sp = plot.Subplot(fig=fig, ylim=(None, 100))

pe = plot.Lineplot(subplot=sp, data=assigned_farms,
                   x='miles', y='perc', 
                   hue='min_workers', hue_order=[0, 1, 2])
pe.legend(title=r'\parbox{4cm}{\centering Min. workers per farm}', loc='upper left')
sp.xlabel('Max. miles between farm and residence', fontsize='small')
sp.ylabel(r'\% of farms', fontsize='small')
sp.title('Coverage of the assignment', fontsize='normalsize')

fig.savefig('worker_farm_assignment_farms_covered.pdf')

# Checking the difference between the avg. number of workers required vs. number assigned
fdf = df.groupby(['farm', 'miles', 'min_workers'], as_index=False)['count'].sum()
fdf = fdf.rename(columns={'count': 'assigned_workers'})
fdf = fdf.merge(farms[['farm', 'type', 'labor']], on='farm')
fdf['rdiff'] = ((fdf.assigned_workers - fdf.labor) / fdf.labor * 100)

fig = plot.Fig(x=5, y=4)
sp = plot.Subplot(fig=fig, ylim=(None, 100))

pe = plot.Boxplot(subplot=sp, data=fdf[fdf.min_workers==1],
                   x='miles', y='rdiff', hue='type')
pe.legend(title=r'', loc='center left', bbox_to_anchor=(1,.5))
sp.xlabel('Max. miles between farm and residence', fontsize='small')
sp.ylabel(r'\% relative difference', fontsize='small')
sp.title('Deviation of assigned workers from target', fontsize='normalsize')

fig.savefig('worker_farm_assignment_relative_difference_by_type.pdf')

# Checking the difference between the avg. number of workers required vs. number assigned
### categorize labor by size using pd.cut
fdf['size_category'] = pd.cut(fdf.labor, bins=[1,5,10,15,20,30,10000],
                             labels=['1-5','5-10','10-15','15-20','20-30','30+'])
fig = plot.Fig(x=5, y=4)
sp = plot.Subplot(fig=fig, ylim=(None, 100))

pe = plot.Boxplot(subplot=sp, data=fdf[fdf.min_workers==1],
                   x='miles', y='rdiff', hue='size_category')
pe.legend(title=r'', loc='center left', bbox_to_anchor=(1,.5))
sp.xlabel('Max. miles between farm and residence', fontsize='small')
sp.ylabel(r'\% relative difference', fontsize='small')
sp.title('Deviation of assigned workers from target', fontsize='normalsize')

fig.savefig('worker_farm_assignment_relative_difference_by_size.pdf')
