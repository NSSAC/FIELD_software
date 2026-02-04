'''
Generating human risk scores.

By AA
'''

import argparse
from kbviz import plot
import pandas as pd
from pdb import set_trace
import utils

# read worker farm assignment data
argparser = argparse.ArgumentParser()
argparser.add_argument('input', type=str, help='Input worker-farm assignment file')
argparser.add_argument('--miles', type=float, required=True, help='Miles threshold for cell neighborhood')
argparser.add_argument('--beta', type=float, required=True, help='Beta parameter for risk calculation')
argparser.add_argument('--type', type=str, required=True, help='Livestock type')

args = argparser.parse_args()
df = pd.read_csv(args.input)

# aggregate cell-level ag worker population by type
df = df[df.type==args.type]
cdf = df.groupby(['x', 'y'], as_index=False)['count'].sum()
cdf = cdf.rename(columns={'count': 'ag_worker_count'})

# read population data and aggregate to cell level
pop = pd.read_csv('../../population/results/population.csv.zip')
pdf = pop.groupby(['x', 'y'], as_index=False)['count'].sum()

# read neighborhood data and select by threshold
nmdf = pd.read_parquet('../../risk_aa/intermediate_data/glw_moore_10.parquet') # obtained from neighborhood_graph.py
nmdf = nmdf[nmdf.dist <= args.miles * 1.6]  # convert miles to km

# now merge with pop data by neighborhood + add self
pdf = pdf.merge(nmdf, on=['x', 'y'], how='left')
pdf = pdf.groupby(['x', 'y'], as_index=False).agg({'count': 'sum'})
pdf = pdf.rename(columns={'count': 'general_population'})

# merge and get total general population
cdf = cdf.merge(pdf, on=['x','y'])

# get county code
cells = pd.read_csv('../../data/glw/glw_sans_geom.csv')
cells = cells[['x', 'y', 'statefp', 'countyfp']].drop_duplicates()
cells['county_code'] = cells.statefp * 1000 + cells.countyfp
cdf = cdf.merge(cells[['x', 'y', 'county_code']], on=['x', 'y'], how='left')

print(f'Found {cdf.county_code.isnull().sum()} cells with no county code!')
cdf = cdf[~cdf.county_code.isnull()]

# compute risk score
cdf['risk_score'] = cdf.ag_worker_count * (cdf.general_population ** args.beta)
county_risk = cdf.groupby('county_code', as_index=False)['risk_score'].sum()

# make percentile categorization (very high (>95%), high (90-95%), medium (75-90%), low (50-75%), very low (<50%))
# or use a standard categorization based on the distribution
county_risk['risk_percentile'] = pd.qcut(county_risk.risk_score, 5, labels=['very low', 'low', 'medium', 'high', 'very high'])
county_risk.risk_percentile = pd.Categorical(county_risk.risk_percentile).as_ordered()


# dump to files
cdf.to_csv(f'human_risk_cells_{args.type}_m{args.miles}_b{args.beta}.csv', index=False)
county_risk.to_csv(f'human_risk_counties_{args.type}_m{args.miles}_b{args.beta}.csv', index=False)

# plot risk map
regions, states = utils.load_shapes()

county_risk = regions.merge(county_risk[['county_code', 'risk_percentile']], on='county_code', how='right')

fig = plot.Fig(x=5, y=4, constrained_layout=True)
sp = plot.Subplot(fig=fig, projection='gcrs.WebMercator()')
poly = plot.Polyplot(subplot=sp, data=states)
chor = plot.Choropleth(subplot=sp, data=county_risk, hue='risk_percentile',
                       cmap='cbYlGnBu')
chor.legend(labelsize='tiny', cbaxis=(.87,.18,.02,.3),
            title=r'Mixing', title_fontsize='small')
if args.type=='dairy':
    type_str = 'dairy cattle'
else:
    type_str = args.type
sp.title(rf'Risk to human pop. from {type_str}, Neighborhood threshold {args.miles}, $\beta=${args.beta}',
             fontsize='small')
fig.savefig(f'human_risk_{args.type}_m{args.miles}_b{args.beta}.pdf')
