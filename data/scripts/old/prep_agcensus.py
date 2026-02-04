DESC = '''
Standardizes GLW and AgCensus data.

By: AA
'''

import argparse
import geopandas as gpd
import numpy as np
import pandas as pd
from pdb import set_trace

LIVESTOCK_CATEGORIES = ['goats', 'chickens', 'hogs', 'cattle', 'sheep']

# AgCensus farms
farms = pd.read_csv('../../data/agcensus/agcensus_ops.csv.zip')
farms = farms[farms.commodity_desc.isin(LIVESTOCK_CATEGORIES)]
farms.to_csv('agcensus_ops_processed.csv.zip', index=False)
print(farms.columns, farms.shape)

# AgCensus heads
heads = pd.read_csv('../../data/agcensus/agcensus_heads.csv.zip')
heads = heads[heads.commodity_desc.isin(LIVESTOCK_CATEGORIES)]
input(f'AgCensus heads: #null rows: {heads.commodity_desc.isnull().sum()} (Enter to proceed)')
heads.to_csv('agcensus_heads_processed.csv.zip', index=False)
print(heads.columns, heads.shape)
