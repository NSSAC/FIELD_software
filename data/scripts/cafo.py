DESC='''
CAFO data processing and linking with GLW cells.

By: AA
'''

from aadata import loader
import geopandas as gpd
import logging
import numpy as np
import pandas as pd
from pdb import set_trace
from re import sub

# Load data
dfl = []
for f in [
        '../cafo/cattle_beef.csv',
        '../cafo/cattle_dairy.csv',
        '../cafo/hogs.csv',
        '../cafo/poultry.csv']:
    dfl.append(pd.read_csv(f))

df = pd.concat(dfl, ignore_index=True)
df.columns = df.columns.str.lower()

df = df[['name', 'address', 'state', 'zip', 'lat', 'long', 'cafo_type',
         'cafo_subty', 'animal_cnt']].drop_duplicates()
lmap = {'Beef cattle': 'cattle', 'Dairy cattle': 'cattle', 
        'Poultry': 'chickens', 'Swine': 'hogs', 'Polutry': 'chickens'}
df['livestock'] = df.cafo_type.map(lmap)

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(
    df.long, df.lat), crs="EPSG:4326")
gdf = gdf.to_crs('EPSG:4269')

counties = loader.load('usa_county_shapes')

gdf = gpd.sjoin(gdf, counties[['statefp', 'countyfp', 'geometry']], 
                predicate='intersects')
df = pd.DataFrame(gdf.drop(['geometry', 'index_right'], axis=1))

print(df.isnull().sum())
df.to_csv('cafomaps.csv.zip', index=False)

