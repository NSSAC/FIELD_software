DESC='''
Extract PUMA shapes corresponding to WA.

Shapes are downloadable from 
https://usa.ipums.org/usa/resources/volii/shapefiles/ipums_puma_2010.zip

By: AA
'''

import geopandas as gpd
from pdb import set_trace

gdf = gpd.read_file('~/Downloads/ipums_puma_2010.zip')
gdf[gdf.State=='Washington'].to_file('wa_puma.shp')
