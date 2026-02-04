DESC='''
Some discrepancies found in GLW centroids. Checking the function.

By AA
'''

import geopandas as gpd
import logging
import numpy as np
import pandas as pd
from pdb import set_trace
from re import sub

from aautils.geometry import glw_to_lonlat, lonlat_to_glw

glw_geom = gpd.read_file('../../data/glw/glw.shp.zip')
cells = glw_geom[['x', 'y', 'statefp', 'countyfp', 'geometry']].drop_duplicates()
cells = cells.astype({'statefp': 'int', 'countyfp': 'int'})
centroids = cells.geometry.centroid
cells['lat'] = centroids.y
cells['lon'] = centroids.x
fcoords = glw_to_lonlat(cells.x, cells.y)
cells[['flon', 'flat']] = fcoords
set_trace()
print('Writing cells to shapefile ...')
cells.to_file(filename='glw_cells.shp.zip', driver='ESRI Shapefile')

