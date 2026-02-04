DESC = '''
Compute the Moore neighborhood graph with distances for GLW cells.

Max. Moore range fixed to MAX_MOORE_RANGE.

AA
'''

import numpy as np
import pandas as pd
from pdb import set_trace

from aautils import geometry

MAX_MOORE_RANGE = 10

def get_moore(row):
    xl = np.arange(row['x']-MAX_MOORE_RANGE, row['x']+MAX_MOORE_RANGE+1, 
                   dtype=int)
    yl = np.arange(row['y']-MAX_MOORE_RANGE, row['y']+MAX_MOORE_RANGE+1, 
                   dtype=int)
    xx, yy = np.meshgrid(xl, yl)
    arr = np.array([xx.ravel(), yy.ravel()]).T
    new_columns = np.full(arr.shape, np.array([row['x'], row['y']]))
    df = pd.DataFrame(np.hstack((new_columns, arr)), 
                      columns=['x','y','x_','y_'])
    return df

# load glw cells
glw = pd.read_csv('../../data/glw/glw_sans_geom.csv.zip')
set_trace()
cells = glw[['x', 'y']].drop_duplicates()
ndf = pd.concat(cells.apply(get_moore, axis=1).tolist())
ndf = ndf[(ndf.x!=ndf.x_) | (ndf.y!=ndf.y_)]

c1 = pd.DataFrame(geometry.glw_to_lonlat(ndf.x, ndf.y))
c2 = pd.DataFrame(geometry.glw_to_lonlat(ndf.x_, ndf.y_))
ndf['dist'] = geometry.haversine(lon1=c1[0], lat1=c1[1], lon2=c2[0], 
                                    lat2=c2[1])

print('moore:', MAX_MOORE_RANGE, 'max distance:', ndf.dist.max(), 
      'min distance:', ndf.dist.min())

# store as parquet
ndf.to_parquet(f'glw_moore_{MAX_MOORE_RANGE}.parquet')
