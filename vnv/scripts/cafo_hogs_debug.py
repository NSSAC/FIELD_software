DESC='''
Less than 50% hog farms are being matched.

AA
'''

import pandas as pd
from pdb import set_trace

df = pd.read_csv('cafo_match_100_10000_10.csv.zip')
udf = df[(df.livestock=='hogs') & (df.fid==-1)]
mdf = df[(df.livestock=='hogs') & (df.fid!=-1)]

udf = udf.merge(mdf[['name', 'fid']], on='name', how='left')
udf = udf[udf.fid_y.isnull()]

set_trace()
