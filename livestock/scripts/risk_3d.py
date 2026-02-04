DESC = '''
Grid-level distribution.

By: AA
'''

import argparse
from aadata import loader
from aaviz import plot
import geopandas as gpd
import json
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdb import set_trace
import pydeck as pdk
from shapely.geometry import mapping, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import wkt

from livestock_3d import *

if __name__ == '__main__':
    # parser
    parser=argparse.ArgumentParser(description=DESC,
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', '--species', required=True, help="Lower case")
    args = parser.parse_args()

    # Load data and process it
    df = pd.read_csv('../results/risk_assessment_colorado.csv.zip')
    df.geometry = df.geometry.apply(wkt.loads)
    data = gpd.GeoDataFrame(df, geometry='geometry')
    data = data.set_crs('EPSG:4326')
    data = data[data.Species==args.species]

    states = loader.load('usa_state_shapes')
    states = states[states.name=='colorado']
    admin1 = json.loads(states.to_json()) 

    statefp = states.statefp.values[0]
    counties = loader.load('usa_county_shapes')
    counties = counties[counties.statefp==statefp]
    admin2 = json.loads(counties.to_json())

    if args.species == 'cattle':
        # color = np.array([102, 194, 165])/300
        # color = np.array([228, 26, 28])/100
        color = np.array([78, 121, 167])/300
        pop_scale = .8 
    elif args.species == 'hogs':
        # color = np.array([252, 141, 98])/300
        # color = np.array([55, 126, 184])/200
        color = np.array([242, 142, 43])/300
        pop_scale =  .8
    elif args.species == 'chickens':
        # color = np.array([141, 160, 203])/300
        color = np.array([225, 87, 89])/300
        pop_scale = .8 

    # color
    data['val'] = np.power(data.Bird_Abundance, pop_scale) #np.log(gdf.heads+1)
    data = data.rename(columns={'Risk_Score': 'farms'})
    data.val = data.val / data.val.max()
    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = create_single_color_cmap(color)
    data['color'] = data.val.apply(get_color_from_cmap, args=(cmap, norm))

    args.admin = 'state' # for function call
    layers_gen(data, admin1, admin2, f'colorado_{args.species}', args)

