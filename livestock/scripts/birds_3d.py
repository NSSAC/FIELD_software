DESC = '''
Grid-level distribution.

By: AA
'''


import argparse
from aadata import loader
from aautils import geometry
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

from livestock_3d import *

if __name__ == '__main__':

    # parser
    parser=argparse.ArgumentParser(description=DESC,
            formatter_class=argparse.RawTextHelpFormatter)
    #parser.add_argument('-s', '--species', required=True, help="Lower case")
    parser.add_argument('-a', '--admin', default='country', help="country/state")
    parser.add_argument('-o', '--outfile_prefix', default='out', 
                        help="Outfile name")
    parser.add_argument('-p', '--palette', default='viridis', help="Colors for plot")
    args = parser.parse_args()

    if args.admin == 'country':
        # Load data and process it
        states = loader.load('usa_state_shapes')
        states = states[~states.name.isin(['alaska', 'hawaii', 'puerto rico', 
                                           'guam',
                                           'commonwealth of the northern mariana islands',
                                           'united states virgin islands',
                                           'american samoa'
                                           ])]

        country_geom = unary_union(states.geometry)
        country_json = json.loads(json.dumps(mapping(country_geom)))
        admin1 = country_json

        states_json = json.loads(states.to_json())
        admin2 = states_json
        # admin2 = geoseries_to_pydeck_polygons(states.geometry)

    elif args.admin =='state':
        # Load data and process it
        states = loader.load('usa_state_shapes')
        states = states[states.name==args.state]
        admin1 = json.loads(states.to_json()) 

        statefp = states.statefp.values[0]
        counties = loader.load('usa_county_shapes')
        counties = counties[counties.statefp==statefp]
        admin2 = json.loads(counties.to_json())

    dates = ['2022-01-04', '2022-04-05', '2022-07-05', '2022-10-04']
    data = gpd.read_file('../../../../data/ldt/top_birds/top_birds.shp')
    ## data = pd.read_parquet('../../../../data/ldt/total_birds.parquet')
    ## data = data[data.date.isin(dates)]
    ## data[['lon', 'lat']] = geometry.glw_to_lonlat(data.x, data.y)
    ## data['geometry'] = geometry.coords_to_geom(data.lon, data.lat)
    ## data = gpd.GeoDataFrame(data)
    ## poly = gpd.GeoDataFrame(index=[0], crs=data.crs, 
    ##                         geometry=[country_geom])
    ## data = gpd.overlay(data, poly, how='intersection')
    ## data.to_file(filename='birds.shp', driver='ESRI Shapefile')

    if args.admin == 'state':
        data = data[data.statefp==int(statefp)]

    # Plot
    fig, gs = plot.initiate_figure(x=8*4, y=5*1, 
                                   gs_nrows=1, gs_ncols=4,
                                   gs_wspace=.001, gs_hspace=.015,
                                   color='tableau10',
                                   scilimits=[-2,2])

    colors = [create_single_color_cmap(np.array([78, 121, 167])/300),
            create_single_color_cmap(np.array([242, 142, 43])/300),
            create_single_color_cmap(np.array([85, 173, 137])/300)]
    lab = ['January', 'April', 'July', 'October']
    species = ['mallar3', 'snogoo', 'cangoo']
    data.abundance = np.power(data.abundance,.05)


    pc = plot.RasterPlot(cols=4, rows=1)

    for i,date in enumerate(dates):
        print(date)
        #color = 'viridis' #colors[i % 2]
        r,c = pc.new()
        try:
            ax = plot.subplot(fig=fig, grid=gs[r,c], func='gpd.boundary.plot', 
                              data=states,
                              pf_facecolor='white', pf_edgecolor='gray',
                              pf_linewidth=0.2
                              )
        except:
            set_trace()
        for j,sp in enumerate(species):
            df = data[(data.date==date) & (data.species==sp)]
            plot.subplot(fig=fig, ax=ax, func='gpd.plot', 
                         data=df,
                         pf_column='abundance',
                         pf_cmap=colors[j],
                         pf_alpha=1.0/(j+1),
                         pf_markersize=2,
                         pf_legend=False, pf_legend_kwds={'shrink': 0.28},
                         la_title='',
                         la_ylabel='',
                         la_xlabel=lab[i], fs_xlabel='huge')
    plot.savefig('birds.png', dpi=300)

