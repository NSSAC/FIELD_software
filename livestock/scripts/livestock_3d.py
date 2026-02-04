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

# Step 1: Convert GeoSeries to Polygon Data for Pydeck
def geoseries_to_pydeck_polygons(geoseries):
    """Convert a GeoSeries of geometries to a format suitable for Pydeck."""
    polygons = []
    for geom in geoseries:
        if isinstance(geom, Polygon):
            polygons.append({"polygon": list(geom.exterior.coords)})
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                polygons.append({"polygon": list(poly.exterior.coords)})
    return polygons

def create_single_color_cmap(base_color, name="custom_cmap", num_shades=256):
    """
    Create a custom colormap based on a single base color.

    Parameters:
    - base_color: A tuple (R, G, B) with values in the range [0, 1].
    - name: The name of the colormap.
    - num_shades: The number of shades in the colormap (default is 256).

    Returns:
    - A LinearSegmentedColormap object.
    """
    # Start from white and go to the base color
    colors = [(1, 1, 1), base_color]  # From white to the base color
    #colors = [np.array([237,201,72])/255, base_color]  # From white to the base color

    cmap = LinearSegmentedColormap.from_list(name, colors, N=num_shades)
    return cmap

def layers_gen(data, admin1, admin2, outfile_prefix, args, locations=None):
    # geometry for data
    data['pydeck_geometry'] = [[c.x, c.y] for c in data.geometry.centroid]
    data_json = json.loads(data.to_json())

    if args.admin == 'state':
        bearing = 20
        zoom = 6
        general_elevation = 20
    else:
        bearing = 0
        zoom = 4
        general_elevation = 30

    # decide flat or 3d
    pitch = 80
    extrude = True
    elevation_data = 'properties.farms'
    try:
        if args.mode == 'flat':
            bearing = 0
            pitch = 0
            elevation_data = 0
    except:
        pass
            

    # Define the view for the map
    view = pdk.ViewState(
        latitude=data.geometry.centroid.y.mean(),  # Center on the average latitude
        longitude=data.geometry.centroid.x.mean(), # Center on the average longitude
        zoom=zoom,  # Zoom level
        pitch=pitch,  # Tilt the map
        bearing=bearing  # Rotate the map
    )

    if args.admin == 'country':
        lincolor='properties.color'
    else:
        lincolor=[0,0,0]

    data_layer = pdk.Layer(
        'GeoJsonLayer', #'ScatterplotLayer',
        data_json,
        pickable=True,
        stroked=True,  # line color
        filled=True,
        extruded=True,
        wireframe=True,
        #line_width_min_pixels=0,
        get_position="properties.pydeck_geometry",
        get_fill_color='properties.color',
        get_line_color=lincolor,
        get_elevation=f'properties.farms',
        elevation_scale=1000,
        opacity=.8,
        material={
            'ambient': .8,              # higher means whiter and less color
            'diffuse': .3,               # Diffuse light reflection
        ##     'shininess': 32,              # Shininess (higher = glossier)
        ##     'specularColor': [100, 100, 100]  # Specular highlight color (white)
        }
    )

    admin2_layer = pdk.Layer(
        'GeoJsonLayer',
        admin2,
        pickable=False,
        opacity=1,
        stroked=True,
        filled=False,
        extruded=True,
        wireframe=True,
        auto_highlight=False,
        elevation_scale=0,
        get_fill_color=[255,255,255],
        get_line_color='[0,0,0]',
        get_line_width=500,
        line_width_min_pixels=1
    )

    admin1_layer = pdk.Layer(
        'GeoJsonLayer',
        admin1,
        pickable=False,
        opacity=1,
        stroked=True,
        filled=False,
        extruded=True,
        wireframe=True,
        auto_highlight=True,
        elevation_scale=-general_elevation,
        get_fill_color=[255,255,255],
        get_line_color=[300,300,300],
        get_line_width=2
    )

    layers = [data_layer, admin2_layer, admin1_layer]

    if not locations is None:
        location_layer = pdk.Layer(
            "ColumnLayer",
            locations,
            extruded=True,
            wireframe=True,
            get_position='[longitude, latitude]',
            radius=10000,  # Radius in meters
            get_fill_color='color', #'color',  # Red color
            get_line_color=[255,255,255],
            get_elevation='elevation',
            elevation_scale=30000,
            opacity=.2,
            pickable=True,
        )
        layers.append(location_layer)

    # Create the map deck
    deck = pdk.Deck(
        initial_view_state=view,
        layers=layers, #[admin2_layer, admin1_layer],
        map_style='mapbox://styles/mapbox/light-v10',  # Base map style
        #map_style='dark_no_labels',  # Base map style
        #effects=[{'@@type': 'LightingEffect', 'shadowColor': [255,0,0,0.5]}]
    )

    # Save the map as an HTML file
    deck.to_html(outfile_prefix+'.html')
    print(outfile_prefix)

def get_color_from_cmap(value, cmap, norm):
    rgba_color = cmap(norm(value))
    return [int(c * 255) for c in rgba_color[:3]] # Convert to RGB

if __name__ == '__main__':
    # parser
    parser=argparse.ArgumentParser(description=DESC,
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', '--species', help="Lower case")
    parser.add_argument('-a', '--admin', default='country', help="country/state")
    parser.add_argument('-d', '--date', default='2023-01-01', help="date")
    parser.add_argument('--state', default='texas', 
                        help="State name for admin=state.")
    parser.add_argument('-m', '--mode', required=True, help="livestock/birds")
    parser.add_argument('-o', '--outfile_prefix', default='out', 
                        help="Outfile name")
    parser.add_argument('-p', '--palette', default='viridis', help="Colors for plot")
    parser.add_argument('-l', '--locations', default=None, help="Locations file")
    args = parser.parse_args()

    if args.locations is not None:
        locations = pd.read_csv(args.locations)
        locations = locations[locations['size'].isin(['Small', 'Large'])]
    else:
        locations = None

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
        # admin1 = country_geom


        states_json = json.loads(states.to_json())
        admin2 = states_json
        # admin2 = geoseries_to_pydeck_polygons(states.geometry)

    elif args.admin =='state':
        # Load data and process it
        states = loader.load('usa_state_shapes')
        states = states[states.name==args.state]
        admin1 = json.loads(states.to_json()) 
        # admin1 = geoseries_to_pydeck_polygons(states.geometry)

        statefp = states.statefp.values[0]
        counties = loader.load('usa_county_shapes')
        counties = counties[counties.statefp==statefp]
        admin2 = json.loads(counties.to_json())
        # admin2 = geoseries_to_pydeck_polygons(counties.geometry)

        # Locations modify
        if not locations is None:
            locations = locations[locations.state==args.state]

    if args.mode == 'livestock':
        colmap = {'yellow': [237,201,72], 'green': [89,161,79]}
        data = pd.read_csv('../results/farms_to_cells.csv.zip')
        data = data[data.livestock==args.species]
        if args.admin == 'state':
            data = data[data.statefp==int(statefp)]

        if args.species == 'cattle':
            # color = np.array([102, 194, 165])/300
            # color = np.array([228, 26, 28])/100
            col = [78, 121, 167]
            # col = [52, 126, 184] # colorbrewer
            color = np.array(col)/300
            pop_threshold = 100
            if args.admin=='state':
                pop_scale = .4
            else:
                pop_scale = .1
            if not args.locations is None:
                locations = locations[locations.type.isin(['slaughter', 'dairy'])]
                locations.loc[locations.type=='slaughter', ['color','elevation']] = ('yellow', 3)
                locations.loc[locations.type=='dairy', ['color','elevation']] = ('green', 10)
                locations.color = locations.color.map(colmap)
        elif args.species == 'hogs':
            # color = np.array([252, 141, 98])/300
            # color = np.array([55, 126, 184])/200
            color = np.array([242, 142, 43])/300
            pop_threshold = 500
            if args.admin=='state':
                pop_scale = .2
            else:
                pop_scale = .05
        elif args.species == 'chickens':
            # color = np.array([141, 160, 203])/300
            color = np.array([225, 87, 89])/300
            pop_threshold = 10000
            pop_scale = .05
            if not args.locations is None:
                locations = locations[locations.type=='poultry']
                locations.loc[locations.type=='poultry', ['color','elevation']] = ('yellow', 3)
                locations.color = locations.color.map(colmap)
    elif args.mode == 'birds':
        data = pd.read_parquet('../../../../data/ldt/total_birds.parquet')
        data = data[data.date==args.date]

    # mapping to glw
    glw = gpd.read_file('../../data/glw/glw_cells.shp.zip')
    gdf = glw[['x', 'y', 'geometry']].merge(data, on=['x', 'y'])
    gdf = gdf[~gdf.farms.isnull()]
    gdf = gdf[~gdf.statefp.isin([2, 15, 72, 66, 69, 78, 60])]
    
    gdfp = gdf[['x','y','geometry','heads']].groupby(['x', 'y', 'geometry']
                                                     ).sum()
    gdff = gdf[gdf.heads/gdf.farms>pop_threshold]
    gdff = gdff[['x','y','geometry','farms']].groupby(['x', 'y', 'geometry']
                                                     ).sum()
    gdf = gdfp.merge(gdff, left_index=True, right_index=True, how='left',
                     ).fillna(0).reset_index()
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')

    # color
    gdf['val'] = np.power(gdf.heads, pop_scale) #np.log(gdf.heads+1)
    gdf.val = gdf.val / gdf.val.max()
    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = create_single_color_cmap(color)
    # cmap = cm.get_cmap('viridis')
    gdf['color'] = gdf.val.apply(get_color_from_cmap, args=(cmap, norm))

    layers_gen(gdf, admin1, admin2, args.outfile_prefix, args, locations=locations)

