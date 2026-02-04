try:
    import contextily as ctx
except ModuleNotFoundError as err:
    print(err)
    print('Skipping this step ...')
import geopandas as gpd
import logging
import numpy as np
import pandas as pd
from pdb import set_trace
from shapely.geometry import Point, Polygon
import unicodedata

# Constants specific to grid
DEFAULT_CRS = 'EPSG:4326'   # a geographic CRS for lat,lon
EARTH_RADIUS = 6371     # in kms (6378 at equator and 6356 at poles)
CELL_SIZE = 0.25        # degrees

# To make flat maps
CRS = {
        'latlon': 'EPSG:4326',
        'area US': 'EPSG:3395'
        }

# AA: haven't made it to work, but promising for future
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

# generate_grid_polygons() takes in cell size and bounding_box of the country. 
# Uses shape file to divide the country into cells of desired size.
def generate_grid_polygons( 
        bounding_box, 
        region_geometry, 
        moore_max,
        crs):
    
    cell_id = 0
    
    # Precision of grid
    # Total number of cells in the world
    numCells = int(180*360/(CELL_SIZE**2))
    
    # Grid initialization
    logging.info('Generating world grid ...')
    matrix = np.arange(1,
            int(numCells)+1).reshape(
                    [int(180/(CELL_SIZE)),int(360/(CELL_SIZE))])
    # All cells 
    cells = []
    edge_list = []

    # represent all of senegal as one polygon
    total_shape = region_geometry.unary_union
    
    # Create cell corresponding with each lat and long step by cell size
    ### Fixed global parameters for world
    lat_min = -90
    lat_max = 90
    lon_min = -180
    lon_max = 180

    logging.info('Processing cells within the specified boundary ...')
    for i in np.arange(lat_min, lat_max, CELL_SIZE):
        for j in np.arange(lon_min, lon_max, CELL_SIZE):
            cell_id += 1
            
            # if not in country, ignore
            if i+CELL_SIZE<bounding_box[0] or i>bounding_box[1] or j+CELL_SIZE<bounding_box[2] or j>bounding_box[3]:
                continue            
            
            # create shapely polygon for each cell
            p1 = Point(j, i)
            p2 = Point(j, i+CELL_SIZE)
            p3 = Point(j+CELL_SIZE, i+CELL_SIZE)
            p4 = Point(j+CELL_SIZE, i)            
            cell_geom = Polygon([[p.x,p.y] for p in [p1,p2,p3,p4]])
            
            # check whether the cell is in the country
            if cell_geom.intersects(total_shape):
                
                # calculate centroid and append
                cells.append([cell_id, i + CELL_SIZE/2, j + CELL_SIZE/2, cell_geom])
                
                # find Moore neighbors of each cell
                m_neighbors = get_moore_neighbor(matrix, cell_id, moore_max)
                edge_list.append(pd.DataFrame(m_neighbors,
                    columns=['source','target','moore']))    
        
    logging.info('Generating geopandas dataframe corresponding to nodes (cells) and edges (Moore neighbors) ...')
    nodes = gpd.GeoDataFrame(cells, columns=['node','y','x','geometry'])
    nodes=nodes.set_index('node')
    edges=pd.concat(edge_list, ignore_index=True)

    node_map = pd.Series(np.ones(nodes.shape[0]), index = nodes.index.values)
    node_map.values[:] = 1
    
    # remove those edges whose endpoints don't belong to nodes list.
    edges = edges[~edges.target.map(node_map).isna()]

    logging.info(f'Setting to CRS: {crs} for area calculations ...')
    nodes = nodes.set_crs(DEFAULT_CRS)
    nodes = nodes.to_crs(crs)
    region_geometry = region_geometry.to_crs(crs)

    # Assign admin 1-4
    logging.info(f'Mapping to admin levels ...')
    admin1_geometry = region_geometry.dissolve(by = 'GID_1')
    admin2_geometry = region_geometry.dissolve(by = 'GID_2')
    admin3_geometry = region_geometry.dissolve(by = 'GID_3')
    admin1 = []
    admin2 = []
    admin3 = []
    admin4 = []
    for i,node in nodes.iterrows():
        #node = gpd.GeoDataFrame(node.transpose().to_dict(), crs = crs, index = [0])
        ind = admin1_geometry.intersection(node.geometry).area.idxmax()
        if ind:
            admin1.append(admin1_geometry.loc[ind].NAME_1)
        else:
            logging.warning('No admin1 intersection for the cell.')
        ind = admin2_geometry.intersection(node.geometry).area.idxmax()
        if ind:
            admin2.append(admin2_geometry.loc[ind].NAME_2)
        else:
            logging.warning('No admin2 intersection for the cell.')
        ind = admin3_geometry.intersection(node.geometry).area.idxmax()
        if ind:
            admin3.append(admin3_geometry.loc[ind].NAME_3)
        else:
            logging.warning('No admin3 intersection for the cell.')
        ind = region_geometry.intersection(node.geometry).area.idxmax()
        if ind:
            admin4.append(region_geometry.loc[ind].NAME_4)
        else:
            logging.warning('No admin4 intersection for the cell.')
    
    nodes.loc[nodes.index,'admin1'] = admin1
    nodes.loc[nodes.index,'admin2'] = admin2
    nodes.loc[nodes.index,'admin3'] = admin3
    nodes.loc[nodes.index,'admin4'] = admin4

    logging.info('Computing Haversine distance between neighbors ...')
    coord_pairs = np.zeros((edges.shape[0],4)) # prepare matrix of x,y pairs
    coord_pairs[:,0] = edges.source.map(nodes.x)
    coord_pairs[:,1] = edges.source.map(nodes.y)
    coord_pairs[:,2] = edges.target.map(nodes.x)
    coord_pairs[:,3] = edges.target.map(nodes.y)
    edges.loc[edges.index, 'haversine'] = haversine(coord_pairs)

    return nodes, edges

# map mapspam to different cell size
def map_to_lower_res_grid(frame, cells, target_cell_size, frame_col, cells_col):
    
    frame['m_x'] = 0
    frame['m_y'] = 0
    
    valx = (2.0*frame['y']/target_cell_size)
    valy = (2.0*frame['x']/target_cell_size)
    
    frame.loc[round(valx)%2 == 1,'m_y'] = round(valx)/(2.0/target_cell_size)
    
    # Assign centroid values
    frame.loc[(round(valx)%2 == 0) & (valx < round(valx)),'m_y'] = \
            round(valx)/(2/target_cell_size) - target_cell_size/2
    
    frame.loc[(round(valx)%2 == 0) & (valx > round(valx)),'m_y'] = \
            round(valx)/(2/target_cell_size) + target_cell_size/2
    
    frame.loc[round(valy)%2 == 1,'m_x'] = round(valy)/(2.0/target_cell_size)
    
    frame.loc[(round(valy)%2 == 0) & (valy < round(valy)),'m_x'] = \
            round(valy)/(2.0/target_cell_size) - target_cell_size/2.0
    
    frame.loc[(round(valy)%2 == 0) & (valy > round(valy)),'m_x'] = \
            round(valy)/(2.0/target_cell_size) + target_cell_size/2.0
    
    frame['coord']=list(zip(frame.m_x,frame.m_y))

    cells['coord']=list(zip(cells.x,cells.y))
    cell_map=pd.Series(cells.index.values, index = cells['coord'])

    frame['cell_map']=frame.coord.map(cell_map)
    frame = frame[~frame.cell_map.isnull()]

    aggregate = frame[['cell_map', frame_col]].groupby('cell_map').sum().squeeze()

    cells = cells.join(aggregate).fillna(0).rename(
            columns = {frame_col: cells_col})
    cells = cells.drop('coord', axis='columns')
    return cells

# Find Moore neighborhood of given cell_id
def get_moore_neighbor(matrix, cell_id, moore_range):
    
    [row, col] = matrix.shape
    
    if cell_id-1<0 or cell_id-1>= row*col:
        raise Exception("Error Moore neighbor: cell_id out of range!")
    
    m = int((cell_id-1)/col) # cell_id starts from 1
    n = (cell_id-1)%col
    
    neighbor_indexs = []
    
    # find Moore neighbors
    for i in range(-1*moore_range, moore_range+1):
        for j in range(-1*moore_range, moore_range+1):
            if i==j and i==0:
                continue
            else:
                neighbor_indexs.append(([m+i,n+j],max(abs(i),abs(j))))
    
    neighbors = list()
    #weights = dict() # {neighbor_id:[short_distance, commodity, travel]}
    
    # find the corresponding ids of the Moore neighbors whose coordinates
    # were already calculated
    for ni in neighbor_indexs:
        
        # if neighbor is not in Senegal, ignore
        if ni[0][0]<0 or ni[0][0]>=row or ni[0][1]<0 or ni[0][1]>=col:
            continue
        else:
            neighbor_id = (cell_id,matrix[ni[0][0]][ni[0][1]],ni[1])
            #weights[neighbor_id] = [1, -1, -1, -1]
            neighbors.append(neighbor_id)
    
    return neighbors #, weights

# https://community.esri.com/t5/coordinate-reference-systems-blog/distance-on-a-sphere-the-haversine-formula/ba-p/902128
def haversine(x1=None, y1=None, x2=None, y2=None, units='kilometers'):
    dx = np.radians(x2 - x1) 
    dy = np.radians(y2 - y1)
    a = np.sin(dx/2)**2 + np.cos(x1) * np.cos(x2) * np.sin(dy/2)**2

    c = 2 * np.arctan2(np.sqrt(a),np.sqrt(1-a)) 
    dist = EARTH_RADIUS * c

    if units == 'kilometers':
        return dist
    elif units == 'miles':
        return dist * 0.621371
    elif units == 'meters':
        return dist * 1000
    else:
        raise ValueError(f'Unsupported unit {units}')

def coords_to_geom(x, y, from_crs=DEFAULT_CRS, to_crs=DEFAULT_CRS):
    gdf = gpd.GeoDataFrame(geometry=[Point(xy) for xy in zip(x, y)])
    gdf = gdf.set_crs(from_crs)
    gdf = gdf.to_crs(to_crs)
    return gdf

# assign population to cells using landscan data
def fill_population(data_grid_frame):

    df = pd.DataFrame.from_dict(data_grid_frame)

    pop_data = pd.read_csv(population_data, delimiter = ' ', nrows = 3, names = ['min','max'])
    df['population']=0
    
    min_lon = float(pop_data['min'].iloc[0])
    min_lat = float(pop_data['min'].iloc[1])
    max_lon = float(pop_data['max'].iloc[0])
    max_lat = float(pop_data['max'].iloc[1])
    
    country_population = pd.read_csv(population_data,delimiter=' ', skiprows = 3)
    country_population.fillna(0, inplace = True)
    
    for num, cell in df.iterrows():
    	i = 0
    	lat = cell['c_x']
    	lon = cell['c_y']

    	if lat <= max_lat and lat >= min_lat and lon <= max_lon and lon >= min_lon:

    		for index, row in country_population.iterrows():

    			j = 0 
    			if abs(max_lat - i*LANDSCAN_CELL_SIZE_DEGREES - lat) <= 0.125:
    				for elt in row:
    					if abs(min_lon + j*LANDSCAN_CELL_SIZE_DEGREES - lon) <= 0.125:
    						cell['population'] += elt
    		
    					j += 1
    			i += 1	
    	df.loc[num]=cell
    return df

def check_for_errors(nodes, edges, localities, locality_edges, hierarchy_tree):
    logging.info('NaN check ...')
    for df in [nodes, edges, localities, locality_edges, hierarchy_tree]:
        if df.isnull().sum().sum():
            raise ValueError('Found a network component with NaNs in it.')
    logging.info('Check if level 0 edges are bidirectional ...')
    df = edges[['source', 'target']].rename(columns = {'source': 'x', 'target': 'y'})
    df_ = edges[['target', 'source']].rename(columns = {'source': 'x', 'target': 'y'})
    dfa = df.append(df_).groupby(['x','y']).size()
    if (dfa != 2).sum():
        raise Exception('Level 0 edges not bidirectional.')

def lonlat_to_glw(lon, lat):
    return list(zip((12*lon + 2160.504).round(), 
        (1080.500009 - 12.000004*lat).round()))

def glw_to_lonlat(x, y):
    return list(zip((x - 2160.504)/12, -(y - 1080.500009)/12.000004))

def centroid_to_cell(lat, lon, arc_minutes=5):
    # size in minutes
    # Convert grid size from arc minutes to degrees
    degrees = arc_minutes / 60.0
    
    # Calculate the SW corner (lower-left) of the grid cell
    lat_sw = lat - (degrees / 2)
    lon_sw = lon - (degrees / 2)
    
    # Calculate the other corners
    lat_se = lat_sw
    lon_se = lon_sw + degrees
    
    lat_nw = lat_sw + degrees
    lon_nw = lon_sw
    
    lat_ne = lat_nw
    lon_ne = lon_se
    
    # Create a Polygon from the corners
    polygon = []
    for i in range(len(lon_sw)):
        polygon.append(Polygon([
            (lon_sw[i], lat_sw[i]),  # SW corner
            (lon_se[i], lat_se[i]),  # SE corner
            (lon_ne[i], lat_ne[i]),  # NE corner
            (lon_nw[i], lat_nw[i]),  # NW corner
            (lon_sw[i], lat_sw[i])   # Closing the polygon
        ]))
    return polygon


