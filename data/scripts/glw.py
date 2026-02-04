DESC='''
Extract GLW data from geodb.
For accessing geodb, using ssh tunnelling:
ssh -L 5433:postgis1:5432 <user name>@rivanna.hpc.virginia.edu (on a separate terminal)
After this, access geodb in the following manner:
    psql -d geodb -h 127.0.0.1 -p 5433 -U aa5ts

PostGIS server credentials should be stored in the following environment variables 
and exported: PSQL_USER and PSQL_PWD
'''

import geopandas as gpd
import logging
import numpy as np
import pandas as pd
from pdb import set_trace
from re import sub

def psql_to_glw():

    us_state_iso_codes = ["al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy"]
    us_state_fips_codes = [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56]

    failed_states = []
    dfl = []
    for state in us_state_iso_codes:
        table_list = [
                f'nssac_agaid.{state}_counties_glw_buffalo2015_dasymetric',
                f'nssac_agaid.{state}_counties_glw_cattle2015_dasymetric',
                f'nssac_agaid.{state}_counties_glw_chickens2015_dasymetric',
                f'nssac_agaid.{state}_counties_glw_duck2015_dasymetric',
                f'nssac_agaid.{state}_counties_glw_goat2015_dasymetric',
                f'nssac_agaid.{state}_counties_glw_horse2015_dasymetric',
                f'nssac_agaid.{state}_counties_glw_pig2015_dasymetric',
                f'nssac_agaid.{state}_counties_glw_sheep2015_dasymetric'
                ]

        for tab in table_list:
            query = f'SELECT * FROM {tab}'
            animal = sub('2015.*', '', sub('.*glw_', '', tab))
            try:
                df = gpd.GeoDataFrame.from_postgis(query, conn)
            except:
                print([state,animal], 'failed')
                failed_states.append([state,animal])
                continue
            if not df.shape[0]:
                print([state,animal], 'failed')
                failed_states.append([state,animal])
                continue
            df['livestock'] = animal
            dfl.append(df)
    print('Failed states', failed_states)
    gpd.GeoDataFrame(pd.concat(dfl, ignore_index=True)).to_file(
            filename='glw.shp.zip', driver='ESRI Shapefile')

def glw_sans_geom():
    glw = gpd.read_file('../../data/glw/glw.shp.zip')
    df = pd.DataFrame(glw.drop('geometry', axis=1))

    manually_fill = pd.DataFrame([
        (1305, 580, 0, '44', '001', 'Bristol', 'all'), 
        (1306, 580, 0, '44', '001', 'Bristol', 'all')
        ], columns=df.columns)
    df = pd.concat([df, manually_fill])

    df.to_csv('glw_sans_geom.csv.zip', index=False)

def glw_livestock():
    # GLW
    glw = pd.read_csv('../../data/glw/glw_sans_geom.csv.zip')
    name_map = {'goat': 'goats', 'chickens': 'poultry',
                'duck': 'poultry', 'pig': 'hogs', 'cattle': 'cattle', 
                'sheep': 'sheep'}
    glw.livestock = glw.livestock.map(name_map).fillna(glw.livestock)
    input(f'GLW: #null rows: {glw.livestock.isnull().sum()} (Enter to proceed)')
    glw = glw.rename(columns={'statefp': 'state_code', 
                              'countyfp': 'county_code'})
    glw[['x', 'y', 'state_code', 'county_code', 'livestock', 'val']].to_csv(
            'glw_livestock.csv.zip', index=False)
    print(glw.columns.tolist())
    print(glw.shape)
    print(glw.livestock.drop_duplicates())

def glw_cells():
    print('Reading GLW shape file ...')
    glw_geom = gpd.read_file('../../data/glw/glw.shp.zip')
    cells = glw_geom[['x', 'y', 'statefp', 'countyfp', 'geometry']].drop_duplicates()
    cells = cells.astype({'statefp': 'int', 'countyfp': 'int'})
    centroids = cells.geometry.centroid
    cells['lat'] = centroids.y
    cells['lon'] = centroids.x
    print('Writing cells to shapefile ...')
    cells.to_file(filename='glw_cells.shp.zip', driver='ESRI Shapefile')

def glw_agland():

    table_list = [
            'nssac_agaid.wa_map_glw_buffalo_arealweighted_to_wsda_crop_2020',
            'nssac_agaid.wa_map_glw_buffalo_dasymetric_to_wsda_crop_2020',
            'nssac_agaid.wa_map_glw_cattle_arealweighted_to_wsda_crop_2020',
            'nssac_agaid.wa_map_glw_cattle_dasymetric_to_wsda_crop_2020',
            'nssac_agaid.wa_map_glw_chickens_arealweighted_to_wsda_crop_2020',
            'nssac_agaid.wa_map_glw_chickens_dasymetric_to_wsda_crop_2020',
            'nssac_agaid.wa_map_glw_duck_arealweighted_to_wsda_crop_2020',
            'nssac_agaid.wa_map_glw_duck_dasymetric_to_wsda_crop_2020',
            'nssac_agaid.wa_map_glw_goat_arealweighted_to_wsda_crop_2020',
            'nssac_agaid.wa_map_glw_goat_dasymetric_to_wsda_crop_2020',
            'nssac_agaid.wa_map_glw_horse_arealweighted_to_wsda_crop_2020',
            'nssac_agaid.wa_map_glw_horse_dasymetric_to_wsda_crop_2020',
            'nssac_agaid.wa_map_glw_pig_arealweighted_to_wsda_crop_2020',
            'nssac_agaid.wa_map_glw_pig_dasymetric_to_wsda_crop_2020',
            'nssac_agaid.wa_map_glw_sheep_arealweighted_to_wsda_crop_2020',
            'nssac_agaid.wa_map_glw_sheep_dasymetric_to_wsda_crop_2020'
            ]

    dfl = []
    for tab in table_list:
        query = f'SELECT * FROM {tab}'
        animal = sub('_.*', '', sub('.*glw_', '', tab))
        df = pd.read_sql(query, conn)
        df['livestock'] = animal
        dfl.append(df)
    df = pd.concat(dfl, ignore_index=True)

def size_to_int(string):
    try:
        return int(sub(',', '', string))
    except:
        return -1
                   
if __name__ == '__main__':
    # psql_to_glw()
    # glw_sans_geom()
    glw_livestock()
