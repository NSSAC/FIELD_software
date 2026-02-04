DESC='''
This script converts the processing_plants contents to 
a csv file with the glw and fips values. This can be 
used for layers for the web portal.
 
By MW and AA

(Processing Plants compiled by AC)
'''

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import sys
import json
import geometry # originally from aautils

PROCESSING_PLANTS_FILE = 'processing_centers.csv'
CLEANED_PROCESSING_CENTERS = 'processing_center_layer.csv.zip'
GLW_TO_COUNTIES = "glw_x_y_to_state_county_lat_long.csv"

# Read the processing centers file
processing_plants_df = pd.read_csv(PROCESSING_PLANTS_FILE)

# add the GLW x and y codes based on the latitude and longitude columns
processing_plants_df[['x', 'y']] = geometry.lonlat_to_glw(processing_plants_df.longitude, processing_plants_df.latitude)
processing_plants_df = processing_plants_df.astype({'x': 'int', 'y': 'int'})

# copy the rows with no fips code to another dataframe -- we will need to deal with those separately
processing_plants_df["fips_code"] = processing_plants_df["fips_code"].fillna(0.0)
processing_plants_no_fips_df = processing_plants_df.loc[processing_plants_df["fips_code"] == 0.0]

# remove the processing_plants with no fips
processing_plants_df = processing_plants_df.loc[processing_plants_df["fips_code"] != 0.0]

# Extract the state_code and county_code and remove the fips_code
processing_plants_df["fips_code"] = processing_plants_df["fips_code"].astype(int)
processing_plants_df["fips_text"] = processing_plants_df["fips_code"].astype(str).str.zfill(5)
processing_plants_df["state_code"] = processing_plants_df["fips_text"].str[:2]
processing_plants_df["county_code"] = processing_plants_df["fips_text"].str[2:5]

# reorder the columns to put state_code, county_code, x and y at the front.
processing_plants_df = processing_plants_df [[ "state_code", "county_code", "x", "y", "establishment_id", "establishment_number", "establishment_name", "duns_number", "street","city","state","zip","phone","grant_date","activities","dbas","district","circuit","size","latitude","longitude","county","type","poultry","plant_no","name","codes" ]]

# Add the state_code, fips_code for the locations that did not have fips_code set.
# load the glw_county_lookup
glw_to_county = pd.read_csv(GLW_TO_COUNTIES)

# rename statefp and countyfp, set to strings
glw_to_county["state_code"] = glw_to_county["statefp"].astype(str).str.zfill(2)
glw_to_county["county_code"] = glw_to_county["countyfp"].astype(str).str.zfill(3)

# merge with the avian_df
processing_plants_no_fips_df = pd.merge(processing_plants_no_fips_df, glw_to_county, how="inner", on=["x","y"]).reset_index()
print(processing_plants_no_fips_df.columns.to_list())

# rename latitude_x, longitude_x, and county_x to remove suffixes
processing_plants_no_fips_df = processing_plants_no_fips_df.rename(columns={"latitude_x": "latitude", "longitude_x": "longitude", "county_x": "county"})

# reorder the columns to put state_code, county_code, x and y at the front.
processing_plants_no_fips_df = processing_plants_no_fips_df [[ "state_code", "county_code", "x", "y", "establishment_id", "establishment_number", "establishment_name", "duns_number", "street","city","state","zip","phone","grant_date","activities","dbas","district","circuit","size","latitude","longitude","county","type","poultry","plant_no","name","codes" ]]

# concatenate the processing_plants dataframes back into a single datafram
processing_plants_df = pd.concat([processing_plants_df, processing_plants_no_fips_df])

# export the converted data to a new file
processing_plants_df.to_csv(CLEANED_PROCESSING_CENTERS, index=False)

