DESC='''
This script converts weekly_birds.zip contents to 
a csv file with the glw and fips values. This can be 
used for layers for the web portal.
 
By MW and AA
'''

import geopandas as gpd
import pandas as pd
import glob
from pdb import set_trace
from shapely.geometry import Point
import sys
import json
import timer, geometry # originally from aautils

WEEKLY_BIRDS_UNZIPPED = '/scratch/alw4ey/livestock_digital_twin_software/genbirds_ayush/data/max_glw_output/'
TEMP_FILE = "avian_lat_longs.csv"
GLW_TO_COUNTIES = "glw_x_y_to_state_county_lat_long.csv"
FINAL_OUTPUT = "avian_layer.csv"

# Step 1: convert all of the avian files to a single csv with avian type, week, abundance, and lat/long
fout = open(TEMP_FILE, "w")
fout.write("week_end,species,x,y,abundance\n")

# open file
file_list = glob.glob(WEEKLY_BIRDS_UNZIPPED + "*/*.json")
for file in file_list:
    avian = file.replace(WEEKLY_BIRDS_UNZIPPED, "")[0:6]

    #there are exceptions to the 6 character rule ...
    if avian == 'mallar':
        # Mallard
        avian = 'mallar3'
    elif avian == 'cacgoo':
        # Cackling Goose
        avian = 'cacgoo1'
    elif avian == 'caster':
        # Caspian tern
        avian = 'caster1'
    elif avian == 'snoowl':
        # Snowy Owl
        avian = 'snoowl1'
    elif avian == 'whwsco':
        # White-winged scoter
        avian = 'whwsco2'

# genbirds_ayush/data/max_glw_output/amewig/amewig_abundance_2022-01-04.json
    file_name = WEEKLY_BIRDS_UNZIPPED + avian + "/" + avian + "_abundance_"
    date_str = file.replace(file_name,"").replace(".json","")

    # open the file
    with open(file, 'r') as f:
    # Load the JSON data
        data = json.load(f)

        for row in data:
            if row["abundance"] != 0:
                fout.write(f"{date_str},{avian},{row['glw_x']},{row['glw_y']},{row['abundance']}\n")

fout.close()

# Step 2: Read the temp file, calculate the glw x and y, then sum abundance within the x, y cell
# read the temp file
avian_df = pd.read_csv(TEMP_FILE)

# Assign the GLW x, y values
avian_df = avian_df.astype({'x': 'int', 'y': 'int'})

# load the glw_county_lookup
glw_to_county = pd.read_csv(GLW_TO_COUNTIES)

# rename statefp and countyfp, set to strings
glw_to_county["state_code"] = glw_to_county["statefp"].astype(str).str.zfill(2)
glw_to_county["county_code"] = glw_to_county["countyfp"].astype(str).str.zfill(3)


# merge with the avian_df
avian_df = pd.merge(avian_df, glw_to_county, how="inner", on=["x","y"])

# drop unwanted columns
avian_df = avian_df [[ "week_end", "species", "state_code", "county_code", "x", "y", "abundance" ]]

del glw_to_county

# Step 3: merge and write out to a single zip file

# Sum the abundance per x, y cell
avian_sum_df = avian_df.groupby(["week_end", "species", "state_code", "county_code", "x", "y"])["abundance"].sum().reset_index()

compression_options = dict(method='zip', archive_name=FINAL_OUTPUT)
avian_sum_df.to_csv(FINAL_OUTPUT + ".zip", index=False, compression=compression_options)

del avian_df
del avian_sum_df




