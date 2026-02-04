DESC='''
Processing centers.

By: AA & AC
'''

from geopy.geocoders import Nominatim
import json
import pandas as pd
from pdb import set_trace
import zipfile

from aadata import loader

# Initialize the geolocator
geolocator = Nominatim(user_agent="geoapiExercises")

# Function to get coordinates
def get_coordinates(city_name):
    location = geolocator.geocode(city_name)
    if location:
        return (location.latitude, location.longitude)
    return None


meat = pd.read_csv('../processing_centers/MPI_Directory_by_Establishment_Name.csv.zip')
meat.county = meat.county.str.replace(' County','').str.lower()
meat['type'] = 'slaughter'

meat.activities = meat.activities.fillna(' ')
meat.loc[meat.activities.str.contains('Poultry'), 'type'] = 'poultry'
meat.loc[meat.activities.str.contains('Egg'), 'poultry'] = 'poultry'

states = loader.load('usa_states')
states_map = pd.Series(index=states.iso, data=states.state.tolist())
meat.state = meat.state.map(states_map)

with zipfile.ZipFile('../processing_centers/pasteurization.zip', 'r') as z:
    # Extract file names in the zip
    json_files = [file for file in z.namelist() if file.endswith('.json')]

    dat = []
    for file in json_files:
        with z.open(file) as f:
            dc = json.load(f)
            df = pd.DataFrame(list(dc.values())[0])
            df['state'] = list(dc.keys())[0]
            dat.append(df)

past = pd.concat(dat)
past.state = past.state.str.lower()
past['type'] = 'dairy'

cities = loader.load('usa_cities')[['city', 'state_name', 'lat', 'lng']]
cities = cities.rename(columns={'lat': 'latitude', 'lng': 'longitude', 'state_name': 'state'})
cities.city = cities.city.str.upper()
cities.state = cities.state.str.lower()

past = past.merge(cities, on=['city', 'state'], how='left')
past = past[~past.latitude.isnull()]
past['size'] = 'Large'

df = pd.concat([meat, past])
df.to_csv('processing_centers.csv.zip', index=False)

