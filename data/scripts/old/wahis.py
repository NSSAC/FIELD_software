DESC = '''
Converting the WAHIS pdf.

AA
'''

from aadata import loader
import csv
import numpy as np
import pandas as pd
from pdb import set_trace
import re

def verify(df):
    print('No states:', df[df.state.isnull()].shape[0])
    print('No counties:', df[df.county.isnull()].shape[0])
    print('No start date:', df[df.start_date.isnull()].shape[0])

    print(df[(df.state.isnull()) | (df.county.isnull()) | 
             (df.start_date.isnull())])

def report_number(rep):
    return re.search(r">ob_(\d+)", rep).group(1)

def outbreak_reference(rep):
    try:
        return re.search(r'>ob_\d+\s*-\s*(.*?)<', rep).group(1)
    except:
        raise ValueError('Error in outbreak reference extraction:')

def dates(rep):
    match = re.findall(r'(\d\d\d\d/\d\d/\d\d)', rep)
    match.sort()
    if len(match) == 1:
        return match[0], np.nan
    elif len(match) == 2:
        return match[0], match[1]
    else:
        return np.nan, np.nan
        raise ValueError('Error in date extraction:', match)

def state_county(rep, states_list, counties_list):
    # state
    states = [x for x in states_list if re.search(r'>' + x + r'<', rep)] 
    ## if '145720' in rep:
    ##     with open('tp', 'w') as f: f.write(rep)
    ##     set_trace()

    if len(states) == 1:
        state = re.sub(r'\*', '', states[0])
    else:
        state = np.nan
        return np.nan, np.nan

    # county
    if ' *' in state:
        state = re.sub(r'\*', '', state)
    counties = [x for x in counties_list[state] if re.search(r'>' + x + r'<', rep)]
    if len(counties) == 1:
        county = re.sub(r'\*', '', counties[0])
        return state, county
    elif len(counties) > 1:
        counties.remove(state)   # state name == county name scenario
        if len(counties) == 1:
            county = re.sub(r'\*', '', counties[0])
            return state, county
        else:
            return state, np.nan
    else:
        return state, np.nan

def location(rep):
    match = re.search(r'(-?\d+\.\d+)[\s,]*(-?\d+\.\d+)', rep, 
                      re.MULTILINE | re.DOTALL)
    try:
        return match.group(1), match.group(2)
    except:
        return np.nan, np.nan
        raise ValueError('Error in location extraction:', match)

# Input and output file paths
summary = '../h5n1/wahis_h5n1_4451_summary.csv'
input_file = '../h5n1/wahis_h5n1_4451.html'
output_file = "wahis_bovine_reports.csv"
counties = loader.load('usa_counties')
states_list = counties.state.drop_duplicates().tolist()
states_list = [re.sub(' ', r' *', x) for x in states_list]
counties_list = counties.groupby('state')['county'].apply(list).to_dict()
for s in counties_list.keys():
    counties_list[s] = [re.sub(' ', r' *', x) for x in counties_list[s]]

# Define the pattern for splitting reports
report_pattern = re.compile(r"(>ob_\d+.*?(?=>ob_\d+|\Z))", 
                            re.MULTILINE | re.DOTALL)

# Read the input file
with open(input_file, "r") as f:
    content = f.read()
content = content.lower()
content = re.sub('\n', '', content)

# Split the content into individual reports
all_reports = report_pattern.findall(content)

# check for bovine
reports = [rep for rep in all_reports if 'bovine' in rep]

parsed_reports = []
headers = ['report_number', 'outbreak_reference', 'start_date', 
           'end_date', 'state', 'county', 'latitude', 'longitude'] 

for i,rep in enumerate(reports):
    repnum = report_number(rep)
    print('--------------------------------------------------')
    print(i+1, repnum)
    st, et = dates(rep)
    state, county = state_county(rep, states_list, counties_list)
    lat, lon = location(rep)
    fields = {
            'report_number': repnum,
            'outbreak_reference': outbreak_reference(rep),
            'start_date': st,
            'end_date': et,
            'state': state,
            'county': county,
            'latitude': lat,
            'longitude': lon,
            }

    parsed_reports.append(fields)
    ## if i>10:
    ##     break

# Write to CSV
with open(output_file, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()
    writer.writerows(parsed_reports)

df = pd.read_csv(output_file)

verify(df)

# Manual updation due to errors
df.loc[df.report_number==143167, 'start_date'] = '2024/10/28'
df.loc[df.report_number==141323, 'start_date'] = '2024/10/03'
df.loc[df.report_number==143758, 'start_date'] = '2024/10/28'
df.loc[df.report_number==141032, 
       ('start_date', 'end_date', 'state', 'county', 'latitude', 'longitude')] = \
                ('2024/10/01', np.nan, 'california', 'sacramento', 38.576 , -121.506)
df.loc[df.report_number==141029, 
       ('start_date', 'end_date', 'state', 'county', 'latitude', 'longitude')] = \
                ('2024/09/30', np.nan, 'california', 'sacramento', 38.576 , -121.503)
df.loc[df.report_number==141027, 
       ('start_date', 'end_date', 'state', 'county', 'latitude', 'longitude')] = \
                ('2024/09/30', np.nan, 'california', 'sacramento', 38.576 , -121.501)
df.loc[df.report_number==141025, 
       ('start_date', 'end_date', 'state', 'county', 'latitude', 'longitude')] = \
                ('2024/09/30', np.nan, 'california', 'sacramento', 38.576 , -121.499)
df.loc[df.report_number==141023, 
       ('start_date', 'end_date', 'state', 'county', 'latitude', 'longitude')] = \
                ('2024/09/30', np.nan, 'california', 'sacramento', 38.576 , -121.497)
df.loc[df.report_number==141021, 
       ('start_date', 'end_date', 'state', 'county', 'latitude', 'longitude')] = \
                ('2024/09/30', np.nan, 'california', 'sacramento', 38.576 , -121.495)
df.loc[df.report_number==141019, 
       ('start_date', 'end_date', 'state', 'county', 'latitude', 'longitude')] = \
                ('2024/09/30', np.nan, 'california', 'sacramento', 38.576 , -121.493)
df.loc[df.report_number==141017, 
       ('start_date', 'end_date', 'state', 'county', 'latitude', 'longitude')] = \
                ('2024/09/30', np.nan, 'california', 'sacramento', 38.576 , -121.491)
df.loc[df.report_number==140635, 
       ('start_date', 'end_date', 'state', 'county', 'latitude', 'longitude')] = \
                ('2024/09/23', np.nan, 'california', 'sacramento', 38.581 , -121.503)
df.loc[df.report_number==135371, 
       ('start_date', 'end_date', 'state', 'county', 'latitude', 'longitude')] = \
                ('2024/05/25', np.nan, 'minnesota', 'sibley', 44.556, -94.221)
df.loc[df.report_number==134994, 
       ('start_date', 'end_date', 'state', 'county', 'latitude', 'longitude')] = \
                ('2024/05/17', np.nan, 'colorado', 'weld', 40.424, -104.693)
df.loc[df.report_number==134993, 
       ('start_date', 'end_date', 'state', 'county', 'latitude', 'longitude')] = \
                ('2024/05/17', np.nan, 'colorado', 'weld', 40.424, -104.694)
df.loc[df.report_number==134786, 
       ('start_date', 'end_date', 'state', 'county', 'latitude', 'longitude')] = \
                ('2024/05/15', np.nan, 'idaho', 'cassia', 40.424, -104.694)
df.loc[df.report_number==134128, 
       ('start_date', 'end_date', 'state', 'county', 'latitude', 'longitude')] = \
                ('2024/04/02', np.nan, 'new mexico', 'curry', 34.403, -103.197)
df.loc[df.report_number==134389, 
       ('start_date', 'end_date', 'state', 'county', 'latitude', 'longitude')] = \
                ('2024/04/02', np.nan, 'new mexico', 'curry', 34.403, -103.195)

print('After manual updates')
verify(df)

df.to_csv(output_file, index=False)
print(f"Processed {len(parsed_reports)} reports into {output_file}.")

