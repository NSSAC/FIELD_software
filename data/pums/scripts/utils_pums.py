DESC='''
Some functions used by more than one analysis scripts.


For accessing geodb, using ssh tunnelling:
ssh -L 5433:postgis1:5432 <user name>@rivanna.hpc.virginia.edu (on a separate terminal)
After this, access geodb in the following manner:
    psql -d geodb -h 127.0.0.1 -p 5433 -U aa5ts

PostGIS server credentials should be stored in the following environment variables 
and exported: PSQL_USER and PSQL_PWD
'''

import logging
import numpy as np
from os import getenv
import pandas as pd
import pandas.io.sql as sqlio
from pdb import set_trace
import psycopg2 as pg

PERSON = '../data/pums_wa_person.zip'
OCCP_AG = [205, 1340, 1600, 1900, 6005, 6010, 6050, 6040, 6100, 6120, 6130,
        7810, 7830, 7840, 7850, 7855]
INDP_AG = [170, 180, 270, 190, 280, 290, 1080, 1090, 1170, 1180, 1190, 1270, 
        1070, 1280, 1290, 1390, 2180, 3070]
FOD1P_AG = [1100, 1101, 1102, 1103, 1104, 1105, 1106, 1199, 1301, 1302, 1303]

# Extract population subset relevant to agricultural occupation.
def extract_pums():
    conn = pg.connect(
            database='geodb', 
            user=getenv('PSQL_USER'),
            host='127.0.0.1',
            port='5433',
            geom_col=None,
            password=getenv('PSQL_PWD'))

    ## # Household data
    ## query = '''SELECT * FROM acs.usa_wa_pums_household_2013_2017'''
    ## df = sqlio.read_sql_query(query, conn)
    ## df.to_csv('pums_wa_household.zip', index=False)

    # Person data (pg. 44)
    query = '''SELECT 
rt,
serialno,
fod1p,
fod2p,
indp,
naicsp,
occp,
powsp,
puma,
socp
FROM acs.usa_wa_pums_person_2013_2017'''
    df = sqlio.read_sql_query(query, conn)
    df.to_csv('pums_wa_person.zip', index=False)

def select_ag(columns=None):
    person = pd.read_csv(PERSON)
    if columns != 'all':
        person = person.drop(['rt', 'fod2p', 'socp', 'naicsp', 'powsp'], axis=1)
    return person[
            (person.occp.isin(OCCP_AG)) |
            (person.fod1p.isin(FOD1P_AG)) |
            (person.indp.isin(INDP_AG))
            ]

def main():
    extract_pums()

if __name__== "__main__":
    main()
