# To get full agcensus table
from os import getenv
import pandas as pd
from pdb import set_trace
import psycopg2 as pg
from sqlalchemy import create_engine

# Replace with your actual database credentials
username = getenv('PSQL_USER')
password = getenv('PSQL_PWD')
host = '127.0.0.1'
port = '5433'  # Default PostgreSQL port is 5432
database = 'geodb'

# Construct the connection string
connection_string = f'postgresql://{username}:{password}@{host}:{port}/{database}'

# Connect to the database and fetch data
engine = create_engine(connection_string)
conn = engine.connect()

query = f'''
SELECT * FROM nssac_agaid.agcensus_2022 WHERE
(group_desc='LIVESTOCK' OR group_desc='POULTRY')
LIMIT 10000
'''
xx = pd.read_sql(query, conn)

query = f'''
SELECT commodity_desc, short_desc, domain_desc, domaincat_desc, agg_level_desc,
statisticcat_desc, class_desc, group_desc,
unit_desc, state_fips_code, state_name, county_code, county_name, value 
FROM nssac_agaid.agcensus_2022 WHERE 
(group_desc='LIVESTOCK' OR group_desc='POULTRY')
'''
df = pd.read_sql(query, conn)
df.to_csv('../agcensus/agcensus_full_data.csv.zip', index=False)
