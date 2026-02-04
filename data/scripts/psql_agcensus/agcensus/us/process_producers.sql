-- queries used to generate :table_name to store county level producers data
-- from :source_table
-- parameters:
--
-- table_name:   full table name used to store county level producers data
-- source_table: full table name for agcensus data
 
\set source_table 'nssac_agaid.agcensus_2022'
\set table_name 'nssac_agaid.agcensus_2022_producers_only_county_total'

drop table if exists :table_name cascade;

create table :table_name as
select 
    substring(concat(state_fips_code, county_code),1,2) state_fips,
    substring(concat(state_fips_code, county_code),3,3) county_fips,
    MAX(CASE WHEN short_desc = 'PRODUCERS - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS demogr_reported_all,
    MAX(CASE WHEN short_desc = 'PRODUCERS, FEMALE - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS demogr_reported_female,
    MAX(CASE WHEN short_desc = 'PRODUCERS, MALE - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS demogr_reported_male,
    MAX(CASE WHEN short_desc = 'PRODUCERS, AGE 25 TO 34 - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS age_25_to_34,
    MAX(CASE WHEN short_desc = 'PRODUCERS, AGE 35 TO 44 - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS age_35_to_44,
    MAX(CASE WHEN short_desc = 'PRODUCERS, AGE 45 TO 54 - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS age_45_to_54,
    MAX(CASE WHEN short_desc = 'PRODUCERS, AGE 55 TO 64 - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS age_55_to_64,
    MAX(CASE WHEN short_desc = 'PRODUCERS, AGE 65 TO 74 - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS age_65_to_74,
    MAX(CASE WHEN short_desc = 'PRODUCERS, AGE GE 75 - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS age_ge_75,
    MAX(CASE WHEN short_desc = 'PRODUCERS, AGE LT 25 - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS age_lt_25,
    MAX(CASE WHEN short_desc = 'PRODUCERS, AMERICAN INDIAN OR ALASKA NATIVE - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS race_aian,
    MAX(CASE WHEN short_desc = 'PRODUCERS, ASIAN - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS race_asian,
    MAX(CASE WHEN short_desc = 'PRODUCERS, BLACK OR AFRICAN AMERICAN - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS race_black,
    MAX(CASE WHEN short_desc = 'PRODUCERS, MULTI-RACE - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS race_multi_race,
    MAX(CASE WHEN short_desc = 'PRODUCERS, NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS race_nhpi,
    MAX(CASE WHEN short_desc = 'PRODUCERS, WHITE - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS race_white,
    MAX(CASE WHEN short_desc = 'PRODUCERS, HISPANIC - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS hispanic,
    MAX(CASE WHEN short_desc = 'PRODUCERS, LIVESTOCK DECISIONMAKING - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS dicisionmaking_livestock,
    MAX(CASE WHEN short_desc = 'PRODUCERS, LAND USE OR CROP DECISIONMAKING - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS decisionmaking_landuse,
    MAX(CASE WHEN short_desc = 'PRODUCERS, (ALL) - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS total_reported_all,
    MAX(CASE WHEN short_desc = 'PRODUCERS, (ALL), FEMALE - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS total_reported_female,
    MAX(CASE WHEN short_desc = 'PRODUCERS, (ALL), MALE - NUMBER OF PRODUCERS' THEN replace(value,',','')::int ELSE NULL END) AS total_reported_male
    --MAX(CASE WHEN short_desc = '' THEN replace(value,',','')::int ELSE NULL END) AS ,
from :source_table
where agg_level_desc = 'COUNTY' 
group by concat(state_fips_code, county_code);

grant select on :table_name to nssac_opendata_viewer;

-- perform some validation check
select count(*) from :table_name
where demogr_reported_all != demogr_reported_female + demogr_reported_male;

select count(*) from :table_name
where demogr_reported_all != age_25_to_34 + age_35_to_44 + age_45_to_54 + age_55_to_64 + age_65_to_74 + age_ge_75 + age_lt_25;

select count(*) from :table_name
where demogr_reported_all != race_aian + race_asian + race_black + race_multi_race + race_nhpi + race_white;
\q
