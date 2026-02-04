-- create a table to store the whole dataset from https://www.nass.usda.gov/datasets/qs.census2022.txt.gz
-- parameter:
-- table_name

drop table if exists :table_name cascade;

create table if not exists :table_name
(
source_desc character varying( 60),
sector_desc character varying( 60),
group_desc character varying( 80),
commodity_desc character varying( 80),
class_desc character varying( 180),
prodn_practice_desc character varying( 180),
util_practice_desc character varying( 180),
statisticcat_desc character varying( 80),
unit_desc character varying( 60),
short_desc character varying( 512),
domain_desc character varying( 256),
domaincat_desc character varying( 512),
agg_level_desc character varying( 40),
state_ansi character varying( 2),
state_fips_code character varying( 2),
state_alpha character varying( 2),
state_name character varying( 30),
asd_code character varying( 2),
asd_desc character varying( 60),
county_ansi character varying( 3),
county_code character varying( 3),
county_name character varying( 30),
region_desc character varying( 80),
zip_5 character varying( 5),
watershed_code character varying( 8),
watershed_desc character varying( 120),
congr_district_code character varying( 2),
country_code character varying( 4),
country_name character varying( 60),
location_desc character varying( 120),
year character varying( 4),
freq_desc character varying( 30),
begin_code character varying( 2),
end_code character varying( 2),
reference_period_desc character varying( 40),
week_ending character varying( 10),
load_time character varying( 19),
value character varying( 24),
CV character varying( 7)
);

grant select on table :table_name to nssac_opendata_viewer;

\q
