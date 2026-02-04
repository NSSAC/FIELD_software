-- queries used to generate :table_name to store county level labor data with categories
-- from :source_table
-- parameters:
--
-- table_name:   full table name used to store county level labor data with categories
-- source_table: full table name for agcensus data

drop table if exists :table_name cascade;

create table :table_name as
select state_fips_code as state_fips, county_code as county_fips, commodity_desc commodity,
       short_desc data_item, domaincat_desc domain_category, value
from :source_table
where commodity_desc = 'LABOR' and short_desc like 'LABOR, HIRED - OPERATIONS WITH WORKERS' 
  and domain_desc != 'TOTAL' and domaincat_desc != 'LABOR: (HIRED WORKERS GE 150 DAYS & LT 150 DAYS)'
  and agg_level_desc = 'COUNTY'
order by domaincat_desc;

--add grant permission
grant select on :table_name to nssac_opendata_viewer;

-- add new columns
alter table :table_name add column size_min int;
alter table :table_name add column size_max int;
alter table :table_name add column num_farm int;
alter table :table_name add column total_worker int;

-- query hangs w/o a temp table for some reasons,
-- this is a walk around
create temp table tmp_total_worker as
select state_fips_code state_fips, county_code county_fips,
       short_desc data_item, domaincat_desc domain_category, replace(value,',','')::int worker
from :source_table
where commodity_desc = 'LABOR' and short_desc = 'LABOR, HIRED - NUMBER OF WORKERS'
  and domain_desc != 'TOTAL' and value != '(D)'
  and agg_level_desc = 'COUNTY'
;

-- set total_worker
update :table_name a
set total_worker = b.worker
from tmp_total_worker b
where a.state_fips = b.state_fips and a.county_fips = b.county_fips and a.domain_category = b.domain_category;

-- set num_farm
update :table_name set num_farm = replace(value,',','')::int
where value != '(D)';

-- set size_min/size_max (ugly codes, i know)
update :table_name set size_min = 1, size_max = 4
WHERE domain_category like '% 4 %' and size_min is null;

update :table_name set size_min = 5, size_max = 9
WHERE domain_category like '% 9 %' and size_min is null;

update :table_name set size_min = 10
WHERE domain_category like '%(10 %' and size_min is null;

\q
