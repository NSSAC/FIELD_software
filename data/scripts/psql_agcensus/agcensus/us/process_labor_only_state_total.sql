-- queries used to generate :table_name to store state level labor data (only total number)
-- from :source_table
-- parameters:
--
-- table_name:   full table name used to store state level labor data (only total number)
-- source_table: full table name for agcensus data

drop table if exists :table_name cascade;

create table :table_name as
select state_fips_code as state_fips, commodity_desc commodity,
       short_desc data_item, domaincat_desc domain_category, value
from :source_table
where commodity_desc = 'LABOR' and short_desc = 'LABOR, HIRED - NUMBER OF WORKERS'
  and domain_desc = 'TOTAL'
  and agg_level_desc = 'STATE'
order by domaincat_desc;

--add grant permission
grant select on :table_name to nssac_opendata_viewer;

-- add new columns
alter table :table_name add column total_worker int;

-- set total_worker
update :table_name set total_worker = replace(value,',','')::int where value != '(D)';

\q
