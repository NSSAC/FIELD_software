-- create a master table called :table_name for livestock and poultry from :source_table
-- the original source for :source_table is https://www.nass.usda.gov/datasets/qs.census2022.txt.gz
-- it was downloaded and loaded on 06/11/2022 using load.sh

\set source_table 'nssac_agaid.agcensus_2022'
\set table_name 'nssac_agaid.agcensus_2022_processed'

drop table if exists :table_name cascade;

create table :table_name as
select state_fips_code as state_fips, county_code as county_fips,
  commodity_desc commodity,short_desc data_item,
  domaincat_desc domain_category,value
from :source_table
where 
(
       (commodity_desc = 'CATTLE' and short_desc like 'CATTLE, INCL CALVES - OPERATIONS WITH INVENTORY' and domain_desc != 'TOTAL')
    or (commodity_desc = 'GOATS' and short_desc like 'GOATS - OPERATIONS WITH INVENTORY')
    or (commodity_desc = 'GUINEAS' and short_desc like 'GUINEAS - OPERATIONS WITH INVENTORY')
    or (commodity_desc = 'HOGS' and short_desc like 'HOGS - OPERATIONS WITH INVENTORY' and domain_desc != 'TOTAL')
    or (commodity_desc = 'POULTRY, OTHER' and short_desc like 'POULTRY, OTHER, INCL DUCKS & GEESE - OPERATIONS WITH INVENTORY')
    or (commodity_desc = 'POULTRY TOTALS' and short_desc like 'POULTRY TOTALS - OPERATIONS WITH INVENTORY')
    or (commodity_desc = 'SHEEP' and short_desc like 'SHEEP, INCL LAMBS - OPERATIONS WITH INVENTORY' and domain_desc != 'TOTAL')
)
and agg_level_desc = 'COUNTY'
order by county_code, commodity_desc, domaincat_desc
;

--add grant permission
grant select on :table_name to nssac_opendata_viewer;

-- add new columns
alter table :table_name add column size_min int;
alter table :table_name add column size_max int;
alter table :table_name add column num_farm int;

-- set num_farm
update :table_name set num_farm = replace(value,',','')::int
where value NOT LIKE '%D%';

-- set size_min/size_max (ugly codes, i know)
update :table_name set size_min = 1, size_max = 9
where commodity = 'CATTLE' and domain_category like '% 9 %' and size_min is null;

update :table_name set size_min = 10, size_max = 19
where commodity = 'CATTLE' and domain_category like '% 19 %' and size_min is null;

update :table_name set size_min = 20, size_max = 49
where commodity = 'CATTLE' and domain_category like '% 49 %' and size_min is null;

update :table_name set size_min = 50, size_max = 99
where commodity = 'CATTLE' and domain_category like '% 99 %' and size_min is null;

update :table_name set size_min = 100, size_max = 199
where commodity = 'CATTLE' and domain_category like '% 199 %' and size_min is null;

update :table_name set size_min = 200, size_max = 499
where commodity = 'CATTLE' and domain_category like '% 499 %' and size_min is null;

update :table_name set size_min = 500
where commodity = 'CATTLE' and domain_category like '%(500%' and size_min is null;

-- HOGS
update :table_name set size_min = 1, size_max = 24
where commodity = 'HOGS' and domain_category like '% 24 %' and size_min is null;

update :table_name set size_min = 25, size_max = 49
where commodity = 'HOGS' and domain_category like '% 49 %' and size_min is null;

update :table_name set size_min = 50, size_max = 99
where commodity = 'HOGS' and domain_category like '% 99 %' and size_min is null;

update :table_name set size_min = 100, size_max = 199
where commodity = 'HOGS' and domain_category like '% 199 %' and size_min is null;

update :table_name set size_min = 200, size_max = 499
where commodity = 'HOGS' and domain_category like '% 499 %' and size_min is null;

update :table_name set size_min = 500, size_max = 999
where commodity = 'HOGS' and domain_category like '% 999 %' and size_min is null;

update :table_name set size_min = 1000
where commodity = 'HOGS' and domain_category like '%(1,000%' and size_min is null;

-- SHEEP
update :table_name set size_min = 1, size_max = 24
where commodity = 'SHEEP' and domain_category like '% 24 %' and size_min is null;

update :table_name set size_min = 25, size_max = 99
where commodity = 'SHEEP' and domain_category like '% 99 %' and size_min is null;

update :table_name set size_min = 100, size_max = 299
where commodity = 'SHEEP' and domain_category like '% 299 %' and size_min is null;

update :table_name set size_min = 300, size_max = 999
where commodity = 'SHEEP' and domain_category like '% 999 %' and size_min is null;

update :table_name set size_min = 1000
where commodity = 'SHEEP' and domain_category like '%(1,000%' and size_min is null;

\q
