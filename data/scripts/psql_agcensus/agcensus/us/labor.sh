#! /bin/bash
# a simple wrapper to run 3 scripts to get labor number at different levels
#
source_table='nssac_agaid.agcensus_2022'
table_w_category_county_level='nssac_agaid.agcensus_2022_labor_w_category_county_level'
table_only_county_level='nssac_agaid.agcensus_2022_labor_only_county_total'
table_only_state_level='nssac_agaid.agcensus_2022_labor_only_state_total'

psql -e -d geodb -f process_labor.sql \
     -v source_table=$source_table -v table_name=$table_w_category_county_level

psql -e -d geodb -f process_labor_only_county_total.sql \
     -v source_table=$source_table -v table_name=$table_only_county_level

psql -e -d geodb -f process_labor_only_state_total.sql \
     -v source_table=$source_table -v table_name=$table_only_state_level
