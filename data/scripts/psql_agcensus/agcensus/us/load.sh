# a simple script used to download data and load it to a central table $table_name
# PGUSER and PGPASSWORD should be set
#
wget https://www.nass.usda.gov/datasets/qs.census2022.txt.gz
gunzip qs.census2022.txt.gz
connection_str="-d geodb"
table_name="nssac_agaid.agcensus_2022"
file_with_path="./qs.census2022.txt"
psql -e $connection_str -f create_table.sql -v table_name=$table_name
psql -e $connection_str -c "\copy $table_name from $file_with_path with delimiter E'\t' CSV HEADER"
