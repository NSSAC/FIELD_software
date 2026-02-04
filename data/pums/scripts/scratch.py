

##     query = '''SELECT * FROM us_populations_staging.va_va_household_ver_2_0
## LIMIT 100000
## '''
##     hh = sqlio.read_sql_query(query, conn)
## 
##     query = '''SELECT * FROM us_populations_staging.va_va_person_ver_2_0 WHERE
## sex=1 AND
## LIMIT 100000
## '''
##     df = sqlio.read_sql_query(query, conn)

