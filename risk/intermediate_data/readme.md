# Risk feature columns

x, y: GLW cell coordinates
lon, lat: EPSG 4326 coordinates
cattle, beef, milk, other, poultry, hogs: counts in (x,y)
cattle_n, beef_n, milk_n, other_n, poultry_n, hogs_n: counts in neighboring
cells weighted by square of distance from the current cell.
state_code, county_code: FIPS codes
birds1, birds2, birds3, birds4: Bird abundance per quarter
pop_ag_112, pop_ag_3115, pop_ag_3116, pop_ag_54194: Ag population
corresponding NAICS codes (112=production, 3115=dairy processing, 3116=meat
processing, 54194=veterinary services).
pop_non_ag: non ag population
pop_ag_112_n, pop_ag_3115_n, pop_ag_3116_n, pop_ag_54194_n, pop_non_ag_n:
Counts in neighboring cells weighted by square of distance from the current
cell.
