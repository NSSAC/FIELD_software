# Data downloading and preprocessing
Currently, two datasets are being explored: GLW and AgCensus.

## GLW and AgCensus data from psql as processed by MW and DX
* ``scripts/download_data.py``: Downloads data from ``geodb``. Probably, it
  is better to have separate scripts for each table.
* ``scripts/prep_glw_agcensus.py``: Modifies GLW and AgCensus datasets
  by choosing livestock types and mapping to standard nomenclature.
  
## NWSS data - Bryan and Andrew
Housed in NWSS_wastewater was was gathered from the CDC H5 wastewater site:
https://www.cdc.gov/nwss/rv/wwd-h5.html

Historical data was gathered using the "Internet Archive" snapshots
https://web.archive.org/web/20240829103055/https://www.cdc.gov/nwss/rv/wwd-h5.html

Several snapshots were combined with a simple merging python code to create the full history of detections

This data has a data use agreement (provide citation) that is described at the bottom of this page:
https://data.wastewaterscan.org/about
