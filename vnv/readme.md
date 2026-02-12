# Verification and Validation of FIELD compmonents
This folder contains code for V&V of the livestock and population layers.
For verification, it includes basic checks of counts and comparisons with
source datasets at different administrative levels. For validation,
independent datasets on livestock locations and labor data are applied.

# V&V of the livestock layer
* ``master``: A script that contains all the calls to the relevant functions.
* ``checks_farms_to_cells.py``: Basic checks on head counts and number of farms.
* ``glw_agcensus.py``: A script to plot analysis of the GLW and AgCensus datasets.
* ``livestock_cafo_analyze.py``: A script to analyze the CAFO locations dataset and compare it with the FIELD livestock layer.
* ``livestock_cafo_match.py``: A script to map the CAFO locations to the FIELD livestock layer. 
* ``livestock_farms_to_cells.py``: Plotting the statistics of farm to cell mapping.
* ``livestock_table_gen.py``: Generating tables for livestock data.
* ``pop.py``: Comparison of FIELD human population with BLS.
* ``worker_farm_analysis.py``: Analysis of worker farm assignment.

