# Risk analysis
Folder organization is as follows: ``scripts/`` holds all code, ``results``
contains output files such as result dataframes and plots, and
``intermediate_data`` contains processed data that will be used by the
scripts for analysis.

* ``master`` is the shell script for invoking the analysis pipeline. It contains the usage of most scripts.
* ``risk.py`` is the main script for training models and computing risk.
* ``utils.py`` contains utility functions for data processing and analysis.
* ``analysis.py`` is the main script for generating all the plots, H5N1 incidence analysis
* ``hmm_plot.py`` plots the HMM models.
* ``plot.py``, ``graph_to_tikz.py``, and ``loader.py``, and  are utility scripts for plotting.
* ``human_risk.py`` analysis of risk to humans.
* ``prep_layers.py`` and ``neighborhood_graph.py``: Restructuring data such that all the independent variables are available
   at the grid level: ``x,y,variable1,variable2,...`` for downstream analysis. 