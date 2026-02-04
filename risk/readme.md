# Risk analysis
Current work is on risk analysis regarding dairy cattle. Folder
organization is as follows: ``scripts/`` holds all code, ``results``
contains output files such as result dataframes and plots, and
``intermediate_data`` contains processed data that will be used by the
scripts for analysis. ``scripts/master`` contains the usage of some
scripts. The main steps are as follows:
1. Restructuring data such that all the independent variables are available
   at the grid level: ``x,y,variable1,variable2,...``. See
   ``prep_layers.py`` and ``neighborhood_graph.py``.
1. Computing dairy risk. ``risk_dairy.py`` computes risk given model
   parameter values. HPC related files where the full factorial design is
   implemented are ``generate_instances_dairy.py``, ``collect_dairy.py``,
   ``run_arr.sbatch``, ``qreg``.
1. Evaluation of models by comparing with H5N1 incidence data and
   subsequent clustering-based analysis of the phase space.
   ``process_results_dairy.py``.
1. Plot results. ``plot_dairy.py``
