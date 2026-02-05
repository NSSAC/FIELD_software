# Livestock Layer

All code is in the ``./scripts`` folder. The ``./results`` folder contains
the output of the construction process.  All the modules involving integer
linear programming (ILP) were run on a HPC cluster. There are some scripts
for verification and analysis as well.

## Farms to cells assessment

``master``: This file provides contains all the functions that were run in
the order in which they are organized in the pipeline. It also has some
example instances for testing.

``fill_gaps.py``: Filling gaps in AgCensus data at the county and state level.
See ``master`` for details.

``generate_instances.py``: When run, it generates sbatch commands for each
instance of farms to cell assignments. This will create ``./run.sh``, which
needs to be executed. It will spawn multiple jobs on the cluster. See
``master`` for how to run.

``collect_results.py``: Collects results from all the spawned jobs and
consolidates them.

``process_agcensus.py``: Processing AgCensus data to extract farm counts by

``farms_to_cells.py``: Assigns farms to grid cells. ``generate_instances.py``
creates instances for this script.

``run_proc.sbatch``: This is the sbatch template file that is used for each
instance of ``farms_to_cells.py``. It is called by ``generate_instances.py``.

``verify.py``: Verifying the results of the farms to cells assignment.

## Visualization

``birds_3d.py``, ``livestock_3d.py``, ``population_3d.py``: These scripts
visualize the bird, livestock, and human population layers respectively in
3D.

``county_dist.py``: Various visualizations of the distribution of farms across counties.

``state_dist.py``: Various visualizations of the distribution of farms across states.

## Results
This folder contains the output of the construction process, some
statistics, and plots.
