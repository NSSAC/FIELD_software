# Population Layer
All code is in the ``./scripts`` folder. Some analysis results are in the ``./results`` folder.

``pop_to_parquet.py``: This script extracts national-scale population from the US population digital similar
hosted by the Biocomplexity Institute, UVA.

``agpop.py``: This script extracts the agricultural population from the US population digital similar.

``master``: This shell script contains the pipeline for worker farm assignment.

``worker_farm_prep_data.py``: This script prepares the data for worker farm assignment. It sets the lower and upper bounds for the number of workers that can be assigned to each farm.

``worker_farm_assign.py``: This script performs the worker farm assignment using integer linear programming. It takes the prepared data and assigns workers to farms while respecting the constraints defined in the previous script.

``worker_farm_assignment_postprocess.py``: This script post-processes the worker farm assignment results by assigning cells and farm types.

Other files like ``geometry.py``, ``timer.py``, and ``run_proc.sbatch`` are utility files used by the main scripts.
