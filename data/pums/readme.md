# Analysis of PUMS as candidate for digital twin

## Preliminaries
``PROJ`` is this repository.

## Installation and setting up environment
### Postgis
Will require postgis package psycopg2.
```
conda install psycopg2
```

### Setting up geopandas
See https://anaconda.org/conda-forge/geopandas
```
conda create -n geo anaconda
conda activate geo
conda install -c conda-forge geopandas
```

## Organization
``scripts/`` contains all code.</br>
``work/`` is where all scripts are executed. No content in this folder is
committed to the repository.</br>
``data/`` contains data downloaded for this subproject.</br>
``results/`` contains analysis results.

## Mode of operation
1. ``cd work``
1. ``. .env.sh`` to setup paths. Now every executable in ``scripts/`` is
   available for execution.

<mark>Do not run scripts in ``scripts/`` folder.</mark>

## Description
* ``master.sh`` contains usage and pipeline of all other scripts.
* Extract PUMS person table from ``geodb`` using script ``extract_pums.py``.
* Shapefiles of PUMA for specific areas are extracted from shapefile
  downloaded from website. These are stored in ``../data/``.
* Analysis
    * ``spatial_distribution.py`` plots distribution of agriculture-linked
      people by PUMA.
