# Data downloading and preprocessing

## Livestock
### AgCensus
The ``./agcensus`` folder contains downloaded raw data, reports, and
processed data.

### GLW
The ``./glw`` folder contains downloaded raw data and processed data. The
processed data decouples geometry related information from tabular
information to make downstream pipelines more efficient.

### Processing Centers
The ``./processing_center`` folder contains downloaded raw data and
processed data for extracting poultry and milk processing centers. It also
contains scripts for processing.

### CAFO
The ``./cafo`` folder contains downloaded raw data and processed data.

## Wild birds
The ``./birds_prevalence`` folder contains downloaded data (``.tif`` format)
and processed file (``bird_h5_prevalence.parquet``).

## Human Population Digital Similar
The ``./population`` folder contains data on NAICS and SOCP codes and
information on workers per farm. The synthetic data from the population
digital twin is very large to be included in the repository. Parts of this
synthetic datasets are available in ``https://va-pgcoe.org/resources``.

### BLS
The ``./bls`` folder contains raw data on agricultural worker from Bureau
of Labor Statistics.

## H5N1
The ``./h5n1`` folder contains downloaded raw data, reports, and
processed data corresponding to incidence reports.

The ``./scripts`` folder has code corresponding to extraction and
processing of the data. Note that some data folders have corresponding
scripts present in the respective folders.
