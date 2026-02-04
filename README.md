# FIELD_software
FIELD:  High-Resolution, US-scale Digital Similar of Interacting Livestock, Wild Birds, and Human Ecosystems for Multi-host Epidemic Spread

## Code and data organization
The code is organized in folers corresponding to the major modules as
defined in the paper: 
1. Livestock ``./livestock``, 
1. Wild birds ``./wildbirds``, 
1. Human population ``./population``,

In addition, we have a folder for 
1. Processing data ``./data``,
1. Verification and validation ``./vnv``,
1. Risk assessment ``./risk``

We have separate README.md files in each folder describing.

Some of the code was written to be run on a Linux/Unix high-performance
computing cluster with SLURM job scheduling
(https://www.rc.virginia.edu/userinfo/rivanna/overview/). It makes heavy
use of parallelism, with pipeline inputs and outputs spread out across many
files. If a SLURM cluster is available, complete with anaconda (or a
similar python installation) and gurobipy, the code may be run as intended
with slight adjustments.

Assume all commands below are run within a ``work`` directory so that
scripts are not accidentally overwritten or deleted.