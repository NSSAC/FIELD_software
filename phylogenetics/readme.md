# Phylogenetic analysis
Cross-species link calculation and figure generation. Folder
organization is as follows: ``scripts/`` holds all code, ``results``
contains output files such as trees and plots, and
``intermediate_data`` contains processed data that will be used by the
scripts for analysis. The main steps are as follows:
1. Parsing risk; combining data from BV-BRC and nextstrain; normalizing host data; and partitioningg records into B3.13 and non-B3.13
2. Parsing NextStrain trees calculated from step 1 and computing cross-species links.
3. Comparing state-month risk to cross-species link count.
