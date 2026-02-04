#!/bin/bash
DOC="/Users/abhijin/github/AgAid_digital_twin/notes/figs/"
function get_person_table(){ # to extract person data
    python ../scripts/pums_extract.py
}

function puma_shapes(){ # to extract person data
    python ../scripts/puma_shapefiles_extract.py
}

function spatial(){ # spatial distribution
    python ../scripts/spatial_distribution.py
}

function corr(){ # correlation between variables
    python ../scripts/fields_correlation.py
}

function transfer(){
    cp people_by_puma.pdf $DOC
    cp pums_cramer.pdf $DOC
}

if [[ $# == 0 ]]; then
   echo "Here are the options:"
   grep "^function" $BASH_SOURCE | sed -e 's/function/  /' -e 's/[(){]//g' -e '/IGNORE/d'
else
   eval $1 $2
fi


