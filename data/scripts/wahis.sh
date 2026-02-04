#!/bin/bash
mkdir -p temp_wahis
cd temp_wahis
pdftohtml -c -hidden ../../h5n1/wahis_h5n1_4451.pdf temp.html
