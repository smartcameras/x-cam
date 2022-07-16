#!/bin/bash

source activate Superglue
C3OD_DIR=/import/smartcameras-002/alessio/C3OD

# conda activate C3OD
# C3OD_DIR=$HOME/Desktop/C3OD

RESPATH=$C3OD_DIR/20220628_results/

python3 convert_res2tex.py --fnamein $RESPATH/results.csv  --fnameout $RESPATH/restex.txt

conda deactivate

echo "Finished!"