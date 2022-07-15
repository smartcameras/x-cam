#!/bin/bash

# source activate C3OD
# C3OD_DIR=$HOME/Desktop/C3OD

source activate Superglue
C3OD_DIR=/import/smartcameras-002/alessio/C3OD

DATAPATH=$C3OD_DIR/data/
RESPATH=$C3OD_DIR/20220628_results/

DATASET=gate
METHOD=dbow # dbow, deepbit, netvlad
MATCHING_MODE=local #local, global

OVERLAP_TH=50
N_RUNS=1

############################################


# for METHOD in dbow deepbit netvlad
for METHOD in superglue superpoint
do
  echo $METHOD

  for DATASET in gate office backyard courtyard
  # for DATASET in gate office
  do
    echo $DATASET

    python plot_feature_matches.py --respath $RESPATH --method $METHOD --datapath $DATAPATH --overlap_th $OVERLAP_TH --dataset $DATASET --n_runs $N_RUNS --matching_mode $MATCHING_MODE
  done    
done

echo "Finished!"

conda deactivate