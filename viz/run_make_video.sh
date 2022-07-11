#!/bin/bash

source activate Superglue
# conda activate C3OD

# C3OD_DIR=$HOME/Desktop/C3OD
C3OD_DIR=/import/smartcameras-002/alessio/C3OD
DATAPATH=$C3OD_DIR/data/
RESPATH=$C3OD_DIR/20220628_results/


############################################

# python process_results.py --respath $RESPATH --method $METHOD --datapath $DATAPATH --overlap_th $OVERLAP_TH --dataset $DATASET --n_runs $N_RUNS --matching_mode $MATCHING_MODE --sac $SAC

for METHOD in dbow rootsift superglue
do
  echo $METHOD

  for DATASET in gate office backyard courtyard
  do
    echo $DATASET

    python make_video.py \
     --datapath       $DATAPATH       \
     --dataset        $DATASET        \
     --respath        $RESPATH        \
     --method         $METHOD         
  done    
done

echo "Finished!"

source run_make_video.sh

conda deactivate
