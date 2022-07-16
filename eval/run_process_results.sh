#!/bin/bash

source activate Superglue
C3OD_DIR=/import/smartcameras-002/alessio/C3OD

# conda activate C3OD
# C3OD_DIR=$HOME/Desktop/C3OD

DATAPATH=$C3OD_DIR/data/
RESPATH=$C3OD_DIR/20220628_results/

DATASET=gate
METHOD=dbow # dbow, deepbit, netvlad
MATCHING_MODE=global #local, global

OVERLAP_TH=50
N_RUNS=30

ANALYSIS_MODE='none' # frequency, rate, none

############################################

# python process_results.py --respath $RESPATH --method $METHOD --datapath $DATAPATH --overlap_th $OVERLAP_TH --dataset $DATASET --n_runs $N_RUNS --matching_mode $MATCHING_MODE --sac $SAC

# for METHOD in dbow-m netvlad-m deepbit-m
# for METHOD in dbow netvlad deepbit dbow-m netvlad-m deepbit-m rootsift superpoint superglue
for METHOD in dbow netvlad deepbit 
do
  echo $METHOD

  for DATASET in gate office backyard courtyard
  do
    echo $DATASET

    python process_results.py \
     --datapath       $DATAPATH       \
     --dataset        $DATASET        \
     --respath        $RESPATH        \
     --method         $METHOD         \
     --overlap_th     $OVERLAP_TH     \
     --n_runs         $N_RUNS         \
     --matching_mode  $MATCHING_MODE  \
     --analysis_mode  $ANALYSIS_MODE
  done    
done

echo "Finished!"

conda deactivate