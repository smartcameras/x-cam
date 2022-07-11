#!/bin/bash

### Read input parameters
DATASET=gate
ncams=4
R=1
v1=1
v2=2
num_max_frames=-1
matching_th=50
b_snn=1
snn_th=0.8

j=0

gf_type=0 # 0: DBoW, 1: NetVLAD, 2: DeepBit
gf_dim=128
FEATPATH=../feats # not used by DBoW

max_num_kps=1000

COLLAB_MODE=1
INIT_WND=30
FEAT_SHARING_FREQ=5
SYNC_MODE=1

VIDEOPATH=/home/$USER/Desktop/AVIS/Datasets/M3CAM-2.0/$DATASET/images

ORBVOC=Vocabulary/ORBvoc.txt

ID1=(5551 5553 5555 5557 5559 5561)
ID2=(5552 5554 5556 5558 5560 5562)

if [ $gf_type -eq 1 ]
then
  RESPATH=/home/$USER/Desktop/XC-PR/tests/${DATASET}_${v1}vs${v2}/netvlad/
  VPRCALLPATH=//home/$USER/Desktop/XC-PR/feat/${DATASET}_${v1}vs${v2}/BTST/run$R
elif [ $gf_type -eq 2 ]
then 
  RESPATH=/home/$USER/Desktop/XC-PR/tests/${DATASET}_${v1}vs${v2}/deepbit/
  VPRCALLPATH=/home/$USER/Desktop/XC-PR/feat/${DATASET}_${v1}vs${v2}/BTST/run$R
else
  RESPATH=/home/$USER/Desktop/XC-PR/tests/${DATASET}_${v1}vs${v2}/dbow/
  VPRCALLPATH=/home/$USER/Desktop/XC-PR/feat/${DATASET}_${v1}vs${v2}/BTST/run$R
fi

mkdir -p $RESPATH
echo $RESPATH

echo $num_max_frames $matching_th $b_snn $snn_th $gf_type $gf_dim

SEQ2=$VIDEOPATH/view$v2/rgb/
FEATPATH2=$FEATPATH/$DATASET/view$v2
VPRCALLFILE2=$VPRCALLPATH/agent2_vpr_res.txt

SEQ1=$VIDEOPATH/view$v1/rgb/
FEATPATH1=$FEATPATH/$DATASET/view$v1
VPRCALLFILE1=$VPRCALLPATH/agent1_vpr_res.txt

./Agent 1 ${ID1[j]} ${ID2[j]} $SEQ1 $RESPATH $ORBVOC $num_max_frames $max_num_kps $matching_th $b_snn $snn_th $gf_type $gf_dim $FEATPATH1 $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $VPRCALLFILE1 2> ${RESPATH}log_agent1_${v1}vs${v2}_run$R.txt & 

./Agent 2 ${ID2[j]} ${ID1[j]} $SEQ2 $RESPATH $ORBVOC $num_max_frames $max_num_kps $matching_th $b_snn $snn_th $gf_type $gf_dim $FEATPATH2 $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $VPRCALLFILE2 2> ${RESPATH}log_agent2_${v1}vs${v2}_run$R.txt &

wait

zip run$R.zip agent*.txt running_times_*.txt
rm agent*.txt running_times_*.txt
mv run$R.zip log_* $RESPATH

sleep 5s

echo 'Finished!'

