#!/bin/bash

### Read input parameters
DATASET=${1}
ncams=${2}
R=${3}
v1=${4}
v2=${5}


### Arguments
# 1. agent_id: the unique identifier of the agent/camera (e.g., 1)

# 2. rcv_tcp_port: the port number to set in ZMQ for the receiver (request-reply mode). Default: 5555.
# 3. snd_tcp_port: the port number to set in ZMQ for the sender. Default: 5556.
ID1=(5551 5553 5555 5557 5559 5561)
ID2=(5552 5554 5556 5558 5560 5562)

j=${11}

# 4. videopath: . Default: "".
VIDEOPATH=/home/$USER/Desktop/AVIS/Datasets/M3CAM-2.0/$DATASET/images
VIDEOPATH=/home/$USER/Desktop/C3OD/data/$DATASET/images
#
# 5. respath: . Default: "../results/";
#
# 6. strVocFile: . Default: "./Vocabulary/ORBvoc.txt";
ORBVOC=Vocabulary/ORBvoc.txt
#
# 7. num_max_frames: the maximum number of frames to process in a recorded video. Default: -1 (no max number, the whole image sequence is processed).
# 8. max_num_kps: the maximum target number of keypoints that can be localised in an image. Default: 1000.
# 9. matching_th: threshold when matching binary local features. Default: -1. 
# 10. b_snn: boolean to use Lowe's ratio test to filter out ambiguous matches for binary features. The test defines a distance ratio between the closest and the second closest binary feature (or second nearest neighbour, SNN). Deafult: false.
# 11. snn_th: threshold on Lowe's ratio test. Matches whose test is lower than this threshold are discarded. When the threshold is lower (e.g., 0.6), the test is more restrictive, enforcing a larger distance between the first and second closest neighbours for a query binary feature (fewer matches). When the threshold is higher (e.g., 0.8), the test is more permissive, allowing more matches that can be also erroneous. The recommended value is between 0.6 and 0.8. Default: 0.6.
# 12. gf_type: DBoW, NetVLAD, DeepBit. Default: DBoW.
# 13. gf_dim: the dimensionality of the global feature (vector representing the whole image). Default: -1. 
# 14. featpath: . Default: "../feats/backyard/view1/netvlad/netvlad_feats.txt";
# 15. collaborative_mode: . Default: true.
# 16. init_wnd: Number of frames for initialisation window. Default: 30.
# 17. feat_sharing_freq: Frequency of sharing features: Default: 5.
# 18. synch_mode: . Default: false.
# 19. vprcallfilename: . Default: "";
num_max_frames=${6}
max_num_kps=${7}
matching_th=${8}
b_snn=${9}
snn_th=${10}
gf_type=${12} # 0: DBoW, 1: NetVLAD, 2: DeepBit
gf_dim=${13}
FEATPATH=${14} # not used by DBoW
COLLAB_MODE=${15}
INIT_WND=${16}
FEAT_SHARING_FREQ=${17}
SYNC_MODE=${18}

if [ $gf_type -eq 1 ]
then
  RESPATH=/home/$USER/Desktop/C3OD/results/${DATASET}_${v1}/netvlad/
  VPRCALLPATH=/home/$USER/Desktop/C3OD/feat/${DATASET}_${v1}/BTST/run$R
elif [ $gf_type -eq 2 ]
then 
  RESPATH=/home/$USER/Desktop/C3OD/results/${DATASET}_${v1}/deepbit/
  VPRCALLPATH=/home/$USER/Desktop/C3OD/feat/${DATASET}_${v1}/BTST/run$R
else
  RESPATH=/home/$USER/Desktop/C3OD/results/${DATASET}_${v1}/dbow/
  VPRCALLPATH=/home/$USER/Desktop/C3OD/feat/${DATASET}_${v1}vs${v2}/BTST/run$R
fi

mkdir -p $RESPATH
echo $RESPATH

SEQ1=$VIDEOPATH/view$v1/rgb/
FEATPATH1=$FEATPATH/$DATASET/view$v1
VPRCALLFILE1=$VPRCALLPATH/agent1_vpr_res.txt

./Agent ${v1}  ${ID1[j]} ${ID2[j]} $SEQ1 $RESPATH $ORBVOC $num_max_frames $max_num_kps $matching_th $b_snn $snn_th $gf_type $gf_dim $FEATPATH1 $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $VPRCALLFILE1 2> ${RESPATH}log_agent${v1}_run$R.txt

zip run$R.zip agent*.txt running_times_*.txt
rm agent*.txt running_times_*.txt
mv run$R.zip log_* $RESPATH

sleep 5s

echo 'Finished!'

