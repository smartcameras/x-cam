#!/bin/bash

source activate Superglue


### PATHS
C3OD_DIR=/import/smartcameras-002/alessio/C3OD
#C3OD_DIR=$HOME/Desktop/C3OD
DATAPATH=$C3OD_DIR/data/
RESPATH=$C3OD_DIR/20220628_results/

DATASET=gate


for DATASET in gate office backyard courtyard
do
#######################################################################################
# PARAMETERS
#
OVERLAP_TH=50 		# default 50
N_RUNS=30 			# default 30
#
REF_GLOBAL=netvlad  # options: 'dbow','deepbit', 'netvlad'
METHOD=rootsift 		# options: 'dbow','deepbit', 'netvlad','rootsift','superpoint','superglue','netvlad'
MATCHING_MODE=local # deafult local, options: local, global
FREQ=5 				# frequency to share the features, default 5
#
SAC=RANSAC 			# default MAGSAC++, options: RANSAC, MAGSAC++
MIN_INLIERS=15		# default 15
REP_TH=2.0			# default 2.0
CONF=0.99			# default 0.99
MAX_ITERS=500		# default 500
#
MATCHING_STRATEGY=NNDR # NNDR
SNN_TH=0.6			# default 0.6
#
MAX_N_KPS=1000		# default 1000
FEATURE_TYPE=RootSIFT # ['RootSIFT','SIFT','SuperPoint','SuperGlue']
#
#
#######################################################################################
# SuperGlue parameters
#
RESIZE=[640,480]
SUPERGLUE=indoor 	# indoor, outdoor
MAX_KEYPOINTS=1000 	# default 1000
KEYPOINT_TH=0.005 	# default 0.005
NMS_RADIUS=4		# default 4
SINKHORN_ITERS=20 	# default 20
MATCH_TH=0.2 		# default 0.2

if [ $DATASET == 'office' ]
then
	SUPERGLUE=indoor 	# indoor, outdoor
else
	SUPERGLUE=outdoor 	# indoor, outdoor
fi
#
############################################


for METHOD in dbow-m netvlad-m deepbit-m
# for METHOD in rootsift superpoint superglue
do
	SAC=MAGSAC++

	if [ $METHOD == 'dbow' ] | [ $METHOD == 'netvlad' ] | [ $METHOD == 'deepbit' ]
	then 
		REF_GLOBAL=$METHOD
	elif [ $METHOD == 'dbow-m' ]
	then
		REF_GLOBAL=dbow
	elif [ $METHOD == 'netvlad-m' ]
	then
		REF_GLOBAL=netvlad
	elif [ $METHOD == 'deepbit-m' ]
	then
		REF_GLOBAL=deepbit
	else
		REF_GLOBAL=netvlad
	fi


	if [ $METHOD == 'rootsift' ]
	then
		FEATURE_TYPE=RootSIFT 
	elif [ $METHOD == 'superpoint' ]
	then
		FEATURE_TYPE=SuperPoint
	elif [ $METHOD == 'superglue' ]
	then
		FEATURE_TYPE=SuperGlue
	fi

	CUDA_VISIBLE_DEVICES=2
	python run_m3cam2-0.py 							\
	--datapath     			$DATAPATH			\
	--dataset      			$DATASET			\
	--respath      			$RESPATH			\
	--overlap_th   			$OVERLAP_TH			\
	--n_runs       			$N_RUNS				\
	--ref_global			$REF_GLOBAL			\
	--method				$METHOD				\
	--matching_mode			$MATCHING_MODE		\
	--frequency				$FREQ				\
	--SACestimator			$SAC 				\
	--min_num_inliers		$MIN_INLIERS		\
	--ransacReprojThreshold	$REP_TH 			\
	--confidence			$CONF				\
	--maxIters				$MAX_ITERS			\
	--matching_strategy		$MATCHING_STRATEGY	\
	--snn_th				$SNN_TH				\
	--max_n_kps				$MAX_N_KPS			\
	--feature_type			$FEATURE_TYPE		\
	--superglue				$SUPERGLUE			\
	--max_keypoints			$MAX_KEYPOINTS		\
	--keypoint_threshold	$KEYPOINT_TH		\
	--nms_radius			$NMS_RADIUS			\
	--sinkhorn_iterations	$SINKHORN_ITERS		\
	--match_threshold		$MATCH_TH			
	# --force_cpu
	# --resize_float								
	# --resize				$RESIZE				\
	# uncomment force_cpu or resize_float if you want to enforce them 
	# (no value needed since it is boolean)

	done
done

conda deactivate
