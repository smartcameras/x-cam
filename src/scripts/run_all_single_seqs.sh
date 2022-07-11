#!/bin/bash


METHODS=(0 1 2) # 0: DBoW, 1: NetVLAD, 2: DeepBit
DIMS=(128 128 4)

for a in 0
do

gf_type=${METHODS[$a]}         # 0: DBoW, 1: NetVLAD, 2: DeepBit
gf_dim=${DIMS[$a]}       # 128: DBoW (number not used), 128: NetVLAD, 4: DeepBit

FEATPATH=../feats # not used by DBoW


	for R in 1
	do
		# matching_th={-1:no_threshold, }
		# b_snn={0:no_snn,1:use_snn}
		# snn_th={0.6,0.8}
		#R=1

		max_num_kps=1000
		num_max_frames=-1
		matching_th=50
		b_snn=1
		snn_th=0.6

		COLLAB_MODE=0
		INIT_WND=30
		FEAT_SHARING_FREQ=12
		SYNC_MODE=1

			#############################################################################################
			DATASET=gate
			ncams=4

			source script/run_single_mode.sh $DATASET $ncams $R 1 0 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 0 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE
			source script/run_single_mode.sh $DATASET $ncams $R 2 0 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 1 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE
			source script/run_single_mode.sh $DATASET $ncams $R 3 0 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 2 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE
			source script/run_single_mode.sh $DATASET $ncams $R 4 0 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 3 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE
			

			##############################################################################################
			DATASET=office
			ncams=3

			# source script/run_single_mode.sh $DATASET $ncams $R 1 2 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 0 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE
			# source script/run_single_mode.sh $DATASET $ncams $R 1 3 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 1 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE
			# source script/run_single_mode.sh $DATASET $ncams $R 2 3 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 2 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE

			#############################################################################################
			DATASET=backyard
			ncams=4

			# #source script/run_single_mode.sh $DATASET $ncams $R 1 2 $num_max_frames $matching_th $b_snn $snn_th 0 $gf_type $gf_dim $FEATPATH
			# #source script/run_single_mode.sh $DATASET $ncams $R 1 3 $num_max_frames $matching_th $b_snn $snn_th 1 $gf_type $gf_dim $FEATPATH
			# source script/run_single_mode.sh $DATASET $ncams $R 1 4 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 2 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE
			# source script/run_single_mode.sh $DATASET $ncams $R 2 3 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 3 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE
			# #source script/run_single_mode.sh $DATASET $ncams $R 2 4 $num_max_frames $matching_th $b_snn $snn_th 4 $gf_type $gf_dim $FEATPATH
			# #source script/run_single_mode.sh $DATASET $ncams $R 3 4 $num_max_frames $matching_th $b_snn $snn_th 5 $gf_type $gf_dim $FEATPATH

			#############################################################################################
			DATASET=courtyard
			ncams=4

			# source script/run_single_mode.sh $DATASET $ncams $R 1 2 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 0 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE
			# source script/run_single_mode.sh $DATASET $ncams $R 1 3 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 1 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE
			# source script/run_single_mode.sh $DATASET $ncams $R 1 4 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 2 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE
			# source script/run_single_mode.sh $DATASET $ncams $R 2 3 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 3 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE
			# source script/run_single_mode.sh $DATASET $ncams $R 2 4 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 4 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE
			# source script/run_single_mode.sh $DATASET $ncams $R 3 4 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 5 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE

			#############################################################################################
			# Loop Closure Detection sequences

			# source script/run_single_mode_LCD.sh SVS 1 $R 1 0 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 0 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE
			# source script/run_single_mode_LCD.sh K00 1 $R 1 0 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 0 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE
			# source script/run_single_mode_LCD.sh MLG 1 $R 1 0 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 0 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE

	done
done

echo "Finished all!"

