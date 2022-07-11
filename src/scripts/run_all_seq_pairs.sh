#!/bin/bash


METHODS=(0 1 2) # 0: DBoW, 1: NetVLAD, 2: DeepBit
DIMS=(128 128 4)

for a in 0 2
do

gf_type=${METHODS[$a]}         # 0: DBoW, 1: NetVLAD, 2: DeepBit
gf_dim=${DIMS[$a]}       # 128: DBoW (number not used), 128: NetVLAD, 4: DeepBit

FEATPATH=$HOME/Desktop/XC-PR/feats # not used by DBoW
# FEATPATH=$HOME/Desktop/C3OD/data/feats # not used by DBoW


	# for R in 1 2 3 4 5
	for R in {1..30}
	do
		# matching_th={-1:no_threshold, }
		# b_snn={0:no_snn,1:use_snn}
		# snn_th={0.6,0.8}

		COLLAB_MODE=1
		SYNC_MODE=1

		INIT_WND=30
		num_max_frames=-1

		max_num_kps=1000
		matching_th=50
		b_snn=1
		snn_th=0.6
		
		FEAT_SHARING_FREQ=5
		FREQ=1
		

		# for FEAT_SHARING_FREQ in 5 7 10 12 15 17 20
		# for INIT_WND in 5 10 15 20 25 30 35 40 45 50 75
		# for FREQ in 1 5 10 15 25 30
		# do

			#############################################################################################
			DATASET=gate
			ncams=4
			FREQ=30


			# source script/run_sync_mode.sh $DATASET $ncams $R 1 2 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 0 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $FREQ
			# source script/run_sync_mode.sh $DATASET $ncams $R 1 3 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 1 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $FREQ
			# source script/run_sync_mode.sh $DATASET $ncams $R 1 4 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 2 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $FREQ
			# source script/run_sync_mode.sh $DATASET $ncams $R 2 3 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 3 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $FREQ
			# source script/run_sync_mode.sh $DATASET $ncams $R 2 4 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 4 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $FREQ
			# source script/run_sync_mode.sh $DATASET $ncams $R 3 4 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 5 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $FREQ

			##############################################################################################
			DATASET=office
			ncams=3

			# source script/run_sync_mode.sh $DATASET $ncams $R 1 2 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 0 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $FREQ
			source script/run_sync_mode.sh $DATASET $ncams $R 1 3 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 1 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $FREQ
			source script/run_sync_mode.sh $DATASET $ncams $R 2 3 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 2 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $FREQ

			#############################################################################################
			DATASET=backyard
			ncams=4
			FREQ=10

			# #source script/run_sync_mode.sh $DATASET $ncams $R 1 2 $num_max_frames $matching_th $b_snn $snn_th 0 $gf_type $gf_dim $FEATPATH $FREQ
			# #source script/run_sync_mode.sh $DATASET $ncams $R 1 3 $num_max_frames $matching_th $b_snn $snn_th 1 $gf_type $gf_dim $FEATPATH $FREQ
			# source script/run_sync_mode.sh $DATASET $ncams $R 1 4 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 2 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $FREQ
			# source script/run_sync_mode.sh $DATASET $ncams $R 2 3 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 3 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $FREQ
			# #source script/run_sync_mode.sh $DATASET $ncams $R 2 4 $num_max_frames $matching_th $b_snn $snn_th 4 $gf_type $gf_dim $FEATPATH $FREQ
			# #source script/run_sync_mode.sh $DATASET $ncams $R 3 4 $num_max_frames $matching_th $b_snn $snn_th 5 $gf_type $gf_dim $FEATPATH $FREQ

			#############################################################################################
			DATASET=courtyard
			ncams=4
			FREQ=25

			# source script/run_sync_mode.sh $DATASET $ncams $R 1 2 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 0 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $FREQ
			# source script/run_sync_mode.sh $DATASET $ncams $R 1 3 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 1 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $FREQ
			# source script/run_sync_mode.sh $DATASET $ncams $R 1 4 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 2 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $FREQ
			# source script/run_sync_mode.sh $DATASET $ncams $R 2 3 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 3 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $FREQ
			# source script/run_sync_mode.sh $DATASET $ncams $R 2 4 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 4 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $FREQ
			# source script/run_sync_mode.sh $DATASET $ncams $R 3 4 $num_max_frames $max_num_kps $matching_th $b_snn $snn_th 5 $gf_type $gf_dim $FEATPATH $COLLAB_MODE $INIT_WND $FEAT_SHARING_FREQ $SYNC_MODE $FREQ

		# done
	done
done

echo "Finished all!"

