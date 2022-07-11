#!/bin/bash

ORBVOC=Vocabulary/ORBvoc.txt
VIDEOPATH=/media/$USER/Elements/AVIS_PhD_Project/Datasets/M3CAM-2.0/office
SEQ1=$VIDEOPATH/view1/rgb/
SEQ2=$VIDEOPATH/view3/rgb/

./Agent 1 5555 5556 $SEQ1 $ORBVOC > agent1.log &
./Agent 2 5556 5555 $SEQ2 $ORBVOC > agent2.log &

wait

echo "Finshed!"


./Agent 1 5555 5556 /media/alessioxompero/Elements/AVIS_PhD_Project/Datasets/M3CAM-2.0/office/view1/rgb/ Vocabulary/ORBvoc.txt
./Agent 2 5556 5555 /media/alessioxompero/Elements/AVIS_PhD_Project/Datasets/M3CAM-2.0/office/view2/rgb/ Vocabulary/ORBvoc.txt


################################################################################
# Gate
ORBVOC=Vocabulary/ORBvoc.txt
VIDEOPATH=/media/$USER/Elements/AVIS_PhD_Project/Datasets/M3CAM-2.0/gate/images
SEQ1=$VIDEOPATH/view3/rgb/
./Agent 1 5555 5556 $SEQ1 $ORBVOC

ORBVOC=Vocabulary/ORBvoc.txt
VIDEOPATH=/media/$USER/Elements/AVIS_PhD_Project/Datasets/M3CAM-2.0/gate/images
SEQ2=$VIDEOPATH/view4/rgb/
./Agent 2 5556 5555 $SEQ2 $ORBVOC


R=3
zip run$R.zip agent*.txt PlaceVoting_*.txt running_times_*.txt
rm agent*.txt PlaceVoting_*.txt running_times_*.txt
mv run$R.zip ../results/gate/3vs4/DBoW2/

################################################################################
# Courtyard
ORBVOC=Vocabulary/ORBvoc.txt
VIDEOPATH=/media/$USER/Elements/AVIS_PhD_Project/Datasets/M3CAM-2.0/courtyard/images
SEQ=$VIDEOPATH/view3/rgb/
./Agent 1 5555 5556 $SEQ $ORBVOC

ORBVOC=Vocabulary/ORBvoc.txt
VIDEOPATH=/media/$USER/Elements/AVIS_PhD_Project/Datasets/M3CAM-2.0/courtyard/images
SEQ=$VIDEOPATH/view3/rgb/
./Agent 2 5556 5555 $SEQ $ORBVOC


R=1
zip run$R.zip agent*.txt PlaceVoting_*.txt running_times_*.txt
rm agent*.txt PlaceVoting_*.txt running_times_*.txt
mv run$R.zip ../results/courtyard/3vs4/DBoW2/
