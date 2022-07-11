#!/bin/bash

## List of scenarios (num of cameras)
# office (3)
# courtyard (4)
# backyard (4)
# gate (4)
# SCENARIOS=(office courtyard backyard gate)
SCENARIOS=(office courtyard)

## List of methods
# DBOW
# LiST
# BTST
# TTST
# METHODS=(DBoW LiST BTST TTST)
# METHODS=(deepbit netvlad)
#METHODS=(deepbit netvlad DBoW LiST BTST TTST)
METHODS=(dbow deepbit netvlad )


## Number of runs
R=30

for METHOD in ${METHODS[*]}
do
  echo $METHOD

  for SCENARIO in ${SCENARIOS[*]}
  do
    echo $SCENARIO

    if [ $SCENARIO == 'office' ]
    then
      ncams=3
    else
      ncams=4
    fi

    for v1 in $(seq 1 1 $(($ncams - 1)))
    do
      a=$(($v1 + 1))
      for v2 in $(seq $a 1 $ncams)
      do
        echo $v1 "vs" $v2
        # DATAPATH=../20220621_results_intwnd/${SCENARIO}_${v1}vs${v2}/$METHOD
        # DATAPATH=../20220621_results_rate_frequency/${SCENARIO}_${v1}vs${v2}/$METHOD

        DATAPATH=../20220628_results/${SCENARIO}_${v1}vs${v2}/$METHOD

        # for f in 5 7 10 12 15 17 20 # SHARING FREQUENCY
        # for f in 5 10 15 20 25 30 35 40 45 50 75 # INITWND
        # for f in 1 5 10 15 25 30 # RATE FREQUENCY
        for f in 5
        do
        
          for r in {1..30}
          do
            # rm -r $DATAPATH/INIT_WND$f/run$r
            # rm -r $DATAPATH/RATE$f/run$r
            # rm -r $DATAPATH/FREQ$f/run$r
            rm -r $DATAPATH/run$r
          done
        done
      done
    done

  done
done
