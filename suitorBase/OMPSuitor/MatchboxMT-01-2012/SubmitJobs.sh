#!/bin/ksh

#for d in 16 ; do
#for d in 16 8 4 2 1; do
for d in 32 64 128; do
  OMP_NUM_THREADS=$d
  echo "**************************"
  echo $OMP_NUM_THREADS
  echo "**************************" 
  export OMP_NUM_THREADS
 
 ./driverForColoring -infile test_graph_rmat_100K_1M_undir_unwt.rmat | tee ResultsColoringScale24-ER.$d
done

#export SUNW_MP_PROCBIND="0-127"
# ./wls | tee dat.out.$OMP_NUM_THREADS

