#!/bin/ksh

OMP_NUM_THREADS=48
export OMP_NUM_THREADS
echo "**************************"
echo "Threads: $OMP_NUM_THREADS"
echo "**************************"

GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
                   24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 393 40 41 42 43 44 45 46 47"
export GOMP_CPU_AFFINITY
echo "**************************"
echo "Affinity: $GOMP_CPU_AFFINITY"
echo "**************************" 

for d in 48 32 16 8 4 2 1; do
    
    OMP_NUM_THREADS=$d
    echo "**************************"
    echo $OMP_NUM_THREADS
    echo "**************************" 
    export OMP_NUM_THREADS

    numactl --interleave=all ./driverForMatchingNewD ../Matrices/mrprob.mtx 

done
