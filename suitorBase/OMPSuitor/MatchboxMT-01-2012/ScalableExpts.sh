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

for d in 48 32 16; do
    
    OMP_NUM_THREADS=$d
    echo "**************************"
    echo $OMP_NUM_THREADS
    echo "**************************" 
    export OMP_NUM_THREADS

    echo "SCALE 27 JOBS"
    numactl --interleave=all ./driverForMatching -infile SnapScale27_ER.rmat   | tee XXL27ER.$d
    numactl --interleave=all ./driverForMatching -infile SnapScale27_Good.rmat | tee XXL27Good.$d
    numactl --interleave=all ./driverForMatching -infile SnapScale27_Bad.rmat  | tee XXL27Bad.$d

    echo "SCALE 26 JOBS"
    numactl --interleave=all ./driverForMatching -infile SnapScale26_ER.rmat   | tee XXL26ER.$d
    numactl --interleave=all ./driverForMatching -infile SnapScale26_Good.rmat | tee XXL26Good.$d
    numactl --interleave=all ./driverForMatching -infile SnapScale26_Bad.rmat  | tee XXL26Bad.$d

     echo "SCALE 25 JOBS"
    numactl --interleave=all ./driverForMatching -infile SnapScale25_ER.rmat   | tee XXL25ER.$d
    numactl --interleave=all ./driverForMatching -infile SnapScale25_Good.rmat | tee XXL25Good.$d
    numactl --interleave=all ./driverForMatching -infile SnapScale25_Bad.rmat  | tee XXL25Bad.$d

done

for d in 8 4 2 1; do
    
    OMP_NUM_THREADS=$d
    echo "**************************"
    echo $OMP_NUM_THREADS
    echo "**************************" 
    export OMP_NUM_THREADS

    echo "SCALE 27 JOBS"
    numactl --interleave=all ./driverForMatching -infile SnapScale27_ER.rmat   | tee XXL27ER.$d
    numactl --interleave=all ./driverForMatching -infile SnapScale27_Good.rmat | tee XXL27Good.$d
    numactl --interleave=all ./driverForMatching -infile SnapScale27_Bad.rmat  | tee XXL27Bad.$d

    echo "SCALE 26 JOBS"
    numactl --interleave=all ./driverForMatching -infile SnapScale26_ER.rmat   | tee XXL26ER.$d
    numactl --interleave=all ./driverForMatching -infile SnapScale26_Good.rmat | tee XXL26Good.$d
    numactl --interleave=all ./driverForMatching -infile SnapScale26_Bad.rmat  | tee XXL26Bad.$d

     echo "SCALE 25 JOBS"
    numactl --interleave=all ./driverForMatching -infile SnapScale25_ER.rmat   | tee XXL25ER.$d
    numactl --interleave=all ./driverForMatching -infile SnapScale25_Good.rmat | tee XXL25Good.$d
    numactl --interleave=all ./driverForMatching -infile SnapScale25_Bad.rmat  | tee XXL25Bad.$d

done


