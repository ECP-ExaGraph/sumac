#!/bin/ksh

OMP_NUM_THREADS=48
export OMP_NUM_THREADS
echo "**************************"
echo "Threads: $OMP_NUM_THREADS"
echo "**************************"

GOMP_CPU_AFFINITY="0 12 24 36 6 18 30 42 1 13 25 37 7 19 31 43 2 14 26 38 8 20 32 44 3 15 27 39 9 21 33 45 4 16 28 40 10 22 34 46 5 17 29 41 11 23 35 47"
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

    echo "SCALE 24 JOBS"
    numactl --interleave=all ./driverForMatching -infile SnapScale24_ER.rmat   | tee XXL24ER.$d
    numactl --interleave=all ./driverForMatching -infile SnapScale24_Good.rmat | tee XXL24Good.$d
    numactl --interleave=all ./driverForMatching -infile SnapScale24_Bad.rmat  | tee XXL24Bad.$d

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

    echo "SCALE 24 JOBS"
    numactl --interleave=all ./driverForMatching -infile SnapScale24_ER.rmat   | tee XXL24ER.$d
    numactl --interleave=all ./driverForMatching -infile SnapScale24_Good.rmat | tee XXL24Good.$d
    numactl --interleave=all ./driverForMatching -infile SnapScale24_Bad.rmat  | tee XXL24Bad.$d    

done


