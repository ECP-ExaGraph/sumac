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

numactl --interleave=all ./driverForBFS -infile RmatScale24Bad.gr  -graph dimacs 




echo "Contiguous processes"
OMP_NUM_THREADS=8
echo "**************************"
echo $OMP_NUM_THREADS
echo "**************************" 
export OMP_NUM_THREADS


#KMP_AFFINITY="verbose,proclist=[36,37,38,39,42,43,44,45]"
GOMP_CPU_AFFINITY="0 1 2 3 4 6 7 8 9"
export GOMP_CPU_AFFINITY
echo "**************************"
echo $GOMP_CPU_AFFINITY
echo "**************************" 
echo "Memory binding on node 0"

OMP_SCHEDULE=static
export OMP_SCHEDULE
echo $OMP_SCHEDULE
./driverForMatching -infile RmatScale24Bad.gr  -graph dimacs 

echo "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

OMP_SCHEDULE=dynamic
export OMP_SCHEDULE
echo $OMP_SCHEDULE
./driverForMatching -infile RmatScale24Bad.gr  -graph dimacs 

OMP_SCHEDULE="guided,64"
export OMP_SCHEDULE
echo $OMP_SCHEDULE
./driverForMatching -infile RmatScale24Bad.gr  -graph dimacs 


#echo "Binding on node 15"
#numactl -m 15 ./driverForMatching -infile RmatScale24Bad.gr  -graph dimacs 
    

echo "Spreadout processes"
#KMP_AFFINITY="verbose,proclist=[2,8,14,20,26,32,38,44]"
GOMP_CPU_AFFINITY="2 8 14 20 26 32 38 44"
export GOMP_CPU_AFFINITY
echo "**************************"
echo $GOMP_CPU_AFFINITY
echo "**************************" 
numactl --interleave=all ./driverForMatching -infile RmatScale24Bad.gr  -graph dimacs

ON_CORES="0,1"
echo "**************************"
echo "Cores: $ON_CORES"
echo "**************************"
numactl --cpunodebind $ON_CORES ./driverForMatching -infile RmatScale24Bad.gr  -graph dimacs 
