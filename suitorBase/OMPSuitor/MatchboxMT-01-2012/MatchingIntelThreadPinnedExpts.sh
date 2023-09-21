#!/bin/ksh

for d in 48 32 16 8 4 2 1; do
    
    OMP_NUM_THREADS=$d
    echo "**************************"
    echo $OMP_NUM_THREADS
    echo "**************************" 
    export OMP_NUM_THREADS

    ./driverForMatching -infile RmatScale24ER.gr   -graph dimacs | tee 02Rmat24Er.$d
    ./driverForMatching -infile RmatScale24Good.gr -graph dimacs | tee 02Rmat24Good.$d
    ./driverForMatching -infile RmatScale24Bad.gr  -graph dimacs | tee 02Rmat24Bad.$d
    
done

