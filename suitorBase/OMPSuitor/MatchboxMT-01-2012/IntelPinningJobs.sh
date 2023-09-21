#!/bin/ksh

echo "One Processor"
for d in 1 2; do
    
    OMP_NUM_THREADS=$d
    echo "**************************"
    echo $OMP_NUM_THREADS
    echo "**************************" 
    export OMP_NUM_THREADS
    KMP_AFFINITY="verbose,proclist=[14]"
    export KMP_AFFINITY
    echo "**************************"
    echo $KMP_AFFINITY
    echo "**************************" 
    
    ./driverForColoring -infile RmatScale24ER.gr   -graph dimacs | tee Pin_Rmat24Er.1.$d
    ./driverForColoring -infile RmatScale24Good.gr -graph dimacs | tee Pin_Rmat24Good.1.$d
    ./driverForColoring -infile RmatScale24Bad.gr  -graph dimacs | tee Pin_Rmat24Bad.1.$d
    
done

echo "Two Processors"
for d in 2 4; do
    
    OMP_NUM_THREADS=$d
    echo "**************************"
    echo $OMP_NUM_THREADS
    echo "**************************" 
    export OMP_NUM_THREADS
    KMP_AFFINITY="verbose,proclist=[14,15]"
    export KMP_AFFINITY
    echo "**************************"
    echo $KMP_AFFINITY
    echo "**************************" 
    
    ./driverForColoring -infile RmatScale24ER.gr   -graph dimacs | tee Pin_Rmat24Er.2.$d
    ./driverForColoring -infile RmatScale24Good.gr -graph dimacs | tee Pin_Rmat24Good.2.$d
    ./driverForColoring -infile RmatScale24Bad.gr  -graph dimacs | tee Pin_Rmat24Bad.2.$d
    
done

echo "Four Processors"
for d in 4 8; do
    
    OMP_NUM_THREADS=$d
    echo "**************************"
    echo $OMP_NUM_THREADS
    echo "**************************" 
    export OMP_NUM_THREADS
    KMP_AFFINITY="verbose,proclist=[14,15,8,9]"
    export KMP_AFFINITY
    echo "**************************"
    echo $KMP_AFFINITY
    echo "**************************" 
    
    ./driverForColoring -infile RmatScale24ER.gr   -graph dimacs | tee Pin_Rmat24Er.4.$d
    ./driverForColoring -infile RmatScale24Good.gr -graph dimacs | tee Pin_Rmat24Good.4.$d
    ./driverForColoring -infile RmatScale24Bad.gr  -graph dimacs | tee Pin_Rmat24Bad.4.$d
    
done

echo "Eight Processors"
for d in 8 16; do
    
    OMP_NUM_THREADS=$d
    echo "**************************"
    echo $OMP_NUM_THREADS
    echo "**************************" 
    export OMP_NUM_THREADS
    KMP_AFFINITY="verbose,proclist=[14,15,8,9,12,13,10,11]"
    export KMP_AFFINITY
    echo "**************************"
    echo $KMP_AFFINITY
    echo "**************************" 
    
    ./driverForColoring -infile RmatScale24ER.gr   -graph dimacs | tee Pin_Rmat24Er.8.$d
    ./driverForColoring -infile RmatScale24Good.gr -graph dimacs | tee Pin_Rmat24Good.8.$d
    ./driverForColoring -infile RmatScale24Bad.gr  -graph dimacs | tee Pin_Rmat24Bad.8.$d
    
done
