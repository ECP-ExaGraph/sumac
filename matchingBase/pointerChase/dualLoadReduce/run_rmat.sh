#!/bin/bash

fileGenType="rmat_er"
basePath="/home/mandum/graphGen/R-MAT-Gen-main/rmat_files/${fileGenType}/exp/exp"

for i in {15..20}
do
    # Construct the file name
    fileName="${fileGenType}_${i}_exp.bin"
    
    # Construct the full path
    fullPath="$basePath/$fileName"
    
    # Run the command 5 times for the current file
    for j in {1..5}
    do
        # Execute the command, appending the output to the results file
        ./redPCMatch4 ${fullPath} 1 1 >> ${fileGenType}_${i}_exp.4.res
        echo "Done ${fullPath} ${j}"
    done
done
