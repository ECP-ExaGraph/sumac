#!/bin/bash
#COBALT -n 1 -t 8:00:00 -q full-node -A GRACE



module load nccl/nccl-v2.17.11-1_CUDA11.4 
export LD_LIBRARY_PATH="/lus/theta-fs0/software/thetagpu/nccl/nccl-v2.17.1-1_gcc-9.4.0-1ubuntu1-20.04/lib"
export NCCL_HOME="/lus/theta-fs0/software/thetagpu/nccl/nccl-v2.17.1-1_gcc-9.4.0-1ubuntu1-20.04/lib"


prog_path="/home/mandum/sumac/matchingBase/pointerChase/broadcastedDataGPU/"
inp_file="/eagle/GRACE/sumacData/U1a.bin"

X_start=1
X_end=1
Y_start=0
Y_end=1
num_runs=5
batch_counts=(5 10)

# Loop over the values of X and Y
for X in $(seq $X_start $X_end); do
    for Y in $(seq $Y_start $Y_end); do
        for batchN in "${batch_counts[@]}"; do
            i=1
            while [ $i -le 5 ]; do
                echo "Run ${i} of NGPU=${X} with Edge Balance: ${Y}"
                ${prog_path}./redPCMatch${X}_${Y} ${inp_file} ${batchN}
                i=$((i+1))
            done
        done
    done
done