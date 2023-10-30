#!/bin/bash
#COBALT -n 1 -t 8:00:00 -q full-node -A GRACE



module load nccl/nccl-v2.17.11-1_CUDA11.4 
export LD_LIBRARY_PATH="/lus/theta-fs0/software/thetagpu/nccl/nccl-v2.17.1-1_gcc-9.4.0-1ubuntu1-20.04/lib"
export NCCL_HOME="/lus/theta-fs0/software/thetagpu/nccl/nccl-v2.17.1-1_gcc-9.4.0-1ubuntu1-20.04/lib"


#prog_path="/home/mandum/sumac/matchingBase/pointerChase/broadcastedDataGPU/"
#inp_file="/eagle/GRACE/sumacData/U1a.bin"
prog_path="/home/mandum/PNNLwork/sumac/matchingBase/pointerChase/broadcastedDataGPU/"
inp_file="/data/graphs/U1a.bin"

X_start=4
X_end=4
Y_start=1
Y_end=1
num_runs=1
batch_counts=(1 5)

# Loop over the values of X and Y
for X in $(seq $X_start $X_end); do
    for Y in $(seq $Y_start $Y_end); do
        for batchN in "${batch_counts[@]}"; do
            i=1
            while [ $i -le ${num_runs} ]; do
                echo "Run ${i} of NGPU=${X} with Edge Balance: ${Y}"
                ${prog_path}./redPCMatch${X} ${inp_file} ${batchN} ${Y}
                i=$((i+1))
            done
        done
    done
done