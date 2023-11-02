#!/bin/bash
#COBALT -n 1 -t 2:00:00 -q full-node -A GRACE
module load nccl/nccl-v2.17.11-1_CUDA11.4 
export LD_LIBRARY_PATH="/lus/theta-fs0/software/thetagpu/nccl/nccl-v2.17.1-1_gcc-9.4.0-1ubuntu1-20.04/lib"
export NCCL_HOME="/lus/theta-fs0/software/thetagpu/nccl/nccl-v2.17.1-1_gcc-9.4.0-1ubuntu1-20.04/lib"


/home/mandum/sumac/matchingBase/pointerChase/dualLoadReduce/./redPCMatch8 /eagle/GRACE/sumacData/AGATHA_2015.bin 10 1