#!/bin/bash
#COBALT -n 1 -t 01:00:00 -q full-node -A GRACE -o FriendsterBatch.out
module load nccl/nccl-v2.17.11-1_CUDA11.4 
export LD_LIBRARY_PATH="/lus/theta-fs0/software/thetagpu/nccl/nccl-v2.17.1-1_gcc-9.4.0-1ubuntu1-20.04/lib"
export NCCL_HOME="/lus/theta-fs0/software/thetagpu/nccl/nccl-v2.17.1-1_gcc-9.4.0-1ubuntu1-20.04/lib"

/home/mandum/sumac/matchingBase/pointerChase/dualLoadReduce/./redPCMatch1 /eagle/GRACE/sumacData/com-Friendster-r.bin 1 1
/home/mandum/sumac/matchingBase/pointerChase/dualLoadReduce/./redPCMatch1 /eagle/GRACE/sumacData/com-Friendster-r.bin 5 1
/home/mandum/sumac/matchingBase/pointerChase/dualLoadReduce/./redPCMatch1 /eagle/GRACE/sumacData/com-Friendster-r.bin 10 1
/home/mandum/sumac/matchingBase/pointerChase/dualLoadReduce/./redPCMatch1 /eagle/GRACE/sumacData/com-Friendster-r.bin 20 1
/home/mandum/sumac/matchingBase/pointerChase/dualLoadReduce/./redPCMatch1 /eagle/GRACE/sumacData/com-Friendster-r.bin 50 1
#/home/mandum/sumac/matchingBase/pointerChase/dualLoadReduce/./redPCMatch2 /eagle/GRACE/sumacData/com-Friendster-r.bin 3 1
#/home/mandum/sumac/matchingBase/pointerChase/dualLoadReduce/./redPCMatch3 /eagle/GRACE/sumacData/com-Friendster-r.bin 2 1
#/home/mandum/sumac/matchingBase/pointerChase/dualLoadReduce/./redPCMatch4 /eagle/GRACE/sumacData/com-Friendster-r.bin 1 1
#/home/mandum/sumac/matchingBase/pointerChase/dualLoadReduce/./redPCMatch5 /eagle/GRACE/sumacData/com-Friendster-r.bin 1 1
#/home/mandum/sumac/matchingBase/pointerChase/dualLoadReduce/./redPCMatch6 /eagle/GRACE/sumacData/com-Friendster-r.bin 1 1
#/home/mandum/sumac/matchingBase/pointerChase/dualLoadReduce/./redPCMatch7 /eagle/GRACE/sumacData/com-Friendster-r.bin 1 1
#/home/mandum/sumac/matchingBase/pointerChase/dualLoadReduce/./redPCMatch8 /eagle/GRACE/sumacData/com-Friendster-r.bin 1 1