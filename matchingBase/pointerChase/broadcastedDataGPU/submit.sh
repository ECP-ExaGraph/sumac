#!/bin/csh
#COBALT -n 1 -t 2:00:00 -q full-node -A GRACE


(setenv LD_LIBRARY_PATH /lus/theta-fs0/software/thetagpu/nccl/nccl-v2.17.1-1_gcc-9.4.0-1ubuntu1-20.04/lib)
(setenv NCCL_HOME /lus/theta-fs0/software/thetagpu/nccl/nccl-v2.17.1-1_gcc-9.4.0-1ubuntu1-20.04/lib)
/home/mandum/sumac/matchingBase/pointerChase/broadcastedDataGPU/./pcMatchGPU /eagle/GRACE/sumacData/U1a.bin 