#!/bin/bash
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH -A pacer
#SBATCH -p a100
#SBATCH --nodelist=a100-06
#SBATCH -J sumac
#SBATCH -o friendsterScalePCIe.out
#SBATCH -e friendsterScalePCIe.err
source /etc/profile.d/modules.sh

module load cuda/10.1.243
module load python/miniconda4.12
source /share/apps/python/miniconda4.12/etc/profile.d/conda.sh
conda activate gpuload

export LD_LIBRARY_PATH=/people/mand884/.conda/envs/gpuload/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/share/apps/cuda/10.1.243/lib64:$LD_LIBRARY_PATH
export PATH=$PATH:/share/apps/cuda/10.1.243/include
export PATH=$PATH:/people/mand884/.conda/envs/gpuload/include

ulimit -a
echo "Pointer Chasing"



gfile="/qfs/projects/pacer/sumacData/com-Friendster-r.bin"

#/people/mand884/sumac/matchingBase/pointerChase/dualLoadReduce/./redPCMatch1 ${gfile} 3 1
#/people/mand884/sumac/matchingBase/pointerChase/dualLoadReduce/./redPCMatch2 ${gfile} 2 1
for i in {3..8}; do
    echo "GPU ${i}"
    /people/mand884/sumac/matchingBase/pointerChase/dualLoadReduce/./redPCMatch${i} ${gfile} 1 1
done