#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:8
#SBATCH -A pacer
#SBATCH -p a100_shared
#SBATCH --constraint=nvlink
#SBATCH -J SuMacBW
#SBATCH -o sumacBW.out
#SBATCH -e sumacBW.err
source /etc/profile.d/modules.sh


module load cuda/11.1


ulimit -a

echo "GPU Benchmark"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

./gpuBW