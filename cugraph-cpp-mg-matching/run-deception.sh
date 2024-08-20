#!/bin/bash

#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH -A pacer
#SBATCH --constraint=nvlink
#SBATCH -p a100
#SBATCH -J CUGXX
#SBATCH -o CUGXX_%A_%a.out
#SBATCH -e CUGXX_%A_%a.err

source /etc/profile.d/modules.sh

module load gcc/12.2.0
module load openmpi/4.1.4
module load cuda/12.1
module load cmake/3.28.1
module load python/miniconda24.4.0

source /share/apps/python/miniconda24.4.0/etc/profile.d/conda.sh
conda activate cugraph-ldgpu2

export LD_LIBRARY_PATH="/people/ghos167/builds/openmpi-4.1.4-cuda12/lib:/people/ghos167/.conda/envs/cugraph-ldgpu2/lib:$LD_LIBRARY_PATH"
#export NCCL_DEBUG=TRACE

ulimit -a

export BIN_PATH="$HOME/proj/cugraph-maximal-matching/updated/matching/test"
export INP_PATH="$HOME/mmio-files/mycielskian16/mycielskian16.mtx"

echo "Multi-GPU cuGraph C++ with MPI"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#nvidia-smi -L
/people/ghos167/builds/openmpi-4.1.4-cuda12/bin/mpirun -np 8 $BIN_PATH/./mg_matching_test $INP_PATH
