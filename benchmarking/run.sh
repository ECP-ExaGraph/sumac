#!/bin/csh
#SBATCH -A pacer
#SBATCH -t 03:00
#SBATCH -N 1
#SBATCH -J gpuCommBW
#SBATCH -o res.txt
#SBATCH --partition a100_shared

source /etc/profile.d/modules.csh
module purge
module load cuda
./gpuBW
