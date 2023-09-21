#!/bin/bash -l

#SBATCH -N 1

export OMP_NUM_THREADS=192
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

echo "*******************************"
echo "Affinity: OMP_PROC_BIND=spread"
echo "*******************************"

for m in *.bin ; do
    numactl --interleave=all ./driverForGraphClustering -f 9 -s $m |tee $m_OutputSpread_Map.txt
    numactl --interleave=all ./driverForGraphClustering -f 9 -s -c 1 $m |tee $m_OutputSpread_Coloring1_Map.txt
    numactl --interleave=all ./driverForGraphClustering -f 9 -s -c 1 -b 1 $m |tee $m_OutputSpread_Coloring1_NoMap.txt
done

export OMP_PROC_BIND=close

echo "*******************************"
echo "Affinity: OMP_PROC_BIND=close"
echo "*******************************"

for m in *.bin ; do
    numactl --interleave=all ./driverForGraphClustering -f 9 -s $m |tee $m_OutputClose_Map.txt
    numactl --interleave=all ./driverForGraphClustering -f 9 -s -c 1 $m |tee $m_OutputClose_Coloring1_Map.txt
    numactl --interleave=all ./driverForGraphClustering -f 9 -s -c 1 -b 1 $m |tee $m_OutputClose_Coloring1_NoMap.txt
done
