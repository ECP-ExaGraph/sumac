#include <omp.h>
#include <cstdlib>
#include <algorithm>
#include <cstdio>
#include <cassert>
#include <cstring>

#include "graph.h"
#include "io.h"
#include "rand.h"
#include "util.h"

#define _mask 0xFFFFFFFF

__device__ void removeEdges(int vid, int* neighbors, int deg, volatile double** weights, volatile int* mate){
    int lane_id = threadIdx.x % warpSize;
    for(int i=lane_id;i<deg;i+=warpSize){
        int currNeighbor = neighbors[i];
        if(weights[vid][i]==-1)
            continue;
        if(mate[currNeighbor]!=-1){
            weights[vid][i] = -1.0;
        }
    }
}   

__device__ void findEdge(int vid, int* neighbors, int deg, volatile double* vertWeights, volatile int* pointers){
    int candidate = -1;
    double heaviest = -1.0;
    int lane_id = threadIdx.x % warpSize;
    for(int i=lane_id;i<deg;i+=warpSize){
        int currNeighbor = neighbors[i];
        double currWeight = vertWeights[i];
        if(currWeight<=0)
            continue;
        if((currWeight < heaviest))
            continue;
        if((currWeight == heaviest) && (currNeighbor < candidate))
            continue;
        if(currNeighbor == vid)
            continue;
        heaviest = currWeight;
        candidate = currNeighbor;
    }
    for (int i = warpSize/2;i>=1;i/=2) {
        double reduceWeight = __shfl_xor_sync(_mask, heaviest, i, warpSize);
        int reduceVert = __shfl_xor_sync(_mask, candidate, i, warpSize);
        if (reduceWeight > heaviest) {
            heaviest = reduceWeight;
            candidate = reduceVert;
        } else if (reduceWeight == heaviest) { 
            if (reduceVert > candidate) {
                heaviest = reduceWeight;
                candidate = reduceVert;
            }
        }
    }
    /*
    if(threadIdx.x == 0)
        printf("%d - %d\n",vid,candidate);
        */
    pointers[vid] = candidate;
}

__global__ void Pointer_Chase_GPU(graph* g, int* verts, volatile double** weights, volatile int* mate, volatile int* pointers, int vertsPerWarp){
    extern __shared__ int smem[];
    int gt_id = blockIdx.x * blockDim.x + threadIdx.x; 
    int w_id = threadIdx.x/warpSize;
    int lane_id = threadIdx.x % warpSize;
    int b_id = blockIdx.x;
    int block_size = blockDim.x;
    int gw_id = w_id + (b_id * (block_size/warpSize));

    int newMatchCount = 0;
    //Fill local shared memory with vertex indices
    for(int i=lane_id;i<vertsPerWarp;i+=warpSize){
        if(i + (gw_id * vertsPerWarp) >= g->num_verts)
            break;
        smem[i + (w_id*vertsPerWarp)] = verts[i + (gw_id * vertsPerWarp)];
        mate[i + (w_id*vertsPerWarp)] = verts[i + (gw_id * vertsPerWarp)];
    }
    volatile int* localMem = &smem[w_id * vertsPerWarp];
    

    int changes = 0;
    do{
        changes = 0;
        for(int i=0;i<vertsPerWarp;i++){
            if(localMem[i] == -1)
                continue;
            int* neighborList = &g->out_adjlist[g->out_offsets[localMem[i]]];
            int deg = g->out_offsets[localMem[i]+1] - g->out_offsets[localMem[i]];
            findEdge(localMem[i],neighborList,deg,weights[localMem[i]],pointers);
        }
        __syncthreads();
        //Keep matching pairs that are mutual, reset pointers
        if(lane_id < vertsPerWarp){
            int vid = localMem[lane_id];
            if(vid != -1){
                if(vid == pointers[pointers[vid]] && vid!=pointers[vid]){
                    mate[vid] = pointers[vid];
                    localMem[lane_id] = -1;
                    atomicAdd_system(&changes,1);
                }
                else{
                    pointers[vid] = -1;
                    mate[vid] = -1;
                }
            }
        }
        __syncthreads();
        //Remove Edges (need to optimize - just do it thru GPUGraph later so I don't have to optimize off this infrastructure)
        
        for(int i=0;i<vertsPerWarp;i++){
            if(localMem[i] == -1)
                continue;
            int* neighborList = &g->out_adjlist[g->out_offsets[localMem[i]]];
            int deg = g->out_offsets[localMem[i]+1] - g->out_offsets[localMem[i]];
            removeEdges(localMem[i],neighborList,deg,weights,mate);
        }
        __syncthreads();
        
    }while(changes!=0);
}