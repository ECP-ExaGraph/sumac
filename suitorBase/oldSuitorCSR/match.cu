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

__device__ int vlock(volatile int* locks, int idx){
    while (atomicCAS((int*) (locks+idx),0,1) != 0){
        __threadfence();
    }
    return 1;
}
__device__ void vunlock(volatile int*  locks, int idx) {
    locks[idx]=0;
    __threadfence();
}

void suitorOMP(graph* g,double** weights,long* mate){

    long current, partner, next;
    double heaviest;
    bool done, newmatch;
    long deg;

    double* ws = new double[g->num_verts];
    
    omp_lock_t *lock = new omp_lock_t[g->num_verts];
    #pragma omp parallel for
    for(long u = 0;u<g->num_verts;u++){
        mate[u] = -1;
        ws[u] = 0;
        omp_init_lock(&(lock[u]));
    }
    printf("Matching Data Set\n");
   //#pragma omp parallel for
    for(long u = 0;u<g->num_verts;u++){
        printf("%ld\n",u);
        current = u;
        done = false;
        while(done == false){
            partner = mate[current];
            heaviest = ws[current];
            next = -1;
            newmatch = false;
            long deg = out_degree(g,u);
            long* adjs = out_vertices(g,u);
            for(long v=0;v<deg;v++){
                //lprintf("w[%ld][%ld]:%f,heaviest:%f,ws[%ld]:%f\n",current,v,weights[current][v],heaviest,adjs[v],ws[adjs[v]]);
                if(weights[current][v]>heaviest && weights[current][v]>ws[adjs[v]]){
                    partner = adjs[v];
                    heaviest = weights[current][v];
                    newmatch = true;
                }
            }
            done = true;
            if(newmatch==true){
                //printf("setting lock\n");
                omp_set_lock(&(lock[partner]));
                //printf("setting lock done\n");
                if(heaviest > ws[partner]){
                    if(mate[partner] != -1){
                        next = mate[partner];
                        done = false;
                    }
                    mate[partner] = current;
                    ws[partner] = heaviest;
                }
                else{
                    done = false;
                    next = u;
                }
                //printf("unsetting lock\n");
                omp_unset_lock(&(lock[partner]));
                //printf("unsetting lock done\n");
            }
            if(done==false){
                current = next;
            }
        }
    }

    free(ws);
    free(lock);
}




__device__ void findPartner(long* neighbors, int vert, int deg, double** weights, double* ws, volatile int* vlocks, int* reducedPartner, double* reducedWeight){
    
    int warp_id = threadIdx.x/warpSize;
    int t_w_id = threadIdx.x % warpSize;
    long currNeighbor;
    double currWeight;
    double currOffer;
    double heaviest = -1;
    int committedChange = 0;
    int currPartner;
    for(int ind = t_w_id; ind < deg; ind+= warpSize){
        currWeight = weights[vert][ind];
        currNeighbor = neighbors[ind];
        if (currWeight<=0){
            continue;
        }
        currOffer = ws[currNeighbor];
        if (currWeight <= heaviest || currWeight <= currOffer)
            continue;
        if(currWeight == ws[currNeighbor]){
            committedChange = 0;
            while(!committedChange){
                if(vlock(vlocks,currNeighbor)){
                    if (currWeight = ws[currNeighbor]){
                        committedChange = 1;
                    }
                    vunlock(vlocks,currNeighbor);
                    committedChange += 1;
                }
            }
            if (committedChange == 2)
                continue;
        }
        heaviest = currWeight;
        currPartner = currNeighbor;
    }
    for (int i = warpSize / 2; i >= 1; i /= 2) {
        double reducedWeight = __shfl_xor_sync(_mask, heaviest, i, warpSize);
        int reducedVert = __shfl_xor_sync(_mask, currPartner, i, warpSize); 
        if (reducedWeight > heaviest) {
            heaviest = reducedWeight;
            currPartner = reducedVert;
        }
    }
    *reducedPartner = currPartner;
    *reducedWeight = currWeight;
    return;


}

__device__ void setMate(int vert, int partner, double heaviest, double* ws, int* mate, volatile int* vlocks, int* done, int *newVert){
    
    *newVert = -1;
    *done = 1;
    int next = -1;
    if (partner >= 0 && heaviest > 0){
        int committedChange = 0;
        *newVert = vert;
        *done = 0;
        while(!committedChange){
            if(vlock(vlocks, partner)){
                if(heaviest >= ws[partner]){
                    next = mate[partner];
                    mate[partner] = vert;
                    ws[partner] = heaviest;
                    *done = 1;
                    vunlock(vlocks, partner);
                    if(next>=0){
                        *done = 0;
                        vert = next;
                        
                    }
                }
                else{
                    *done = 0;
                    vunlock(vlocks, partner);
                }
                committedChange = 1;

            }
        }

    }

}


__device__ void addInRedos(int t_w_id, int* warpmem, int *redos, int newVert, int VertsPerWarp,int fin) {

    int countFinVal = 1-fin;

    for (int i = 1; i <= warpSize / 2; i *= 2) {
        int shflVal = __shfl_up_sync(_mask, countFinVal, i, warpSize);
        if (t_w_id >= i)
        	countFinVal += shflVal;
    }
    if (fin == 0){
        int newidx = VertsPerWarp - (*redos) - (countFinVal - 1);
        warpmem[newidx] = newVert;
    }
    *redos += __shfl_sync(_mask, countFinVal, warpSize - 1, warpSize);

}

__global__ void GPU_Suitor_Matching(graph* g, double** weights, int* mate, double* ws, int VertsPerWarp, volatile int* vlocks){
    extern __shared__ int sharedmem[];
    int warp_id = threadIdx.x/warpSize;
    int t_w_id = threadIdx.x & 0x1f;
    int block_size = blockDim.x;
    int* localMem = &sharedmem[warp_id * VertsPerWarp];
    int partner = -1;
    double partnerWeight = -1;
    int count = 0;
    int s_vert=-1;
    int s_partner=-1;
    double s_weight=-1;
    int done = 0;
    int redos = 0;
    for(int vert = 0; vert<VertsPerWarp; vert++){
        int vertDeg = g->out_offsets[localMem[vert+1]] - g->out_offsets[localMem[vert]];
        long* neighborList = &g->out_adjlist[g->out_offsets[localMem[vert]]];
        findPartner(neighborList, vert, vertDeg, weights, ws, vlocks,&partner, &partnerWeight);
        if ((count%warpSize) == t_w_id){
            s_weight = partnerWeight;
            s_vert = vert + (blockIdx.x * block_size) + warpSize*warp_id; 
            s_partner = partner;
        }
        count+=1;
        if(count%warpSize==0){
            setMate(s_vert,s_partner,s_weight,ws,mate,vlocks,&done,&s_partner);
            addInRedos(t_w_id,localMem,&redos,s_partner,VertsPerWarp,done);
       }
    }
    done = 1;
    int firstFlag = 1;
    int lastRoundRedos = 0;
    while(firstFlag==1 && redos!=0){
        firstFlag = 0;
        lastRoundRedos = redos;
        redos = 0;
        for(int i=0;i<lastRoundRedos;i++){
            partner = -1;
            partnerWeight = -1;
            int newVert = localMem[VertsPerWarp-i];
            int vertDeg = g->out_offsets[localMem[newVert+1]] - g->out_offsets[localMem[newVert]];
            long* neighborList = &g->out_adjlist[g->out_offsets[localMem[newVert]]];
            findPartner(neighborList, newVert, vertDeg, weights, ws, vlocks,&partner, &partnerWeight);
            if ((i%warpSize) == t_w_id){
                s_weight = partnerWeight;
                s_vert = newVert + (blockIdx.x * block_size) + warpSize*warp_id; 
                s_partner = partner;
            }
            if(((i+1)%warpSize)==0){
                setMate(s_vert,s_partner,s_weight,ws,mate,vlocks,&done,&s_partner);
                addInRedos(t_w_id,localMem,&redos,s_partner,VertsPerWarp,done);
            }
        }
        done = 1;
        if( t_w_id<(redos%warpSize) )
    		setMate(s_vert,s_partner,s_weight,ws,mate,vlocks,&done,&s_partner);
        addInRedos(t_w_id,localMem,&redos,s_partner,VertsPerWarp,done);
    }

}