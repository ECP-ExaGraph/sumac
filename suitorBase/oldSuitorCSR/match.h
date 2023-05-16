#ifndef _MATCH_H_
#define _MATCH_H_


void suitorOMP(graph* g,double** weights,long* mate);
__global__ void GPU_Suitor_Matching(graph* g, double** weights, int* mate, double* ws, int VertsPerWarp, volatile int* vlocks);
__device__ void addInRedos(int t_w_id, int* warpmem, int *redos, int newVert, int VertsPerWarp,int fin);
__device__ void setMate(int vert, int partner, double heaviest, double* ws, int* mate, volatile int* vlocks, int* done, int *newVert);
__device__ void findPartner(long* neighbors, int vert, int deg, double** weights, double* ws, volatile int* vlocks, int* reducedPartner, double* reducedWeight);
__device__ void vunlock(volatile int*  locks, int idx);
__device__ int vlock(volatile int* locks, int idx);
#endif