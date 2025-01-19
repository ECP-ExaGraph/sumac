#ifndef _MATCH_H_
#define _MATCH_H_


__global__ void Pointer_Chase_GPU(graph* g, int* verts, volatile double** weights, volatile int* mate, volatile int* pointers, int vertsPerWarp);
__device__ void findEdge(int vid, int* neighbors, int deg, volatile double* vertWeights, volatile int* pointers);
__device__ void removeEdges(int vid, int* neighbors, int deg, volatile double** weights, volatile int* mate);
#endif