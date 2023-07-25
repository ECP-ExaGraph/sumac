#include <cstdlib>
#include <iostream>
#include <string>
#include <omp.h>
#include "graph.hpp"
#include "graph_gpu.hpp"
#include "types.hpp"
#include "cuda_wrapper.hpp"

#include <unistd.h>
#include <fstream>
#include <sstream>

using namespace std;



#define gpuErrchk(ans) {gpuAssert((ans), __FILE__,__LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


void GPU_copy_data(int gpuID, GraphElem numVerts, GraphElem** indices, float* elapsed_t){
	

	cudaEvent_t start,fin;
	gpuErrchk(cudaEventCreate(&start));
	gpuErrchk(cudaEventCreate(&fin));

	long* data;
    int canAccessOut = 0;
    int canAccessIn = 0;
    gpuErrchk(cudaSetDevice(gpuID));
    cudaDeviceCanAccessPeer(&canAccessOut,gpuID,gpuID+1);
    if(canAccessOut==0)
        cudaDeviceEnablePeerAccess(gpuID+1,0);
    gpuErrchk(cudaMalloc(&data,numVerts));

    gpuErrchk(cudaSetDevice(gpuID+1));
    cudaDeviceCanAccessPeer(&canAccessIn,gpuID+1,gpuID);
    if(canAccessIn==0)
        cudaDeviceEnablePeerAccess(gpuID+1,0);
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&fin));
    
    gpuErrchk(cudaEventRecord(start));
    gpuErrchk(cudaMemcpyPeerAsync(data,gpuID,indices[gpuID+1],gpuID+1,numVerts));
    gpuErrchk(cudaEventRecord(fin));
    gpuErrchk(cudaEventSynchronize(fin));
        
    gpuErrchk(cudaEventElapsedTime(&elapsed_t[gpuID],start,fin));		

    gpuErrchk(cudaSetDevice(gpuID));
    gpuErrchk(cudaFree(data));
    gpuErrchk(cudaSetDevice(gpuID+1));
	
	gpuErrchk(cudaEventDestroy(start));
	gpuErrchk(cudaEventDestroy(fin));

}


int main(int argc, char** argv)
{
    Graph* graph = nullptr;
    std::string inputFileName = "/data/graphs/U1a.bin";
    graph = new Graph(inputFileName);
    float* elapsed_t = new float[NGPU-1];
    GraphGPU* graph_gpu = new GraphGPU(graph, DEFAULT_BATCHES, 1, 1);
    GraphElem** indices = graph_gpu->get_indices_device();
    GraphElem* nv = graph_gpu->get_nv_device();
    for(int i=0;i<NGPU-1;i++){
        GPU_copy_data(i,nv[i+1],indices,elapsed_t);
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    printf("To,From,Time(ms)\n");
    for(int i=0;i<NGPU-1;i++){
        //printf("Time %d->%d:%f\n",i+1,i,elapsed_t[i]);
        printf("%d,%d,%f\n",i,i+1,elapsed_t[i]);
    }
    
    return 0;
}