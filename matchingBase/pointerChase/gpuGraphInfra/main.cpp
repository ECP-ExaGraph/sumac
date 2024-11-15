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




int main(int argc, char** argv)
{
    Graph* graph = nullptr;
    std::string inputFileName = "/home/mandum/PNNLwork/G33.bin";
    graph = new Graph(inputFileName);
    float* elapsed_t = new float[NGPU-1];
    GraphGPU* graph_gpu = new GraphGPU(graph, DEFAULT_BATCHES, 1, 1);
    graph_gpu->move_edges_to_device_UVA();
    cudaDeviceSynchronize();
    printf("Starting Matching\n");
    graph_gpu->run_pointer_chase();
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    printf("Finished Matching\n");
    
    return 0;
}