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
    std::string inputFileName = argv[1];
    graph = new Graph(inputFileName);

    GraphGPU* graph_gpu = new GraphGPU(graph, 12, 1, 1);
    cudaDeviceSynchronize();
    
    printf("Starting Matching\n");
    double start; 
    double end; 
    start = omp_get_wtime(); 
    graph_gpu->run_pointer_chase();
    cudaDeviceSynchronize();
    //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );

    end = omp_get_wtime(); 

    printf("Finished Matching\n");
    printf("Time Elapsed: %f seconds\n", end - start);
    graph_gpu->output_matching();

    
    return 0;
}