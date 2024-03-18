#include <cstdlib>
#include <iostream>
#include <string>
#include <omp.h>
#include <cuda_runtime.h>
#include "types.hpp"
#include "graph.hpp"
#include "graph_gpu.hpp"
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

    if(argc!=4){
        std::cout << argv[0] << " [binGraph] [numBatches] [edgebalance/logical (1/0)]" << std::endl;
        return 1;
    }
    Graph* graph = nullptr;
    std::string inputFileName = argv[1];
    std::string batchNum = argv[2];
    std::string edgebal = argv[3];
    graph = new Graph(inputFileName);
    std::cout << "Graph: " << inputFileName << " NGPU: " << NGPU << " Batches: " << batchNum << " EdgeBalance? " << edgebal << std::endl;  
    GraphGPU* graph_gpu = new GraphGPU(graph, stoi(batchNum), 1, 1, stoi(edgebal));
    cudaDeviceSynchronize();
    
    printf("Starting Matching\n");
    double total;
    total = graph_gpu->run_pointer_chase();
    cudaDeviceSynchronize();

    printf("Finished Matching\n");
    printf("Total Time Elapsed: %f seconds\n", total);

    
    return 0;
}