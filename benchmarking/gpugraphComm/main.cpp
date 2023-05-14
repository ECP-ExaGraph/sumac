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

int main(int argc, char** argv)
{
    using namespace std;
    Graph* graph = nullptr;
    std::string inputFileName = "/qfs/projects/pacer/vite_input_bins/cugraph_binfiles/midsize/U1a.bin";
    graph = new Graph(inputFileName);
    GraphGPU* graph_gpu = new GraphGPU(graph, DEFAULT_BATCHES, 1, 1);
    return 0;
}