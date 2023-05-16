
using namespace std;

#include <cstdlib>
#include <iostream>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <chrono>
#include <random>
#include <omp.h>

#include "io.h"
#include "graph.h"
#include "metric.h"
#include "match.h"

int VertsPerWarp = 8;

int seed = 0;

int main(int argc, char** argv)
{
  setbuf(stdout, NULL);
  srand(time(0));

  if (argc < 1)
  {
    printf("To run: %s [graphFile]\n\n",
      argv[0]);
    exit(0);
  }

  char* graphFile = argv[1];
  graph* g_host = create_graph(graphFile);

  graph* g = NULL;
  assert(cudaMallocManaged(&g, sizeof(graph)) == cudaSuccess);
  cudaMemcpy(&g->num_verts, &g_host->num_verts, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&g->num_edges, &g_host->num_edges, sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(&g->max_degree, &g_host->max_degree, sizeof(long), cudaMemcpyHostToDevice);

  long num_verts = g_host->num_verts;
  long num_edges = g_host->num_edges;
  assert(cudaMallocManaged(&g->out_adjlist, num_edges*sizeof(int)) == cudaSuccess);
  assert(cudaMallocManaged(&g->out_offsets, (num_verts + 1)*sizeof(long)) == cudaSuccess);
  cudaMemcpy(g->out_adjlist, g_host->out_adjlist, num_edges*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(g->out_offsets, g_host->out_offsets, (num_verts + 1)*sizeof(long), cudaMemcpyHostToDevice);
  

  printf("Graph created: %s\n",graphFile);

  //double** weights = new double*[g->num_verts];
  double** weights;
  cudaMallocManaged(&weights,g->num_verts*sizeof(double));
  #pragma omp parallel for
  for(long u=0;u<g->num_verts;u++){
    long deg = out_degree(g,u);
    //weights[u] = new double[deg];
    cudaMallocManaged(&weights[u],deg*sizeof(double));
    for(long v=0;v<deg;v++)
      weights[u][v] = rand() % 1000;
  }
  //long* mate = new long[g->num_verts];
  int* mate;
  double* ws;
  long* h_verts = new long[g->num_verts];
  long* d_verts;
  cudaMallocManaged(&mate,g->num_verts*sizeof(int));
  cudaMallocManaged(&ws,g->num_verts*sizeof(double));

  for(long u=0;u<g->num_verts;u++){
    h_verts[u] = u;
    mate[u]=-1;
  }


  cudaMalloc(&d_verts, g->num_verts * sizeof(long));
  cudaMemcpy(d_verts, h_verts, g->num_verts * sizeof(long), cudaMemcpyHostToDevice);


  int device = -1;
  cudaGetDevice(&device);
  cudaMemPrefetchAsync(g->out_adjlist, num_edges*sizeof(int), device, NULL);
  cudaMemPrefetchAsync(g->out_offsets, (num_verts + 1)*sizeof(long), device, NULL);
  int* vlocks;
  cudaMalloc(&vlocks,g->num_verts*sizeof(int));

  printf("Matching data structs initialized\n");
  int block_size = 128;
  int num_blocks = (g->num_verts + block_size - 1) / block_size;
  printf("NumBlocks:%ld\n",num_blocks);
  printf("Starting Matching\n");

  double elt = omp_get_wtime();
  GPU_Suitor_Matching<<<num_blocks,block_size,(VertsPerWarp*(block_size/32))*sizeof(int)>>>(g,weights,mate,ws,VertsPerWarp,vlocks);
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
if(error!=cudaSuccess)
{
   fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
   exit(-1);
}
  printf("Finished Matching, Time: %9.6f(s)\n",omp_get_wtime()-elt);

  clear_graph(g_host);
  cudaFree(g_host->out_adjlist);
  cudaFree(g_host->out_offsets);
  cudaFree(mate);
  cudaFree(weights);
  cudaFree(vlocks);
  return 0;
}
