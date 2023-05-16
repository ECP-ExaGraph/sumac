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

extern long* global_map;

int create_csr(long num_verts, long num_edges, long& max_degree,
  long* srcs, long* dsts,
  long*& out_adjlist, long*& out_offsets)
{
  double elt = omp_get_wtime();
  //printf("Creating graph\n");

  out_adjlist = new long[num_edges];
  out_offsets = new long[num_verts+1];
  long* temp_counts = new long[num_verts];

#pragma omp parallel for
  for (long i = 0; i < num_edges; ++i)
    out_adjlist[i] = 0;
#pragma omp parallel for
  for (int i = 0; i < num_verts+1; ++i)
    out_offsets[i] = 0;
#pragma omp parallel for
  for (int i = 0; i < num_verts; ++i)
    temp_counts[i] = 0;

#pragma omp parallel for
  for (long i = 0; i < num_edges/2; ++i) {
#pragma omp atomic
    ++temp_counts[srcs[i]];
#pragma omp atomic
    ++temp_counts[dsts[i]];
  }
  parallel_prefixsums(temp_counts, out_offsets+1, num_verts);
  for (int i = 0; i < num_verts; ++i)
    assert(out_offsets[i+1] == out_offsets[i] + temp_counts[i]);
#pragma omp parallel for  
  for (int i = 0; i < num_verts; ++i)
    temp_counts[i] = out_offsets[i];
#pragma omp parallel for
  for (long i = 0; i < num_edges/2; ++i) {
    long index = -1;
    int src = srcs[i];
#pragma omp atomic capture
  { index = temp_counts[src]; temp_counts[src]++; }
    out_adjlist[index] = dsts[i];
    int dst = dsts[i];
#pragma omp atomic capture
  { index = temp_counts[dst]; temp_counts[dst]++; }
    out_adjlist[index] = srcs[i];
  }

  max_degree = 0;
#pragma omp parallel for reduction(max:max_degree)
  for (int i = 0; i < num_verts; ++i)
    if (out_offsets[i+1] - out_offsets[i] > max_degree)
      max_degree = out_offsets[i+1] - out_offsets[i];

  delete [] temp_counts;
  /*
  printf("OUTOFF: ");
  for(int i=0;i<num_verts+1;i++){
    printf("%ld ",out_offsets[i]);
  }
  printf("\n");
  printf("ADJ LIST: ");
  for(int i=0;i<num_verts;i++){
    printf("%ld ",out_adjlist[i]);
  }
  printf("\n");
  printf("Sources: ");
  for(int i=0;i<num_edges;i++){
    printf("%ld ",srcs[i]);
  }
  printf("\n");
  printf("Dests: ");
  for(int i=0;i<num_edges;i++){
    printf("%ld ",dsts[i]);
  }
  printf("\n");
  printf("Done : %9.6lf\n", omp_get_wtime() - elt);
  printf("Graph: n=%ld, m=%li, davg=%li, dmax=%li\n", 
    num_verts, num_edges, num_edges / num_verts / 2, max_degree);
  */
  return 0;
}

graph* create_graph(char* filename)
{
  long* srcs;
  long* dsts;
  long num_verts;
  long num_edges;
  long max_degree;
  long* out_adjlist;
  long* out_offsets;

  //read_edge(filename, num_verts, num_edges, srcs, dsts);
  read_bin(filename,num_verts,num_edges,srcs,dsts);
  create_csr(num_verts, num_edges, max_degree, srcs, dsts, 
    out_adjlist, out_offsets);
  free(srcs);
  free(dsts);
  /*
  printf("GRAPH STATS\n");
  printf("Out AdjList\n");
  printf("Size: %ld\n",num_edges);  
  for(int i=0;i<num_edges;i++){
    printf("%ld ",out_adjlist[i]);
  }
  */
  graph* g = (graph*)malloc(sizeof(graph));
  g->num_verts = num_verts;
  g->num_edges = num_edges;
  g->max_degree = max_degree;
  g->out_adjlist = out_adjlist;
  g->out_offsets = out_offsets;
  g->label_map = (long*)malloc(g->num_verts * sizeof(long));
/*
  printf("Outoffsets ");
  for(int i = 0; i < sizeof(*out_offsets)/sizeof(out_offsets[0]);i++){
    printf("%ld ",out_offsets[i]);
  }
*/
#pragma omp parallel for
  for (int i = 0; i < g->num_verts; ++i)
    g->label_map[i] = i;

  return g;
}


int clear_graph(graph*& g)
{
  g->num_verts = 0;  
  g->num_edges = 0;
  g->max_degree = 0;
  delete [] g->out_adjlist;
  //delete [] g->out_offsets; //This statement is causing a seg fault ?
  free(g->label_map); //added this, was having a memory leak.
  free(g);

  return 0;
}

int copy_graph(graph* g, graph* new_g)
{
  // if (new_g == NULL) {}
  new_g->num_verts = g->num_verts;
  new_g->num_edges = g->num_edges;
  new_g->out_offsets = new long[g->num_verts+1];
  new_g->out_adjlist = new long[g->num_edges];
  new_g->label_map = (long*)malloc(new_g->num_verts * sizeof(long));

#pragma omp parallel for
  for (int i = 0; i < g->num_verts+1; ++i)
    new_g->out_offsets[i] = g->out_offsets[i];

#pragma omp parallel for
  for (long i = 0; i < g->num_edges; ++i)
    new_g->out_adjlist[i] = g->out_adjlist[i];
#pragma omp parallel for
  for (int i = 0; i < new_g->num_verts; ++i)
    new_g->label_map[i] = g->label_map[i];
  return 0;
}
