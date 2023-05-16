
#include <omp.h>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdint>

#include "io.h"
#include "graph.h"
#include "util.h"


void read_bin(char* filename,
 long& num_verts, long& num_edges,
 long*& srcs, long*& dsts)
{
  num_verts = 0;
  double elt = omp_get_wtime();
  //printf("Reading %s ", filename);
#pragma omp parallel
{
  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  FILE *infp = fopen(filename, "rb");
  if(infp == NULL) {
    printf("%d - load_graph_edges() unable to open input file", tid);
    exit(0);
  }

  fseek(infp, 0L, SEEK_END);
  uint64_t file_size = ftell(infp);
  fseek(infp, 0L, SEEK_SET);

  uint64_t nedges_global = file_size/(2*sizeof(uint32_t));

#pragma omp single
{
  num_edges = (long)nedges_global;
  srcs = new long[num_edges];
  dsts = new long[num_edges];
}

  uint64_t read_offset_start = tid*2*sizeof(uint32_t)*(nedges_global/nthreads);
  uint64_t read_offset_end = (tid+1)*2*sizeof(uint32_t)*(nedges_global/nthreads);

  if (tid == nthreads - 1)
    read_offset_end = 2*sizeof(uint32_t)*nedges_global;

  uint64_t nedges = (read_offset_end - read_offset_start)/(2*sizeof(uint32_t));
  uint32_t* edges_read = (uint32_t*)malloc(2*nedges*sizeof(uint32_t));
  if (edges_read == NULL) {
    printf("%d - load_graph_edges(), unable to allocate buffer", tid);
    exit(0);
  }

  fseek(infp, read_offset_start, SEEK_SET);
  fread(edges_read, nedges, 2*sizeof(uint32_t), infp);
  fclose(infp);
  //printf(".");

  uint64_t array_offset = (uint64_t)tid*(nedges_global/nthreads);
  uint64_t counter = 0;
  for (uint64_t i = 0; i < nedges; ++i) {
    int src = (int)edges_read[counter++];
    int dst = (int)edges_read[counter++];
    srcs[array_offset+i] = src;
    dsts[array_offset+i] = dst;
  }

  free(edges_read);
  //printf(".");

#pragma omp barrier

#pragma omp for reduction(max:num_verts)
  for (uint64_t i = 0; i < nedges_global; ++i)
    if (srcs[i] > num_verts)
      num_verts = srcs[i];
#pragma omp for reduction(max:num_verts)
  for (uint64_t i = 0; i < nedges_global; ++i)
    if (dsts[i] > num_verts)
      num_verts = dsts[i]; 
           
} // end parallel

  num_edges *= 2;
  num_verts += 1;
  //printf(" Done %9.6lf\n", omp_get_wtime() - elt);
}


void read_edge(char* filename,
  long& num_verts, long& num_edges,
  long*& srcs, long*& dsts)
{
  FILE* infile = fopen(filename, "r");
  char line[256];

  num_verts = 0;

  long count = 0;
  long cur_size = 1024*1024;
  //printf("cursize: %ld\n",cur_size);

  srcs = (long*)malloc(cur_size*sizeof(long));
  dsts = (long*)malloc(cur_size*sizeof(long));

  while(fgets(line, 256, infile) != NULL) {
    if (line[0] == '%') continue;
    //printf("Read line: %s",line);
    sscanf(line, "%ld %ld", &srcs[count], &dsts[count]);
    // dsts[count+1] = srcs[count];
    // srcs[count+1] = dsts[count];
    //printf("Reading %d, %d, +1: %d, %d\n", srcs[count], dsts[count],srcs[count+1],dsts[count+1]);

    if (srcs[count] > num_verts)
      num_verts = srcs[count];
    if (dsts[count] > num_verts)
      num_verts = dsts[count];

    count += 1;
    if (count > cur_size) {
      cur_size *= 2;
      srcs = (long*)realloc(srcs, cur_size*sizeof(long));
      dsts = (long*)realloc(dsts, cur_size*sizeof(long));
    }
  }  
  num_verts += 1;
  num_edges = count*2;
  //printf("Read: n: %ld, m: %li\n", num_verts, num_edges);
  /*
  printf("IOSources: ");
  for(int i=0;i<count;i++){
    printf("%ld ",srcs[i]);
  }
  printf("\n");
  printf("IODests:   ");
  for(int i=0;i<count;i++){
    printf("%ld ",dsts[i]);
  }
  printf("\n");
  */
  fclose(infile);

  return;
}

void read_adj(char* filename, int& num_verts, long& num_edges,
  int*& out_array, long*& out_offsets)
{
  std::ifstream infile;
  std::string line;
  std::string val;

  out_array = new int[num_edges];
  out_offsets = new long[num_verts+1];

#pragma omp parallel for
  for (int i = 0; i < num_verts+1; ++i)
    out_offsets[i] = 0;

  long count = 0;
  int cur_vert = 0;

  infile.open(filename);
  getline(infile, line);

  while (getline(infile, line))
  {
    std::stringstream ss(line);
    out_offsets[cur_vert] = count;
    ++cur_vert;

    while (getline(ss, val, ' ')) {
      out_array[count++] = atoi(val.c_str())-1;
    }
  }
  out_offsets[cur_vert] = count;

  assert(cur_vert == num_verts);
  assert(count == num_edges);

  infile.close();  
}
void write_ebin(graph* g, char* outfilename)
{ 
  FILE* mapGraphFile = fopen(outfilename, "wb");
  
  uint32_t write[2];
  for (int i = 0; i < g->num_verts; ++i) {
    long v = g->label_map[i];
    long* adjs = out_vertices(g, i);
    long degree = out_degree(g, i);
    for (int j = 0; j < degree; ++j) {
      long u = g->label_map[adjs[j]];
      if (v < u) {
        write[0] = (uint32_t)v;
        write[1] = (uint32_t)u;
        fwrite(write, sizeof(uint32_t), 2, mapGraphFile);
      }
    }
  }
  
  fclose(mapGraphFile);
}