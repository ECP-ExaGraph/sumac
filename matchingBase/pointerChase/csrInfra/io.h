#ifndef _IO_H_
#define _IO_H_

#include "graph.h"

void read_bin(char* filename,
 int& num_verts, int& num_edges,
 int*& srcs, int*& dsts);

void read_edge(char* filename,
  long& num_verts, long& num_edges,
  long*& srcs, long*& dsts);

void read_adj(char* filename, int& num_verts, int& num_edges,
  int*& out_array, int*& out_offsets);

void write_ebin(graph* g, char* outfilename);

#endif
