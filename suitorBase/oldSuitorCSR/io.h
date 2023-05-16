#ifndef _IO_H_
#define _IO_H_

#include "graph.h"

void read_bin(char* filename,
 long& num_verts, long& num_edges,
 long*& srcs, long*& dsts);

void read_edge(char* filename,
  long& num_verts, long& num_edges,
  long*& srcs, long*& dsts);

void read_adj(char* filename, long& num_verts, long& num_edges,
  long*& out_array, long*& out_offsets);

void write_ebin(graph* g, char* outfilename);

#endif
