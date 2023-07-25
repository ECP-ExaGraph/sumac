#ifndef _GRAPH_H_
#define _GRAPH_H_

struct graph {
  int num_verts;
  int num_edges;
  int max_degree;
  int* out_adjlist;
  int* out_offsets;
} ;

inline int out_degree(graph* g, int v) 
{ 
  return g->out_offsets[v+1] - g->out_offsets[v];
}

inline int* out_vertices(graph* g, int v) 
{ 
  return &g->out_adjlist[g->out_offsets[v]];
}

int create_csr(int num_verts, int num_edges, int& max_degree,
  int* srcs, int* dsts,
  int*& out_adjlist, int*& out_offsets);

graph* create_graph(char* filename);

int clear_graph(graph*& g);

int copy_graph(graph* g, graph* new_g);

#endif
