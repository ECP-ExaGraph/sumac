#ifndef _GRAPH_H_
#define _GRAPH_H_

struct graph {
  long num_verts;
  long num_edges;
  long max_degree;
  long* out_adjlist;
  long* out_offsets;
  long* label_map;
} ;

inline long out_degree(graph* g, int v) 
{ 
  return g->out_offsets[v+1] - g->out_offsets[v];
}

inline long* out_vertices(graph* g, int v) 
{ 
  return &g->out_adjlist[g->out_offsets[v]];
}

int create_csr(long num_verts, long num_edges, long& max_degree,
  long* srcs, long* dsts,
  long*& out_adjlist, long*& out_offsets);

graph* create_graph(char* filename);

int clear_graph(graph*& g);

int copy_graph(graph* g, graph* new_g);

#endif
