#ifndef _GRAPH_METRICS_H
#define _GRAPH_METRICS_H

void vertex_betweenness_centrality(graph_t* G, double* BC, long numSrcs);
void vertex_betweenness_centrality_simple(graph_t* G, double* BC, long numSrcs);
void vertex_betweenness_centrality_parBFS(graph_t* G, double* BC, long numSrcs);

void edge_betweenness_centrality(graph_t* G, double* BC, long numSrcs);
void edge_betweenness_centrality_simple(graph_t* G, double* BC, long numSrcs);
void edge_betweenness_centrality_parBFS(graph_t* G, double* BC, long numSrcs);

void vertex_closeness_centrality(graph_t* G, double* CC, long numSrcs);
void vertex_closeness_centrality_simple(graph_t* G, double* CC, long numSrcs);
void vertex_closeness_centrality_parBFS(graph_t* G, double* CC, long numSrcs);

void edge_closeness_centrality(graph_t* G, double* CC, long numSrcs);
void edge_closeness_centrality_simple(graph_t* G, double* CC, long numSrcs);
void edge_closeness_centrality_parBFS(graph_t* G, double* CC, long numSrcs);

void evaluate_edge_centrality_bcpart(graph_t* g, attr_id_t* ebc_eval_data1, 
        double* ebc_eval_data2, comm_list_bc_t* comm_list, attr_id_t num_components, 
        attr_id_t curr_component1, attr_id_t curr_component2);
double get_community_modularity(graph_t* g, attr_id_t* membership, attr_id_t
        num_communities);

#endif
