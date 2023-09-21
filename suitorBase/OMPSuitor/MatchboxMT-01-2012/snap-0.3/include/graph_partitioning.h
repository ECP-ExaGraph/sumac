#ifndef _GRAPH_PARTITIONING_H
#define _GRAPH_PARTITIONING_H
#include "graph_defs.h"
void modularity_spectral(graph_t *G, attr_id_t *membership, 
        attr_id_t *numCommunities, attr_id_t use_improvement);
void modularity_spectral_wo_klin(graph_t *G, attr_id_t *membership, 
        attr_id_t *numCommunities);
void computeModularityValue(graph_t *G, attr_id_t *membership, 
        attr_id_t numCommunities, double *modularity);
void modularity_betweenness(graph_t *g, attr_id_t *membership, 
        attr_id_t *num_communities, double *modularity, 
        double sampling_val);
void modularity_greedy_agglomerative(graph_t *g, char *alg_type, 
        attr_id_t *membership, attr_id_t *num_communities, double *modularity);
#endif
