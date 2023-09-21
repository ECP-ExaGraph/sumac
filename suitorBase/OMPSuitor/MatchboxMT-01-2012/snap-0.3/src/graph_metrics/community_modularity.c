#include "graph_defs.h"
#include "graph_metrics.h"
#include "utils.h"
#include "sprng.h"

double get_community_modularity(graph_t *g, attr_id_t *membership, attr_id_t
        num_components) {

    attr_id_t u, v, j;
    attr_id_t n, m;
    attr_id_t cid;
    double m2_inv; 
    attr_id_t u_edge_start, u_edge_end, u_degree;
    double mod;
    double *A, *E;

    n = g->n;
    m = g->m; 
    m2_inv = 1.0/(g->m);
    mod = 0.0;
    
    A = (double *) calloc(num_components, sizeof(double));
    E = (double *) calloc(num_components, sizeof(double));
    assert(A != NULL);
    assert(E != NULL);

    for(u=0; u<n; u++) {
        cid = membership[u];
        u_edge_start = g->numEdges[u];
        u_edge_end   = g->numEdges[u+1];
        u_degree     = u_edge_end - u_edge_start;

        for(j=u_edge_start; j<u_edge_end; j++) {
            v = g->endV[j];
            if (membership[v] == cid)
                E[cid] += 1.0;
        }
        A[cid] += (double) u_degree;
    }

    for (j=0; j<num_components; j++) {
        mod += E[j] - m2_inv * A[j] * A[j]; 
    }
    mod = mod * m2_inv;
    free(A);
    free(E);

    return mod;
}
