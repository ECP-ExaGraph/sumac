#include "utils.h"

double get_seconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double) (tp.tv_sec + ((1e-6)*tp.tv_usec));
}

void usage_graph_options() {
    fprintf(stdout, "The input file name is a required argument. We try to determine the graph");
    fprintf(stdout, " format from the file extension; optionally, you can specify the graph type with");
    fprintf(stdout, " the -graph option.\n\n");
    fprintf(stdout, "Supported graph file formats (for use with the -graph option):\n");
    fprintf(stdout, "snap    (.gr)       SNAP file format (default file format supported by this package).\n");
    fprintf(stdout, "dimacs  (.dim)      graph format used in the 9th DIMACS Shortest Paths Challenge.\n");
    fprintf(stdout, "GML     (.gml)      a limited implementation of the GML graph format (only supports undirected graphs, edge weights represented as doubles.)\n");
#if 0
    /* Commenting out unimplemented options */
    fprintf(stdout, "metis   (.met)      graph representation used by the Metis partitioning package.\n");
    fprintf(stdout, "graphml (.graphml)  GraphML representation.\n");
    fprintf(stdout, "sqmesh  (.sqm)      a synthetic 2D square mesh generator is invoked."
            " The input file specifies the generator parameters.\n");
    fprintf(stdout, "lmesh   (.lm)       a synthetic 2D long mesh generator is invoked."
            " The input file specifies the generator parameters.\n");
#endif    
    fprintf(stdout, "rand    (.rnd)      a synthetic random graph generator is invoked."
            " The input file specifies the generator parameters.\n");
    fprintf(stdout, "rmat    (.rmat)       a synthetic scale-free graph generator is invoked."
            " The input file specifies the generator parameters.\n");
   fprintf(stdout, "\n");
    fprintf(stdout, "The default output filename is \"results.PID.txt\"."
            " (PID is the process ID). "
            "Specify an alternate file name with the -outfile option.\n");
}

void graph_ext_check(char* filename, char* graph_type) {

    char *ptr, *tmp; 

    ptr = filename;
    
    /* Find the last "." in the file name */
    do {
        tmp = strstr(ptr+1, ".");
        if (tmp != NULL) 
            ptr = tmp;
    }  while (tmp != NULL);
    ptr++;   

    if (graph_type[0] == 0) {
        if (strcmp(ptr, "gr") == 0) {
            strcpy(graph_type, "snap");
        }

        if (strcmp(ptr, "dim") == 0) {
            strcpy(graph_type, "dimacs");    
        }

        if (strcmp(ptr, "gml") == 0) {
            strcpy(graph_type, "gml");    
        }

        if (strcmp(ptr, "rnd") == 0) {
            strcpy(graph_type, "rand");    
        }

        if (strcmp(ptr, "rmat") == 0) {
            strcpy(graph_type, "rmat");    
        }
#if 0
        if (strcmp(ptr, "met") == 0) {
            strcpy(graph_type, "metis");    
        }

        if (strcmp(ptr, "graphml") == 0) {
            strcpy(graph_type, "graphml");    
        }

        if (strcmp(ptr, "sqmesh") == 0) {
            strcpy(graph_type, "sqmesh");    
        }

        if (strcmp(ptr, "lmesh") == 0) {
            strcpy(graph_type, "lmesh");    
        }
#endif
    }

}

void prefix_sums(attr_id_t *input, attr_id_t* result, attr_id_t* p, attr_id_t n) {

#ifdef _OPENMP
    attr_id_t i, j, r, start, end, add_value;
    int tid, nthreads;

    tid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
    r = n/nthreads;

    result[0] = 0;

    #pragma omp for
    for (i=1; i<n+1; i++)
        result[i] = input[i-1];

    start =  tid*r + 1;
    end   = (tid+1)*r;

    if (tid == nthreads-1)
        end = n+1;

    for (j=start; j<end; j++)
        result[j] = input[j-1] + result[j-1];

    p[tid] = result[end-1];

    #pragma omp barrier

    if (tid == 0) {
        for (j=1; j<nthreads; j++)
            p[j] += p[j-1];
    }

    #pragma omp barrier

    if (tid>0) {
        add_value=p[tid-1];
        for (j=start-1; j<end; j++)
            result[j] += add_value;
    }

    #pragma omp barrier
#else

    attr_id_t i;
    result[0] = 0;
    for (i=1; i<n+1; i++) {
        result[i] = result[i-1] + input[i-1];
    }
#endif
    
}



void print_graph(graph_t* G)
{
    attr_id_t i,j;
    attr_id_t degree,start;

    fprintf(stdout, "\n Printing graph to stdout\n");
    
    fprintf(stdout, "Number of vertices: %ld, edges: %ld\n", G->n, G->m);
    if(G->undirected) 
        fprintf(stdout, "Graph is undirected\n");
    else
        fprintf(stdout, "Graph is directed\n");

    fprintf(stdout, "Graph weight type is %d\n",G->weight_type);

#if ENABLE_64BIT_VID

    for(i = 0; i < G->n; i++) {
        degree = G->numEdges[i+1] - G->numEdges[i];
        start = G->numEdges[i];
        fprintf(stdout, "Degree of vertex i %ld is %ld." 
                "Its neighbours are:", i, degree);

        for(j=0;j<degree;j++) {
            if(G->weight_type == 1)
                fprintf(stdout, "%ld [%d], ",G->endV[start+j], 
                        G->int_weight_e[start+j]);
            else if(G->weight_type == 2) 
                fprintf(stdout, "%ld [%ld], ",G->endV[start+j], 
                        G->l_weight_e[start+j]);
            else if(G->weight_type == 3) 
                fprintf(stdout, "%ld [%f], ",G->endV[start+j], 
                        G->fl_weight_e[start+j]);
            else if(G->weight_type == 4) 
                fprintf(stdout, "%ld [%f], ",G->endV[start+j], 
                        G->dbl_weight_e[start+j]);
        }
        fprintf(stdout, "\n");
    }

#else

    for(i = 0; i < G->n; i++) {
        degree = G->numEdges[i+1] - G->numEdges[i];
        start = G->numEdges[i];
        fprintf(stdout, "Degree of vertex i %d is %d." 
                "Its neighbours are:", i, degree);

        for(j=0;j<degree;j++) {
            if(G->weight_type == 1)
                fprintf(stdout, "%d [%d], ",G->endV[start+j], 
                        G->int_weight_e[start+j]);
            else if(G->weight_type == 2) 
                fprintf(stdout, "%d [%ld], ",G->endV[start+j], 
                        G->l_weight_e[start+j]);
            else if(G->weight_type == 3) 
                fprintf(stdout, "%d [%f], ",G->endV[start+j], 
                        G->fl_weight_e[start+j]);
            else if(G->weight_type == 4) 
                fprintf(stdout, "%d [%f], ",G->endV[start+j], 
                        G->dbl_weight_e[start+j]);
        }
        fprintf(stdout, "\n");
    }
#endif
}

void print_snap_header(FILE *outfile) {
    fprintf(outfile, "\n"
                     "************************************************\n");
    fprintf(outfile, "  SNAP Complex Network Analysis Framework v0.3\n");
    fprintf(outfile, "  Authors: Kamesh Madduri, David A. Bader  \n");
    fprintf(outfile, "  Last Updated: March 2009\n");
    fprintf(outfile, "  http://snap-graph.sourceforge.net\n");
    fprintf(outfile, "************************************************\n");
}


void print_graph_header(FILE *outfile, graph_t *g, const char *problem_type) {
    fprintf(outfile, "  Number of vertices : %ld\n", g->n);
    if (g->undirected) {
        fprintf(outfile, "  Number of edges    : %ld\n", g->m/2);
        fprintf(outfile, "  Undirected network.\n");
    } else {
        fprintf(outfile, "  Number of edges    : %ld\n", g->m);
        fprintf(outfile, "  Directed network.\n");
    }
    /*
    if (g->weight_type == 0)
        fprintf(outfile, "  Unweighted network.\n");
    else if (g->weight_type == 1)
        fprintf(outfile, "  32-bit integer weight.\n");
    else if (g->weight_type == 2)
        fprintf(outfile, "  Long integer (%d bytes) weights.\n", sizeof(long));
    else if (g->weight_type == 3)
        fprintf(outfile, "  Single-precision float weights.\n");
    else if (g->weight_type == 4)
        fprintf(outfile, "  Double-precision float weights.\n");
    */ 
    fprintf(outfile, "------------------------------------------------\n");
    fprintf(outfile, "  Problem type       : %s\n", problem_type); 
}
