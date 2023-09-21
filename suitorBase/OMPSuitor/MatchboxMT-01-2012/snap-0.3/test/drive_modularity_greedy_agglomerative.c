#include "graph_defs.h"
#include "graph_gen.h"
#include "graph_kernels.h"
#include "graph_metrics.h"
#include "graph_partitioning.h"
#include "utils.h"

int main(int argc, char** argv) {

    char *infilename, *outfilename, *graph_type, *alg_type;
    FILE* fp;
    graph_t* g;
    int curArgIndex;
    attr_id_t *membership;
    int num_communities;
    double modularity, mod_val;
    long i;
    int run_approxBC;

    /* Step 1: Parse command line arguments */
    if (argc < 3) {
        fprintf(stdout, "\nUsage: %s -infile <graph filename>"
                " -graph <graph type> -outfile <output filename>)"
                " -alg <algorithm type>\n\n",
                "eval_modularity_betweenness");
        fprintf(stdout, "Algorithm type can be one of the following:\n"
                "CNM         -- greedy agglomerative strategy of "
                "Clauset, Newman and Moore.\n\n"
                /*"WT1/WT2/WT3 -- Wakita and Tsurami's consolidation ratio heuristics.\n"
                "DDA         -- Normalized modularity heuristics by Danon, "
                "Diaz-Guilera, and Arenas.\n"
                "CCA         -- An agglomerative clustering heuristic based "
                "on local clustering coefficient.\n" */
                );
        usage_graph_options();
        exit(-1);
    }

    curArgIndex = 0;
    run_approxBC = 0;
    infilename = (char *) calloc(500, sizeof(char));
    outfilename = (char *) calloc(500, sizeof(char));
    graph_type = (char *) calloc(500, sizeof(char));
    alg_type = (char *) calloc(500, sizeof(char));

    strcpy(outfilename, "output.txt");

    while (curArgIndex < argc) {
        
        if (strcmp(argv[curArgIndex],"-infile")==0) {
            strcpy(infilename, argv[++curArgIndex]);
        }

        if (strcmp(argv[curArgIndex], "-outfile")==0) {
            strcpy(outfilename, argv[++curArgIndex]);
        } 
 
        if (strcmp(argv[curArgIndex], "-graph")==0) {
            strcpy(graph_type, argv[++curArgIndex]);
        } 
 
        if (strcmp(argv[curArgIndex],"-alg")==0) {
            strcpy(alg_type, argv[++curArgIndex]);
        }

        curArgIndex++; 

    }

    fp = fopen(infilename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error! Could not open input file. Exiting ...\n");
        exit(-1);
    }
    fclose(fp);

    fp = fopen(outfilename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error! Could not write to output file. Exiting ...\n");   
        exit(-1);
    }
    fclose(fp);

    graph_ext_check(infilename, graph_type);

    fprintf(stdout, "\n");
    fprintf(stdout, "Input Graph File    : %s\n", infilename);
    fprintf(stdout, "Output Graph File   : %s\n\n", outfilename);

    /* Step 2: Generate graph */
    g = (graph_t *) malloc(sizeof(graph_t));
    graph_gen(g, infilename, graph_type);
   
    fprintf(stdout, "Number of vertices     : %ld\n", g->n);
    if (g->undirected)
        fprintf(stdout, "Number of edges        : %ld\n\n", g->m/2);
    else 
        fprintf(stdout, "Number of edges        : %ld\n\n", g->m);

    if (g->undirected == 0) {
        fprintf(stderr, "Error: the graph has to be undirected.\n");
        fprintf(stderr, "Please check input file.\n");
        exit(-1);
    }

    /* Step 3: Run algorithm */
    
    membership = (attr_id_t *) malloc(g->n * sizeof(attr_id_t));
    assert(membership != NULL);
    modularity = 0.0;
    num_communities = 0;
    modularity_greedy_agglomerative(g, alg_type, membership, 
            &num_communities, &modularity);

    mod_val = get_community_modularity(g, membership, num_communities);
    /* Step 4: Write output to file */
    fp = fopen(outfilename, "w");
    fprintf(fp, "Number of communities: %d\n", num_communities);
    fprintf(fp, "Modularity score: %f (full), %f (w/o dup)\n", mod_val,
            modularity);
    fprintf(fp, "\n<Vertex ID> <Community ID>\n\n");
    
    for (i=0; i<g->n; i++) {
        if (g->zero_indexed)
            fprintf(fp, "%ld %d\n", i, membership[i]);  
        else
            fprintf(fp, "%ld %d\n", i+1, membership[i]);  

    }
    fprintf(stderr, "Modularity: %f (full), %f (w/o dup)\n", mod_val,
            modularity);
    fprintf(stderr, "Number of communities: %d\n", num_communities);
    
    fclose(fp);

    /* Step 5: Clean up */
    free(infilename);
    free(outfilename);
    free(graph_type);
    free(alg_type);
    free_graph(g);
    free(g);
    free(membership);

    return 0;
}
