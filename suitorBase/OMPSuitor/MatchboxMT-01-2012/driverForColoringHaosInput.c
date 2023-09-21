
#include "coloringAndMatchingKernels.h"
#include "defs.h"

int main(int argc, char** argv) {
  
  /* Step 1: Parse command line arguments */
  if (argc < 3) {
    printf("======================================\n");
    printf("Usage: %s <Input filename> <|V|> <|E|>\n", argv[0]);
    printf("======================================\n");
    exit(-1);
    
  }
  graph G;
  
  G.numVertices = atol(argv[2]);
  G.sVertices = G.numVertices;
  G.numEdges = atol(argv[3]);  
  printf("|V|= %ld   |E|= %ld\n", G.numVertices, G.numEdges);
  
  long i=0, v1=0, v=0, g=0;
  //Parallel Initialization for first-touch purposes:
  double time1 = omp_get_wtime();
  long    *vtxPtr   = (long*) malloc((G.numVertices+1)*sizeof(long));
  assert(vtxPtr != 0); 
  edge    *vtxInd   = (edge*)malloc((G.numEdges)*sizeof(edge));
  assert(vtxInd != 0);
#pragma omp parallel for
  for (long i=0; i<=G.numVertices; i++) {
    vtxPtr[i] = 0;
  }
#pragma omp parallel for
  for (long i=0; i<G.numEdges; i++) {
    vtxInd[i].tail   = -1;
    vtxInd[i].weight = 0;
  }
  double time2 = omp_get_wtime();
  printf("Time for memory allocation: %3.3lf\n", time2-time1);
  
  G.edgeListPtrs = vtxPtr;
  G.edgeList     = vtxInd;
  
  //Read input from a file:
  printf("About to read file: %s\n", argv[1]);
  printf("****WARNING**** Setting all weights to 1.\n");
  time1 = omp_get_wtime();
  
  FILE* in = fopen(argv[1],"r");
  int counter= 0;
  G.edgeListPtrs[counter++] = 0;
  for(i =0; i <G.numEdges; i++) {
    fscanf(in,"%ld %ld %ld",&v1,&(G.edgeList[i].tail),&(G.edgeList[i].weight));
    G.edgeList[i].head = v1;
    G.edgeList[i].weight = 1;
    if(v!= v1) { //This if() check will create problems with there are duplicates.
      G.edgeListPtrs[counter++] = i;
      v = v1;
    }
  }//End of for(i)
  fclose(in);
  
  G.edgeListPtrs[counter]=i;	
  time2 = omp_get_wtime();
  printf("Time to parse input file: %3.3lf\n", time2-time1);

  G.numEdges = G.numEdges / 2;  //Fix the number of edges (each edge represented twice

  //displayGraph(&G);

  /* Step 3: Run algorithm */
  long *vtxColor = (long *) malloc (G.numVertices * sizeof(long));
#pragma omp parallel for
  for(long i=0; i<G.numVertices; i++)
    vtxColor[i] = -1;
  
  //Call the Coloring Algorithm:
  
  //algoDistanceOneVertexColoring(g, vtxColor);
  algoDistanceOneVertexColoringNew(&G, vtxColor);

  /* Step 4: Clean up */
  free(vtxColor);
  free(G.edgeListPtrs);
  free(G.edgeList);
  
  return 0;
}


