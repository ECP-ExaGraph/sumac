#ifndef _DEFS_H
#define _DEFS_H

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define MilanRealMax HUGE_VAL       // +INFINITY
#define MilanRealMin -MilanRealMax  // -INFINITY

typedef struct /* the edge data structure */
{
  //long id;
  long head;
  long tail;
  //int weight;
  double weight;
} edge;

typedef struct /* the graph data structure */
{ 
  long numVertices;        /* Number of columns                                */
  long sVertices;          /* Number of rows: Bipartite graph: number of S vertices; T = N - S */
  long numEdges;           /* Each edge stored twice, but counted once        */
  long * edgeListPtrs;    /* start vertex of edge, sorted, primary key        */
  edge * edgeList;        /* end   vertex of edge, sorted, secondary key      */
 } graph;

///COMPRESSED SPARSE COLUMN FORMAT: (edges stored only once)
typedef struct 
{ 
  long nRows;      /* Number of rows    */
  long nCols;      /* Number of columns */
  long nNNZ;               /* Number of nonzeros -- Each edge stored only once       */
  long *RowPtr;            /* Row pointer        */
  long *RowInd;	   /* Row index          */
  double *Weights;         /* Edge weights       */
} matrix_CSC;

///COMPRESSED SPARSE ROW FORMAT: (edges stored only once)
typedef struct 
{ 
  long nRows;      /* Number of rows    */
  long nCols;      /* Number of columns */
  long nNNZ;               /* Number of nonzeros -- Each edge stored only once       */
  long *ColPtr;            /* Col pointer        */
  long *ColInd;	   /* Col index          */
  double *Weights;         /* Edge weights       */
} matrix_CSR;

/* Utility functions */
void   prand(int howMany, double *randList); //Pseudorandom number generator (serial)
void   intializeCsrFromCsc(matrix_CSC*, matrix_CSR*);

int  removeEdges(int, int, edge *, int);
void SortEdgesUndirected(int, int, edge *, edge *, int *);
void SortNodeEdgesByIndex(int, edge *, edge *, int *);
void SortNodeEdgesByWeight(int, edge *, edge *, int *);
void writeGraphInMetisFormat(graph *, char *);
void displayGraphCharacterists(graph *);
void displayGraph(graph *);
void displayMatrixCsc(matrix_CSC *X);
void displayMatrixCsr(matrix_CSR *Y);
void displayMatrixProperties(matrix_CSC *X);

void sortEdgesMatrixCsc(matrix_CSC *X);
void sortEdgesMatrixCsr(matrix_CSR *Y);

/*
inline double timer() {
  return omp_get_wtime();
}
*/
double timer();
//File Parsers:
void parse_MatrixMarket(graph * G, char *fileName);
void parse_MatrixMarket_CSC(matrix_CSC * M, char *fileName);
void parse_Simple_CSC(matrix_CSC * M, char *fileName);
void parse_MatrixMarket_Sym_AsGraph(graph * G, char *fileName);
void parse_Dimacs1Format(graph * G, char *fileName);

//Coloring Algorithms:
void algoDistanceTwoVertexColoring( matrix_CSC *X, matrix_CSR *Y,  int *vtxColor, int *numColors,  int MaxDegree );


//Matching Algorithms:
void algoEdgeApproxDominatingEdgesSuitorSerialNew( graph* G, long *Mate);
void algoEdgeApproxInitialExtremeMatchingBipartiteSerial( graph *G, long *Mate );


#ifdef __cplusplus
extern "C" {
#endif
  graph * generateRMAT(int SCALE, int SCALE_WT, double a, double b, double c, double d);
  void exact_match(graph* G);
#ifdef __cplusplus
} /* closing brace for extern "C" */
#endif


#endif
