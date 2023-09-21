/*---------------------------------------------------------------------------*/
/*                                                                           */
/*                          Mahantesh Halappanavar                           */
/*                        High Performance Computing                         */
/*                Pacific Northwest National Lab, Richland, WA               */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*                                                                           */
/* Copyright (C) 2010 Mahantesh Halappanavar                                 */
/*                                                                           */
/* This program is free software; you can redistribute it and/or             */
/* modify it under the terms of the GNU General Public License               */
/* as published by the Free Software Foundation; either version 2            */
/* of the License, or (at your option) any later version.                    */
/*                                                                           */
/* This program is distributed in the hope that it will be useful,           */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/* GNU General Public License for more details.                              */
/*                                                                           */
/* You should have received a copy of the GNU General Public License         */
/* along with this program; if not, write to the Free Software               */
/* Foundation, Inc., 59 Temple Place-Suite 330,Boston,MA 02111-1307,USA.     */
/*                                                                           */
/*---------------------------------------------------------------------------*/

#include "defs.h"

void intializeCsrFromCsc(matrix_CSC *X, matrix_CSR *Y) 
{
    printf("Building CSR format from CSC format...\n");
    double time1, time2;

    //Get the iterators for the graph:
    long NRows       = X->nRows;
    long NCols       = X->nCols;
    long NNnz        = X->nNNZ;
    long *rowPtr     = X->RowPtr;  //Row Pointer
    long *rowInd     = X->RowInd;  //Row Index 
    double *weights = X->Weights; //NNZ value
    printf("|Rows|= %ld, |Cols|= %ld, |E|= %ld \n", NRows, NCols, NNnz);
    
    //Every edge stored ONLY ONCE!
    long *colPtr         = (long*)         malloc ((NRows+1) * sizeof(long));    /* Col pointer  */
    long *colInd = (long*) malloc (NNnz      * sizeof(long));    /* Col index    */
    double *weightsC     = (double*)       malloc (NNnz      * sizeof(double)); /* edge weight  */

    //////Build the ColPtr Array: 
    time1 = timer();
#pragma omp parallel for
    for (long i=0; i<(NRows+1); i++) {
      colPtr[i] = 0;
    }
#pragma omp parallel for
    for (long i=0; i<NNnz; i++) {
      colPtr[rowInd[i]+1]++; //Count degree of row i
    }
    //////Cumulative addition 
    time1 = timer();
    for (long i=0; i<NRows; i++) {
      colPtr[i+1] += colPtr[i]; //Prefix Sum
    }
    //The last element of Cumulative will hold the total number of characters
    time2 = timer();
    printf("Built colPtrs in %2.3lf seconds\n", time2 - time1);
    printf("Sanity Check: |E| = %ld, colPtr[NRows]= %ld\n", NNnz, colPtr[NRows]);    
    
    printf("Building edgeList...\n");
    time1 = timer();
    //Keep track of how many edges have been added for a vertex:
    long *added    = (long *)  malloc( NRows  * sizeof(long));
#pragma omp parallel for
    for (long i = 0; i < NRows; i++)
      added[i] = 0;
    
#pragma omp parallel for
    for (long i=0; i<NCols; i++){
      long adj1 = rowPtr[i];
      long adj2 = rowPtr[i+1];
      for(long j=adj1; j<adj2; j++) {
	long rowId = rowInd[j];
	long Where = colPtr[rowId] + __sync_fetch_and_add(&added[rowId], 1);
	//printf("(%d,%d) in %d\t", head, tail, Where);
	colInd[Where]    = i;  //Add the col id
	weightsC[Where]  = weights[j];
	//added[rowId]++;
      }
    }
    time2 = timer();
    printf("Time for building edgeList = %lf\n", time2 - time1);

    Y->nRows    = NRows;
    Y->nCols    = NCols;
    Y->nNNZ     = NNnz;
    Y->ColPtr   = colPtr;
    Y->ColInd   = colInd;
    Y->Weights  = weightsC;
    free(added);
}

void displayGraph(graph *G) {
  long    NV        = G->numVertices;  
  long    NE        = G->numEdges;
  long    *vtxPtr   = G->edgeListPtrs;
  edge    *vtxInd   = G->edgeList;  
  printf("|V|= %ld, |E|= %ld \n", NV, NE);
  printf("***********************************");
  for (long i = 0; i < NV; i++) {
    long adj1 = vtxPtr[i];
    long adj2 = vtxPtr[i+1];
    printf("\nVtx: %ld [%ld]: ",i+1,adj2-adj1);
    for(long j=adj1; j<adj2; j++) {      
      printf("(%ld,%g) ", vtxInd[j].tail+1, vtxInd[j].weight);
    }
  }
  printf("\n***********************************");
}

void displayGraphCharacterists(graph *G) {
  long    sum = 0, sum_sq = 0;
  double  average, avg_sq, variance, std_dev;
  long    maxDegree = 0;
  long    isolated  = 0;
  long    NS        = G->sVertices;    
  long    NV        = G->numVertices;
  long    NE        = G->numEdges;
  long    *vtxPtr   = G->edgeListPtrs;
  long    tNV       = NV; //Number of vertices    
  if ( NS > 0 ) //Biparite graph
    tNV = NS;    

  long degree;
  for (long i = 0; i < tNV; i++) {
    degree = vtxPtr[i+1] - vtxPtr[i];
    sum_sq += degree*degree;
    sum    += degree;
    if (degree > maxDegree)
      maxDegree = degree;
    if ( degree == 0 )
      isolated++;
  }
  
  average  = (double) sum / tNV;
  avg_sq   = (double) sum_sq / tNV;
  variance = avg_sq - (average*average);
  std_dev  = sqrt(variance);
  
  printf("*******************************************\n");
  printf("Number of S vertices :  %ld\n", NS);
  printf("Number of vertices   :  %ld\n", NV);
  printf("Number of edges      :  %ld\n", NE);
  printf("Maximum out-degree is:  %ld\n", maxDegree);
  printf("Average out-degree is:  %lf\n", average);
  printf("Expected value of X^2:  %lf\n", avg_sq);
  printf("Variance is          :  %lf\n", variance);
  printf("Standard deviation   :  %lf\n", std_dev);
  printf("Isolated (S)vertices :  %ld (%3.2lf%%)\n", isolated, ((double)isolated/tNV)*100);
  printf("Density              :  %lf%%\n",((double)NE/(NV*NV))*100);
  printf("*******************************************\n");    
}

void displayMatrixCsc(matrix_CSC *X) 
{
  //Get the iterators for the graph:
  long NRows       = X->nRows;
  long NCols       = X->nCols;
  long NNnz        = X->nNNZ;
  long *rowPtr     = X->RowPtr;  //Row Pointer
  long *rowInd     = X->RowInd;  //Row Index 
  double *weights = X->Weights; //NNZ value
  
  printf("Input Matrix:\n");
  printf("*******************************************\n");
  for(long c=0; c<NCols; c++){
    long adj1 = rowPtr[c];
    long adj2 = rowPtr[c+1];	
    printf("Col[%d: %d]: ", c+1, adj2 - adj1);
    for(long k = adj1; k < adj2; k++ ) {
      printf("%d (%lf) ",rowInd[k]+1, weights[k]);
    }
    printf("\n");
  }
  printf("*******************************************\n");
}

void displayMatrixCsR(matrix_CSR *Y) {
  //Get the iterators for the graph:
  long NRows       = Y->nRows;
  long NCols       = Y->nCols;
  long NNnz        = Y->nNNZ;
  long *colPtr     = Y->ColPtr;  //Row Pointer
  long *colInd     = Y->ColInd;  //Row Index 
  double *weights = Y->Weights; //NNZ value
  
  printf("Input Matrix:\n");
  printf("*******************************************\n");
  for(long r=0; r<NRows; r++){
    long adj1 = colPtr[r];
    long adj2 = colPtr[r+1];	
    printf("Row[%d: %d]: ", r+1, adj2 - adj1);
    for(long k = adj1; k < adj2; k++ ) {
      printf("%d (%lf) ",colInd[k]+1, weights[k]);
    }
    printf("\n");
  }
  printf("*******************************************\n");
}

void displayMatrixProperties(matrix_CSC *X) {
  long    sum = 0, sum_sq = 0;
  double average, avg_sq, variance, std_dev;
  long    maxDegree = 0;
  long    isolated  = 0;
  //Get the iterators for the graph:
  long NRows       = X->nRows;
  long NCols       = X->nCols;
  long NNnz        = X->nNNZ;
  long *rowPtr     = X->RowPtr;  //Row Pointer
  long *rowInd     = X->RowInd;  //Row Index 
  double *weights = X->Weights; //NNZ value
  
  long tNV         = NCols; //Number of columns
#pragma omp parallel for
  for (long i = 0; i < tNV; i++) {
    long degree = rowPtr[i+1] - rowPtr[i];
    sum_sq += degree*degree;
    sum    += degree;
    if (degree > maxDegree)
      maxDegree = degree;
    if ( degree == 0 )
      __sync_fetch_and_add(&isolated, 1); 
  }
  
  average  = (double) sum / tNV;
  avg_sq   = (double) sum_sq / tNV;
  variance = avg_sq - (average*average);
  std_dev  = sqrt(variance);
  
  printf("*******************************************\n");
  printf("Compressed Sparse Column:");
  printf("*******************************************\n");
  printf("Number of Rows       :  %ld\n", NRows);
  printf("Number of Columns    :  %ld\n", NCols);
  printf("Number of Non-zeros  :  %ld\n", NNnz);
  printf("Maximum out-degree is:  %ld\n", maxDegree);
  printf("Average out-degree is:  %lf\n",average);
  printf("Expected value of X^2:  %lf\n",avg_sq);
  printf("Variance is          :  %lf\n",variance);
  printf("Standard deviation   :  %lf\n",std_dev);
  printf("Isolated (S)vertices :  %ld (%3.2lf%%)\n", isolated, ((double)isolated/tNV)*100);
  printf("Density              :  %lf%%\n",((double)NNnz/(NRows*NCols))*100);
  printf("*******************************************\n");    
}

void sortEdgesMatrixCsc(matrix_CSC *X)
{
  printf("Within function sortEdges()");
  double time1, time2;
  time1 = timer();
  //Get the iterators for the graph:
  long NRows       = X->nRows;
  long NCols       = X->nCols;
  long NNnz        = X->nNNZ;
  long *rowPtr     = X->RowPtr;  //Row Pointer
  long *rowInd     = X->RowInd;  //Row Index 
  double *weights = X->Weights; //NNZ value

  //Create New vectors to store sorted information
  long *NrowInd   = (long*)    malloc (NNnz * sizeof(long)); 
  double *Nweights = (double*) malloc (NNnz * sizeof(double));
  
  //#pragma mta assert no dependence
  for (long i = 0; i < NCols; i++) {
    long *edges1 = rowInd  + rowPtr[i];
    long *edges2 = NrowInd + rowPtr[i];
    double *weight1 = weights  + rowPtr[i];
    double *weight2 = Nweights + rowPtr[i];      
    
    long size    = rowPtr[i+1] - rowPtr[i];
    /* Merge Sort */
    for (long skip = 2; skip < 2 * size; skip *= 2) {
      for (long sect = 0; sect < size; sect += skip)  {
	long j = sect;
	long l = sect;
	long half_skip = skip / 2;
	long k = sect + half_skip;
	
	long j_limit = (j + half_skip < size) ? j + half_skip : size;
	long k_limit = (k + half_skip < size) ? k + half_skip : size;
	
	while ((j < j_limit) && (k < k_limit)) {
	  if   (edges1[j] < edges1[k]) {
	    edges2[l] = edges1[j];
	    weight2[l]= weight1[j];		      
	    j++; l++;		      
	  }
	  else {
	    edges2[l] = edges1[k]; 
	    weight2[l]= weight1[k];
	    k++; l++;
	  }
	}	      
	while (j < j_limit) {
	  edges2[l] = edges1[j]; 
	  weight2[l]= weight1[j];
	  j++; l++;
	}
	while (k < k_limit) {
	  edges2[l] = edges1[k]; 
	  weight2[l]= weight1[k];
	  k++; l++;
	}
      }	  
      long *tmpI = edges1;
      edges1 = edges2;
      edges2 = tmpI;
      
      double *tmpD = weight1;
      weight1 = weight2;
      weight2 = tmpD;
    }      
    // result is in list2, so move to list1
    if (edges1 == NrowInd + rowPtr[i])
      for (long j = rowPtr[i]; j < rowPtr[i+1]; j++) {
	rowInd[j]  = NrowInd[j];
	weights[j] = Nweights[j];	      
      }
  }
  time2 = timer();
  printf("Time to sort:  %3.3lf seconds\n", time2 - time1);
  
  free(NrowInd);
  free(Nweights);  
}

void sortEdgesMatrixCsr(matrix_CSR *Y)
{
  printf("Within function sortEdges()");
  double time1, time2;
  time1 = timer();
  //Get the iterators for the graph:
  long NRows       = Y->nRows;
  long NCols       = Y->nCols;
  int NNnz        = Y->nNNZ;
  int *colPtr     = Y->ColPtr;  //Row Pointer
  int *colInd     = Y->ColInd;  //Row Index 
  double *weights = Y->Weights; //NNZ value
  
  //Create New vectors to store sorted information
  int *NcolInd   = (int*)    malloc (NNnz * sizeof(int)); 
  double *Nweights = (double*) malloc (NNnz * sizeof(double));
  
  //#pragma mta assert no dependence
  
  for (int i = 0; i < NRows; i++) {
    int *edges1 = colInd  + colPtr[i];
    int *edges2 = NcolInd + colPtr[i];
    double *weight1 = weights  + colPtr[i];
    double *weight2 = Nweights + colPtr[i];      
    
    int size    = colPtr[i+1] - colPtr[i];
    /* Merge Sort */
    for (int skip = 2; skip < 2 * size; skip *= 2) {
      for (int sect = 0; sect < size; sect += skip)  {
	int j = sect;
	int l = sect;
	int half_skip = skip / 2;
	int k = sect + half_skip;
	
	int j_limit = (j + half_skip < size) ? j + half_skip : size;
	int k_limit = (k + half_skip < size) ? k + half_skip : size;
	
	while ((j < j_limit) && (k < k_limit)) {
	  if   (edges1[j] < edges1[k]) {
	    edges2[l] = edges1[j];
	    weight2[l]= weight1[j];		      
	    j++;  l++;		      
	  }
	  else {
	    edges2[l] = edges1[k]; 
	    weight2[l]= weight1[k];
	    k++; l++;
	  }
	}	      
	while (j < j_limit) {
	  edges2[l] = edges1[j]; 
	  weight2[l]= weight1[j];
	  j++; l++;
	}
	while (k < k_limit) {
	  edges2[l] = edges1[k]; 
	  weight2[l]= weight1[k];
	  k++; l++;
	}
      }	  
      int *tmpI = edges1;
      edges1 = edges2;
      edges2 = tmpI;
      
      double *tmpD = weight1;
      weight1 = weight2;
      weight2 = tmpD;
    }      
    // result is in list2, so move to list1
    if (edges1 == NcolInd + colPtr[i])
      for (int j = colPtr[i]; j < colPtr[i+1]; j++) {
	colInd[j]  = NcolInd[j];
	weights[j] = Nweights[j];	      
      }
  }
  time2 = timer();
  printf("Time to sort:  %3.3lf seconds\n", time2 - time1);
  
  free(NcolInd);
  free(Nweights);  
}
