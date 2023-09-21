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

#include "coloringAndMatchingKernels.h"



int main(int argc, char** argv) {
    char *infilename, *graph_type;
    FILE* fp;
    graph_t* g;
    int curArgIndex;    
    int proc_pid;
  
    /* Step 1: Parse command line arguments */
    if (argc < 2) {
	printf("Usage: Executable -infile <graph filename> (-graph <graph type>)\n");        
        exit(-1);
    }

    curArgIndex = 0;
    infilename = (char *) calloc(500, sizeof(char));
    graph_type = (char *) calloc(500, sizeof(char));

    while (curArgIndex < argc) {      
      if (strcmp(argv[curArgIndex],"-infile")==0) {
	strcpy(infilename, argv[++curArgIndex]);
      }
      
      if (strcmp(argv[curArgIndex], "-graph")==0) {
	strcpy(graph_type, argv[++curArgIndex]);
      } 
      curArgIndex++; 
    }
    
    fp = fopen(infilename, "r");
    if (fp == NULL) {
      fprintf(stderr, "Error! Could not open input file. Exiting ...\n");
      exit(-1);
    }
    fclose(fp);
    
    graph_ext_check(infilename, graph_type);
    
    /* Step 2: Generate graph */
    g = (graph_t *) malloc(sizeof(graph_t));
    graph_gen(g, infilename, graph_type);
    
    //writeGraphDimacsFormat(g, "Dimacs.gr");

    /* Step 3: Run algorithm */
    long *Mate = (long *) malloc (g->n * sizeof(long));

    for (int try = 0; try < 3; try++) {
      printf("Trial: %d\n", try+1);
#pragma omp parallel for
      for(long i=0; i<(g->n); i++)
	Mate[i] = -1;
      
      //Call the Matching Algorithm:
      algoEdgeApproxDominatingEdgesLinearSearch(g, Mate);     
    }

    long NVer         = g->n;
    long NEdge        = g->m;
    attr_id_t *verPtr = g->numEdges;   //Vertex Pointer: pointers to endV
    attr_id_t *verInd = g->endV;       //Vertex Index: destination id of an edge (src -> dest)
    int *edgeWeight   = g->int_weight_e;   //Edge Weight - Int
    //double *edgeWeight   = g->dbl_weight_e;   //Edge Weight - Double
    
    double weight = 0;
    long cardinality = 0;
    //#pragma omp parallel for
    for(long i=0; i<NVer; i++) {
      if ( Mate[i] >= 0 ) {
	long adj1 = verPtr[i];
	long adj2 = verPtr[i+1];
	for(long j=adj1; j < adj2; j++)
	  if(verInd[j] == (Mate[i])) {
	    //#pragma omp critical
	    {
	      weight = weight + edgeWeight[j];
	      cardinality++;
	      break;
	    }
	  } //End of inner if
      } //End of outer if
    } //End of for
    printf("Weight      : %lf \n",weight/2);	    
    printf("Cardinality : %ld \n", cardinality/2);	    
    printf("***********************************************\n");
    
 
    for(long i=0; i<(g->n); i++)
      Mate[i] = -1;
    //Call the Matching Algorithm - SERIAL:
    algoEdgeApproxDominatingEdgesLinearSearchSerial(g, Mate);

    weight = 0;
    cardinality = 0;
    //#pragma omp parallel for
    for(long i=0; i<NVer; i++) {
      if ( Mate[i] >= 0 ) {
	long adj1 = verPtr[i];
	long adj2 = verPtr[i+1];
	for(long j=adj1; j < adj2; j++)
	  if(verInd[j] == Mate[i]) {
	    //#pragma omp critical
	    {
	      weight += edgeWeight[j];
	      cardinality++;
	      break;
	    }
	  } //End of inner if
      } //End of outer if
    } //End of for
    printf("***********************************************\n");
    printf("Serial Algorithm:\n");
    printf("Weight      : %lf \n",weight/2);	    
    printf("Cardinality : %ld \n", cardinality/2);	    
    printf("***********************************************\n");
    
    /* Step 4: Clean up */    
    free(Mate);
    free(g);
    return 0;
}


