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
#include "defs.h"

int main(int argc, char** argv) {
  
  /* Step 1: Parse command line arguments */
  if (argc < 2) {
    printf("Usage: %s <graph filename>\n", argv[0]);        
    exit(-1);
  }
  
  /* Step 2: Parse the graph in Matrix Market format */
  graph* G = (graph *) malloc (sizeof(graph)) ;
  parse_MatrixMarket(G, argv[1]);
  //parse_MatrixMarket_Sym_AsGraph(G, argv[1]);
  printf("Reading file in MM format...\n");
  //parse_Dimacs1Format(G, argv[1]);
  displayGraphCharacterists(G);


  //char *outfilename = (char *) calloc(500, sizeof(char));
  //outfilename = strcat(argv[1], ".dimacs1");
  //printf("Writing file to: %s\n", outfilename);
  //writeGraphDimacsOneFormatNewD(G, outfilename);
    


  /* Step 3: Run algorithm */
  long *Mate = (long *) malloc (G->numVertices * sizeof(long));
  
  for (int trial = 0; trial < 1; trial++) {
    printf("Trial: %d\n", trial+1);
#pragma omp parallel for
    for(long i=0; i<(G->numVertices); i++)
      Mate[i] = -1;

    //Call the Matching Algorithm:
    //algoEdgeApproxDominatingEdgesLinearSearchNew(G, Mate);
    //algoEdgeApproxDominatingEdgesSuitorSerialNew(G, Mate);
    algoEdgeApproxDominatingEdgesSuitorNew(G, Mate);
  }
  
  /* Step 4: Compute weight and cardinality of the matching */
  unsigned int NVer = G->numVertices;
  long NEdge        = G->numEdges;
  unsigned int NS   = G->sVertices;
  long *verPtr      = G->edgeListPtrs;  //Vertex Pointer
  edge *verInd      = G->edgeList;      //Vertex Index
  
  double weight = 0;
  int Check = 1;
  unsigned int cardinality = 0;
  for(unsigned int i=0; i<NS; i++) {
    if ( Mate[i] >= 0 ) {
      long adj1 = verPtr[i];
      long adj2 = verPtr[i+1];
      for(long j=adj1; j < adj2; j++)
	if( verInd[j].tail == Mate[i] ) {
          if (Mate[verInd[j].tail] != i) {
             Check = 0;
          }
	  weight += verInd[j].weight;
	  cardinality++;
	  break;
	} //End of inner if
    } //End of outer if
  } //End of for
  printf("***********************************************\n");
  printf("Weight      : %g \n", weight);
  printf("Cardinality : %d \n", cardinality);
  if (Check == 1)
     printf("SUCCESS: Valid matching\n");
  else
     printf("FAILURE: Not a valid matching\n");
  printf("***********************************************\n");
  
  /* Step 5: Clean up */    
  free(Mate);
  free(G->edgeListPtrs);
  free(G->edgeList);
  free(G);

  return 0;
}


