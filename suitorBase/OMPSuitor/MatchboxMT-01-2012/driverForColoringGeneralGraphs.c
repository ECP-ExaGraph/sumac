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
  //parse_MatrixMarket(G, argv[1]);
  parse_MatrixMarket_Sym_AsGraph(G, argv[1]);
  displayGraphCharacterists(G);
  /* Step 3: Run algorithm */
  int *vtxColor = (int *) malloc (G->numVertices * sizeof(int));
  
  for (int trial = 0; trial < 1; trial++) {
    printf("Trial: %d\n", trial+1);
#pragma omp parallel for
    for(long i=0; i<(G->numVertices); i++)
      vtxColor[i] = -1;

    //Call the Coloring Algorithm:
    algoDistanceOneVertexColoringNew(G, vtxColor);
  }
  printf("Cleaning memory...\n");
  /* Step 4: Clean up */
  //free(vtxColor);
  //free(G->edgeListPtrs);
  //free(G->edgeList);
  //free(G);
  
  return 0;

}


