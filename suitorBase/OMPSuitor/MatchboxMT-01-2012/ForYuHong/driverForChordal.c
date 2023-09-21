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

#include<stdlib.h>
#include<stdio.h>

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
  
  /* Step 3: Run algorithm */
  
  int *chordalEdge = (int *) malloc (g->m * 2 * sizeof(int));
#pragma omp parallel for
  for(long i=0; i<(g->m * 2); i++)
    chordalEdge[i] = 0;
  
  for (int nt=0; nt<=5; nt++) {
    int numT = (int) pow(2, nt);
    
    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    printf("Running job with %d threads\n", numT);
    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    
    printf("***********************************************\n");
    printf("(NON-OPTIMAL) Compute Maximal Chordal Subgraph:\n");
    printf("***********************************************\n");
    //Call the Coloring Algorithm:
    omp_set_num_threads(numT);
    algoMaximalChordalSubGraph(g, chordalEdge);      
    
    for (int trial=0; trial<3; trial++) {	
      printf("\n TRIAL %d\n", trial+1);  
#pragma omp parallel for
      for(long i=0; i<(g->m * 2); i++)
	chordalEdge[i] = 0;	
      printf("***********************************************\n");
      printf("(OPTIMAL) Compute Maximal Chordal Subgraph:\n");
      printf("***********************************************\n");
      //Call the Coloring Algorithm:
      omp_set_num_threads(numT);
      algoMaximalChordalSubGraphOpt(g, chordalEdge);
    }//End of for(num Trials)       
    printf("\n\n");
    
  }//End of for(num Threads)
  
  /* Step 4: Clean up */
  free(chordalEdge);
  free(infilename);   
  free(graph_type);
  
  free_graph(g);
  free(g);
  return 0;  
}


