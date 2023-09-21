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
   
    char *outfilename = (char *) calloc(500, sizeof(char));
    outfilename = strcat(infilename, ".out");
    printf("Writing file to: %s\n", outfilename);
    writeGraphDimacsOneFormat(g, outfilename);
    
    

    /* Step 3: Run algorithm */
    long *vtxColor = (long *) malloc (g->n * sizeof(long));
#pragma omp parallel for
    for(long i=0; i<(g->n); i++)
      vtxColor[i] = -1;
    
    //Call the Coloring Algorithm:
    algoDistanceOneVertexColoring(g, vtxColor);
    
    /*
    //SERIAL COLORING:
    #pragma omp parallel for
    for(long i=0; i<(g->n); i++)
    vtxColor[i] = -1;
    */
    //printf("Results from serial coloring:\n");
    //printf("***********************************************\n");
    //algoDistanceOneVertexColoringSerial(g, vtxColor);
    //printf("***********************************************\n");
      
    /* Step 4: Clean up */
    free(vtxColor);
    free(infilename);   
    free(graph_type);
    
    free_graph(g);
    free(g);
    return 0;

    /*
    printf("About to compute degree distributions:\n");
    long const MaxDegree = 38200;
    long *degCount = (long *) malloc (MaxDegree * sizeof(long));
#pragma omp parallel for
    for (long v=0; v<MaxDegree; v++) {
      degCount[v] = 0;
    }
    long NVer         = g->n;
    attr_id_t *verPtr = g->numEdges;   //Vertex Pointer: pointers to endV
#pragma omp parallel for
    for (long v=0; v<NVer; v++) {
      long degree = verPtr[v+1] - verPtr[v];
#pragma omp atomic
      degCount[degree]++; //Frequency of that degree
    }
    
    printf("Writing distributions to file DegDist.dat\n");
    FILE *fout;
    fout = fopen("DegDist.dat", "w");
    if (!fout) {
      printf("Could not open the file \n");
      exit(1);
    }
    for (long v=0; v<MaxDegree; v++) {
      fprintf(fout, "%ld\t%ld\n", v, degCount[v]);
    }
    fclose(fout);
    printf("Done writing distributions to file DegDist.dat\n");
    free(degCount);
    */



}


