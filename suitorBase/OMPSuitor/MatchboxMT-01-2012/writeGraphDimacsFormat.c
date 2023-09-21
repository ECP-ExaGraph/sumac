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

/// Write file in DIMACS-9 Format: Directed graph
void writeGraphDimacsFormat(graph_t* G, char * filename)
{
  //Get the iterators for the graph:
  long NVer        = G->n;
  long NEdge       = G->m;
  attr_id_t *verPtr     = G->numEdges;   //Vertex Pointer: pointers to endV
  attr_id_t *verInd     = G->endV;       //Vertex Index: destination id of an edge (src -> dest)
  double *edgeWeight    = G->dbl_weight_e;
  printf("Vertices: %ld  Edges: %ld\n", NVer, NEdge/2);	 
  printf("Writing graph in DIMACS-9 format - Undirected graph - each edge represented twice\n");
  printf("Graph will be stored in file %s\n",filename);
  
  FILE *fout;
  fout = fopen(filename, "w");
  if (!fout) {
    printf("Could not open the file \n");
    exit(1);
  }
  fprintf(fout, "c File generated on Catamount\n");
  fprintf(fout, "p sp %ld %ld", NVer, NEdge);
  for (long v=0; v<NVer; v++) {
    long adj1 = verPtr[v];
    long adj2 = verPtr[v+1];
    //Browse the adjacency set of vertex v
    for(long k = adj1; k < adj2; k++ ) {
      fprintf(fout, "\na %ld %d %lf", v+1, (verInd[k]+1), edgeWeight[k]);
    } //End of for loop to traverse adjacency of v
  }
  fclose(fout);
}


/// Write file in DIMACS-9 Format: Directed graph
/* 
-------------------------------------------------------------------------
INPUT FORMAT FOR WMATCH:
-------------------------------------------------------------------------
   Graph I/O is performed by a generic graph library package, 
   so some of the fields are ignored by the "wmatch" code (but 
   you must include dummy fields in the input). 

   There are three types of lines: the first line, vertex lines, 
   and edge lines. The fields in each line type are as follows. 

   First line-> size edges U
      size: integer giving number of vertices
      edges: integer giving number of edges 
      U: character ``U'' or ``u'' specifying an undirected graph

   Vertex lines->  degree vlabel xcoord ycoord
      degree: edge degree of the vertex
      vlabel: vertex label (ignored--vertices are referred to by index)
      xcoord: integer x-coordinate location (ignored)
      ycoord: integer y-coordinate location (ignored) 

      *****Each vertex line is followed immediately by the lines 
      for all its adjacent edges (thus each edge appears twice, 
      once for each vertex).******

   Edge lines-> adjacent  weight
      adjacent: index (not vlabel) of the adjacent vertex
      weight: integer edge weight 
*/
void writeGraphDimacsOneFormat(graph_t* G, char * filename)
{
  //Get the iterators for the graph:
  long NVer             = G->n;
  long NEdge            = G->m;
  attr_id_t *verPtr     = G->numEdges;   //Vertex Pointer: pointers to endV
  attr_id_t *verInd     = G->endV;       //Vertex Index: destination id of an edge (src -> dest)
  double *edgeWeight    = G->dbl_weight_e;
  printf("Vertices: %ld  Edges: %ld\n", NVer, NEdge/2);	 
  printf("Writing graph in DIMACS-1 format - Undirected graph - each edge represented twice\n");
  printf("Graph will be stored in file %s\n",filename);
  
  long isolated = 0; 
  /* Check if there are isolated vertices */
  for (long v=0; v<NVer; v++) {
    if (verPtr[v] == verPtr[v+1])
      isolated++;
  }

  if (isolated == 0) {
    FILE *fout;
    fout = fopen(filename, "w");
    if (!fout) {
      printf("Could not open the file \n");
      exit(1);
    }
    //First Line:
    fprintf(fout, "%ld %ld U\n", NVer, NEdge);
    
    for (long v=0; v<NVer; v++) {
      long adj1 = verPtr[v];
      long adj2 = verPtr[v+1];
      //Vertex line: <degree> <vlabel> <xcoord> <ycoord>
      fprintf(fout, "%ld 3 0 0\n", (adj2 - adj1) );
      
      //Edge lines: <adjacent> <weight>
      for(long k = adj1; k < adj2; k++ ) {
	fprintf(fout, "%ld %d\n", (verInd[k]+1), (int)edgeWeight[k] );
      } //End of for loop to traverse adjacency of v
    }
    fclose(fout);
  }
  else {
    printf("There are %d isolated vertices. Graph not written to file.\n", isolated);
  }    
}//End of writeGraphDimacsOneFormat()

void writeGraphDimacsOneFormatNewD(graph* G, char * filename)
{
  //Get the iterators for the graph:
  long NVer     = G->numVertices;
  long NS       = G->sVertices;
  long NT       = NVer - NS;
  long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
  long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
  printf("NVer= %ld --  NE=%ld\n", NVer, NEdge);

  printf("Writing graph in DIMACS-1 format - Undirected graph - each edge represented twice\n");
  printf("Graph will be stored in file %s...\n",filename);
  
  long isolated = 0; 
  /* Check if there are isolated vertices */
  for (long v=0; v<NVer; v++) {
    if (verPtr[v] == verPtr[v+1])
      isolated++;
  }
  
  if (isolated == 0) {
    FILE *fout;
    fout = fopen(filename, "w");
    if (!fout) {
      printf("Could not open the file \n");
      exit(1);
    }
    //First Line:
    fprintf(fout, "%ld %ld U\n", NVer, NEdge);
    
    for (long v=0; v<NVer; v++) {
      long adj1 = verPtr[v];
      long adj2 = verPtr[v+1];
      //Vertex line: <degree> <vlabel> <xcoord> <ycoord>
      fprintf(fout, "%ld 3 0 0\n", (adj2 - adj1) );
      
      //Edge lines: <adjacent> <weight>
      for(long k = adj1; k < adj2; k++ ) {
	fprintf(fout, "%ld %d\n", (verInd[k].tail+1), (int)(verInd[k].weight) );
      } //End of for loop to traverse adjacency of v
    }
    fclose(fout);
    printf("Graph has been stored in file: %s\n",filename);
  }
  else {
    printf("There are %d isolated vertices. Graph not written to file.\n", isolated);
  }  
}//End of writeGraphDimacsOneFormatNewD()
