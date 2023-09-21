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

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////  DISTANCE ONE COLORING      ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
void algoDistanceOneVertexColoringSerial(graph_t* G, long *vtxColor)
{
  double time1=0, totalTime=0;
  //Get the iterators for the graph:
  long NVer        = G->n;
  long NEdge       = G->m;
  attr_id_t *verPtr     = G->numEdges;   //Vertex Pointer: pointers to endV
  attr_id_t *verInd     = G->endV;       //Vertex Index: destination id of an edge (src -> dest)
  printf("Vertices: %ld  Edges: %ld\n", NVer, NEdge/2);
  
  const int MaxDegree = 132; //Increase if number of colors is larger    
  
  /////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////// START THE WHILE LOOP ///////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  long nColors = 0; //Will hold the maximum number of colors used
  long *Mark = (long *) malloc ( MaxDegree * NVer * sizeof(long) );
  if( Mark == NULL ) {
    printf("Not enough memory to allocate for Mark \n");
    exit(1);
  }
  //Initialize Mark with -1
  for (long i=0; i<MaxDegree*NVer; i++)
     Mark[i]= -1;	    

  time1 = get_seconds();
  for (long v=0; v<NVer; v++) {
      long StartIndex = v*MaxDegree; //Location in Mark
      long adj1 = verPtr[v];
      long adj2 = verPtr[v+1];	    
      long adjColor = -1;
      long maxColor = -1;
      //printf("v= %d, StartVertex= %d\n",v, StartIndex);	
      
      //Browse the adjacency set of vertex v
      for(long k = adj1; k < adj2; k++ ) {
	if ( v == verInd[k] ) //Self-loops
  	  continue;
	adjColor =  vtxColor[verInd[k]];
	if ( adjColor >= 0 ) {
	  Mark[StartIndex+adjColor] = v;
	  //Find the largest color in the neighborhood
	  if ( adjColor > maxColor )
	    maxColor = adjColor;
	}
      } //End of for loop to traverse adjacency of v
      long myColor;
      for (myColor=0; myColor<=maxColor; myColor++) {
	if ( Mark[StartIndex+myColor] != v )
	  break;	    
      }		
      if (myColor == maxColor)
	myColor++; /* no available color with # less than cmax */	 
      vtxColor[v] = myColor; //Color the vertex 	
      //Keep a global counter for number of colors used
      if (myColor > nColors) {
	nColors++; //Add a new color
      }
    } //End of outer for loop: for each vertex 
    time1  = get_seconds() - time1;
    totalTime += time1;	
    printf("Number of colors used   : %ld \n",nColors);    
    printf("Time taken for Coloring : %lf sec.\n", time1);  
  
  /////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////// VERIFY THE COLORS /////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  //Verify Results and Cleanup
  long nConflicts = 0;
  for (int v=0; v < NVer; v++ ) {
    int adj1 = verPtr[v]; 
    int adj2 = verPtr[v+1];
    //Browse the adjacency set of vertex v
    for(int k = adj1; k < adj2; k++ ) {
      if ( v == verInd[k] ) //Self-loops
  	  continue;
      if ( vtxColor[v] == vtxColor[verInd[k]] ) {
	nConflicts++; //increment the counter
      }		    
    }//End of inner for loop: w in adj(v)
  }//End of outer for loop: for each vertex
  
  nConflicts = nConflicts / 2; //Have counted each conflict twice    
  if (nConflicts > 0)
    printf("Check - WARNING: Number of conflicts detected after resolution: %ld \n", nConflicts);
  else
    printf("Check - SUCCESS: No conflicts exist\n");
  //Clean Up:
  free(Mark); //Free memory from Mark
} //End of algoEdgeApproxDominatingEdgesLinearSearch


