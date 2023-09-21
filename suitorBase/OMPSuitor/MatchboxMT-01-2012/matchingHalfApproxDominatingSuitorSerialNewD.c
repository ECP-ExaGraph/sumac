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

/* Original Source Code: Fredrik Manne                                       */

#include "coloringAndMatchingKernels.h"
#include "defs.h"

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////  DOMINATING EDGE ALGORITHM  ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
void algoEdgeApproxDominatingEdgesSuitorSerialNew( graph* G, long *Mate)
{
  double time1=0, time2=0, totalTime=0;
  //Get the iterators for the graph:
  //Get the iterators for the graph:
  long NVer     = G->numVertices;
  long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
  long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd  = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
  printf("NV= %ld  NE=%ld\n", NVer, NEdge);

  //Allocate Data Structures:
  time1 = get_seconds();
  double *ws   = (double *) malloc (NVer * sizeof(double)); //ws[i] stores the weight of (ws[i],i)
  if( ws==NULL ) {
    printf("Not enough memory to allocate the internal variables ws\n");
    exit(1);
  }
  for (long i=0; i<NVer; i++) {
    ws[i]   = 0.0f;       //Set current weight of best suitor to zero
  }
  time1  = get_seconds() - time1;

  time2 = get_seconds();
  long i, j, next_vertex, partner;
  int done = 0;
  double heaviest=0;
  for(long x=0; x<NVer; x++) {  // Loop over vertices
    done = 0;                   //FALSE;
    i = x;                      // i is the vertex we are trying to match
    while (!done) {             // Trying to find a (possibly new) partner for i
      heaviest = 0.0;
      for(j=verPtr[i]; j<verPtr[i+1]; j++) { // Loop over neighbors of vertex i
        long y = verInd[j].tail;            // y is the current neighbor of i
        if ((verInd[j].weight > heaviest) && (ws[y] < verInd[j].weight)) {// Check if w(i,y) is the best so far, and if it is a better option for y
          heaviest = verInd[j].weight;      // Store the best so far
          partner = y;
        }
      } // loop over neighbors
      done = 1;
      if (heaviest > 0) {            // Test if we found a possible partner
        if (heaviest > ws[partner]) {
          if (Mate[partner] != -1) {
            next_vertex = Mate[partner];
            done = 0;
          }
          Mate[partner] = i;       // i is now the current suitor of s
          ws[partner] = heaviest;  // The weight of the edge (i,partner) 
        }
        else {   // can no longer use the result for node i, must find new partner
          done = 0;
          next_vertex = i;
        }
      }
      if (!done) {  // Continue with the next vertex
        i = next_vertex;
      }
    } // while not done
  } // loop over vertices
  time2  = get_seconds() - time2;
  totalTime += time2;

  printf("***********************************************\n");
  printf("Time for Phase-1           : %lf sec\n", time1);
  printf("Time for Phase-2           : %lf sec\n", totalTime);
  printf("Total Time                 : %lf sec\n", totalTime+time1);
  printf("***********************************************\n");

  //Clean Up:
  free(ws);
} //End of algoEdgeApproxDominatingEdgesLinearSearch

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////  DOMINATING EDGE ALGORITHM  ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//Assumes Edges are sorted in descending order
void algoEdgeApproxDominatingEdgesSuitorSerialNewSorted( graph* G, long *Mate)
{
  printf("Within algoEdgeApproxDominatingEdgesSuitorSerialNewSorted()\n");
  printf("Warning: Assumes that edges are in descending order of weights \n");
  double time1=0, time2=0, totalTime=0;
  //Get the iterators for the graph:
  //Get the iterators for the graph:
  long NVer     = G->numVertices;
  long NS       = G->sVertices;
  long NT       = NVer - NS;
  long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
  long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
  printf("NS= %ld  NT=%ld  NE=%ld\n", NS, NT, NEdge);

  //Allocate Data Structures:
  time1 = get_seconds();
  double *ws   = (double *) malloc (NVer * sizeof(double)); //ws[i] stores the weight of (ws[i],i)
  long   *next = (long *)   malloc (NVer * sizeof(long));   //next[i] points to where vertex i should start looking for a candidate to match with.
  if( (ws==NULL)||(next==NULL) ) {
    printf("Not enough memory to allocate the internal variables ws and next\n");
    exit(1);
  }
  //Initialization:
  for (long i=0; i<NVer; i++) {
    ws[i]   = 0.0f;       //Set current weight of best suitor to zero
    next[i] = verPtr[i];  //Set where in edge list to start searching for a partner
  }
  time1  = get_seconds() - time1;

  time2 = get_seconds();
  long i, e, y, next_vertex;
  int done;
  for(long x=0; x<NVer; x++) {  // Loop over vertices
    done = 0;                   //FALSE;
    i = x;                      // i is the vertex we are trying to match
    while (!done) {             // Trying to find a (possibly new) partner for i
      y = next[i];              // y points to the next possible neighbor (in the edge list of i) that i can match with
      e = verInd[y].tail;       // Get the id of the neighbor that we will try to match with

      while((y < verPtr[i+1]) && (verInd[y].weight <= ws[e])) { // Stop if there are no more edges or if we found a candidate
         y++;                   // Move to next position in edge list
         e = verInd[y].tail;    // Get id of neighbor
      }
      done = 1;
      if (y < verPtr[i+1]) {            // Test if we found a possible partner
        next[i] = y+1;               // Set where to search from next time i needs a partner
        ws[e]   = verInd[y].weight;  // Store the weight of (i,e) as the weight given by the current suitor for e

        if (Mate[e] != -1) {          // True if e already had a suitor
          next_vertex = Mate[e];     // Pick up the old suitor
          Mate[e] = i;               // i is now the current suitor of e
          i = next_vertex;           // Continue and try to match the previous suitor of e
          done = 0;
        }
        else {
          Mate[e] = i;               // Set i as the current suitor of e
        }
      }
    } // while not done
  } // loop over vertices
  time2  = get_seconds() - time2;
  totalTime += time2;

  printf("***********************************************\n");
  printf("Time for Phase-1           : %lf sec\n", time1);
  printf("Time for Phase-2           : %lf sec\n", totalTime);
  printf("Total Time                 : %lf sec\n", totalTime+time1);
  printf("***********************************************\n");

  //Clean Up:
  free(ws);
  free(next);

} //End of algoEdgeApproxDominatingEdgesLinearSearch

