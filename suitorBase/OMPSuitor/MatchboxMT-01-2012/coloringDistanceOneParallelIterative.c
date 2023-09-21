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
//#pragma vector aligned 
//#pragma vector nontemporal
void algoDistanceOneVertexColoring(graph_t* G, long *vtxColor)
{
#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int nprocs   = omp_get_num_procs();
    int tid = omp_get_thread_num();
    if (tid == 0)
      printf("Number of threads: %d\n Number of procs: %d\n", nthreads, nprocs);
  }
  double time1=0, time2=0, totalTime=0;
  //Get the iterators for the graph:
  long NVer         = G->n;
  long NEdge        = G->m;
  attr_id_t *verPtr = G->numEdges;   //Vertex Pointer: pointers to endV
  attr_id_t *verInd = G->endV;       //Vertex Index: destination id of an edge (src -> dest)
  printf("Vertices: %ld  Edges: %ld\n", NVer, NEdge/2);
  
  const int MaxDegree = 128; //Increase if number of colors is larger    
  
  //Build a vector of random numbers
  double *randValues = (double*) malloc (NVer * sizeof(double));
  if( randValues == NULL ) {
    printf("Not enough memory to allocate for random numbers \n");
    exit(1);
  }
  /* Initialize RNG stream */
  int* stream, seed;
  seed = 12345;
  stream = init_sprng(0, 0, 1, seed, SPRNG_DEFAULT);

#pragma omp parallel for
  for (long i=0; i<NVer; i++)
    randValues[i] = NVer * NEdge * sprng(stream); //Some large value
  free_sprng(stream);

  //The Queue Data Structure for the storing the vertices 
  //   the need to be colored/recolored
  //Have two queues - read from one, write into another
  //   at the end, swap the two.
  long *Q    = (long *) malloc (NVer * sizeof(long));
  long *Qtmp = (long *) malloc (NVer * sizeof(long));
  long *Qswap;    
  if( (Q == NULL) || (Qtmp == NULL) ) {
    printf("Not enough memory to allocate for the two queues \n");
    exit(1);
  }
  long QTail=0;    //Tail of the queue 
  long QtmpTail=0; //Tail of the queue (implicitly will represent the size)
  
#pragma omp parallel for
  for (long i=0; i<NVer; i++) {
      Q[i]= i;     //Natural order
      Qtmp[i]= -1; //Empty queue
  }
  QTail = NVer;	//Queue all vertices
  /////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////// START THE WHILE LOOP ///////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  long nConflicts = 0; //Number of conflicts 
  long nLoops = 0;     //Number of rounds of conflict resolution
  long *Mark = (long *) malloc ( MaxDegree * NVer * sizeof(long) );
  if( Mark == NULL ) {
    printf("Not enough memory to allocate for Mark \n");
    exit(1);
  }
#pragma omp parallel for
  for (long i=0; i<MaxDegree*NVer; i++)
     Mark[i]= -1;	    

  printf("Results from parallel coloring:\n");
  printf("***********************************************\n");
  do {       
    ///////////////////////////////////////// PART 1 ////////////////////////////////////////
    //Color the vertices in parallel - do not worry about conflicts
    printf("** Iteration : %ld \n", nLoops);
    time1 = get_seconds();
#pragma omp parallel for
    for (long Qi=0; Qi<QTail; Qi++) {
      long v = Q[Qi]; //Q.pop_front();
      long StartIndex = v*MaxDegree; //Location in Mark
      if (nLoops > 0) //Skip the first time around
	for (long i=0; i<MaxDegree; i++)
	  Mark[StartIndex+i]= -1;
      long adj1 = verPtr[v];
      long adj2 = verPtr[v+1];      
      long maxColor = -1;
      long adjColor = -1;
      //Browse the adjacency set of vertex v
      for(long k = adj1; k < adj2; k++ ) {
	//if ( v == verInd[k] ) //Skip self-loops
	//continue;
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
    } //End of outer for loop: for each vertex
    time1  = get_seconds() - time1;
    totalTime += time1;
    printf("Time taken for Coloring:  %lf sec.\n", time1);
    
    ///////////////////////////////////////// PART 2 ////////////////////////////////////////
    //Detect Conflicts:
    //printf("Phase 2: Detect Conflicts, add to queue\n");    
    //Add the conflicting vertices into a Q:
    //Conflicts are resolved by changing the color of only one of the 
    //two conflicting vertices, based on their random values 
    time2 = get_seconds();
#pragma omp parallel for
    for (long Qi=0; Qi<QTail; Qi++) {
      long v = Q[Qi]; //Q.pop_front();
      long adj1 = verPtr[v];
      long adj2 = verPtr[v+1];      
      //Browse the adjacency set of vertex v
      for(long k = adj1; k < adj2; k++ ) {
	//if ( v == verInd[k] ) //Self-loops
	//continue;
	if ( vtxColor[v] == vtxColor[verInd[k]] ) {
	  //Q.push_back(v or w)
	  if ( (randValues[v] < randValues[verInd[k]]) || 
	       ((randValues[v] == randValues[verInd[k]])&&(v < verInd[k])) ) {
	    long whereInQ = __sync_fetch_and_add(&QtmpTail, 1);
	    Qtmp[whereInQ] = v;//Add to the queue
	    vtxColor[v] = -1;  //Will prevent v from being in conflict in another pairing
	    break;
	  }
	} //End of if( vtxColor[v] == vtxColor[verInd[k]] )
      } //End of inner for loop: w in adj(v)
    } //End of outer for loop: for each vertex
    time2  = get_seconds() - time2;
    totalTime += time2;    
    nConflicts += QtmpTail;
    nLoops++;
    printf("Conflicts          : %ld \n", QtmpTail);
    printf("Time for detection : %lf sec\n", time2);
    //Swap the two queues:
    Qswap = Q;
    Q = Qtmp; //Q now points to the second vector
    Qtmp = Qswap;
    QTail = QtmpTail; //Number of elements
    QtmpTail = 0; //Symbolic emptying of the second queue    
  } while (QTail > 0);
  //Check the number of colors used
  long nColors = -1;
  for (int v=0; v < NVer; v++ ) 
    if (vtxColor[v] > nColors) nColors = vtxColor[v];
  printf("***********************************************\n");
  printf("Total number of colors used: %ld \n", nColors);    
  printf("Number of conflicts overall: %ld \n", nConflicts);  
  printf("Number of rounds           : %ld \n", nLoops);      
  printf("Total Time                 : %lf sec\n", totalTime);
  printf("***********************************************\n");
  
  /////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////// VERIFY THE COLORS /////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  //Verify Results and Cleanup
  nConflicts = 0;
#pragma omp parallel for
  for (long v=0; v < NVer; v++ ) {
    long adj1 = verPtr[v];
    long adj2 = verPtr[v+1];
    //Browse the adjacency set of vertex v
    for(long k = adj1; k < adj2; k++ ) {
      if ( v == verInd[k] ) //Self-loops
	continue;
      if ( vtxColor[v] == vtxColor[verInd[k]] ) {
#pragma omp atomic
	nConflicts++; //increment the counter
      }
    }//End of inner for loop: w in adj(v)
  }//End of outer for loop: for each vertex
  nConflicts = nConflicts / 2; //Have counted each conflict twice
  if (nConflicts > 0)
    printf("Check - WARNING: Number of conflicts detected after resolution: %ld \n\n", nConflicts);
  else
    printf("Check - SUCCESS: No conflicts exist\n\n");
  //Clean Up:
  free(Q);
  free(Qtmp);
  free(Mark); 
  free(randValues);
}


