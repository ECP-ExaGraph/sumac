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

#include "coloringAndMatchingKernels.h"

#include <math.h>
#define MilanRealMax HUGE_VAL // +INFINITY
#define MilanRealMin -HUGE_VAL // -INFINITY

//j = __sync_fetch_and_add(&k, 1);
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////  DOMINATING EDGE ALGORITHM  ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
// Perform breadth-first search from vector Source
// SSize indicates the size of Source
void algoBreadthFirstSearch( graph_t* G, long * Source, long SSize )
{
#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    if (tid == 0)
      printf("Number of threads: %d \n", nthreads);
  }
  
  double time1=0, time2=0, totalTime=0;
  //Get the iterators for the graph:
  long NVer         = G->n;
  long NEdge        = G->m;
  attr_id_t *verPtr = G->numEdges;   //Vertex Pointer: pointers to endV
  attr_id_t *verInd = G->endV;       //Vertex Index: destination id of an edge (src -> dest)
  int *edgeWeight   = G->int_weight_e;   //Edge Weight - Int
  //long *edgeWeight      = G->l_weight_e;   //Edge Weight - Long
  //double *edgeWeight    = G->dbl_weight_e;   //Edge Weight
  printf("Vertices: %ld  Edges: %ld\n", NVer, NEdge/2);
  
  //Allocate Data Structures:
  long *visited  = (long *) malloc (NVer * sizeof(long));
  if( visited == NULL ) {
    printf("Not enough memory to allocate the internal variable HeaviestPointer \n");
    exit(1);
  }
  //Initialize the Vectors:
#pragma omp parallel for
  for (long i=0; i<NVer; i++)
   visited[i]= 0; //zero means not visited
  
  //The Queue Data Structure for the Dominating Set:
  //The Queues are important for synchornizing the concurrency:
  //Have two queues - read from one, write into another
  // at the end, swap the two.
  long *Q    = (long *) malloc (NVer * sizeof(long));
  long *Qtmp = (long *) malloc (NVer * sizeof(long));
  long *Qswap;    
  if( (Q == NULL) || (Qtmp == NULL) ) {
    printf("Not enough memory to allocate for the two queues \n");
    exit(1);
  }
#pragma omp parallel for
  for (long i=0; i<NVer; i++) {
    Q[i]= -1; 
    Qtmp[i]= -1;
  }
  long QTail=0; //Tail of the queue (implicitly will represent the size)
  long QtmpTail=0; //Tail of the queue (implicitly will represent the size)
  /////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////// PART 1 ////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  //Build the initial Frontier Set
  if ( SSize > NVer ) {
    printf("The initial frontier set is larger than number of vertices\n");
    exit(1);
  }
  time1 = get_seconds();
#pragma omp parallel for 
  for (long v=0; v <SSize ; v++ ) {
    Q[v] = Source[v];
    visited[v]= 1;
    __sync_fetch_and_add(&QTail, 1); //Increment the queue tail
  }
  time1  = get_seconds() - time1;
  if ( SSize != QTail ) {
    printf("The initial frontier not correctly added to the queue\n");
    exit(1);
  }
  //The size of Q1 is now QTail+1; the elements are contained in Q1[0] through Q1[Q1Tail]
  /////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////// PART 2 ////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  int nLoops=0; //Count number of iterations in the while loop
  while ( /*!Q.empty()*/ QTail > 0 ) {      
    printf("Loop %d, QSize= %ld\n",nLoops,QTail);
    //KEY IDEA: Process all the members of the queue concurrently:
    time2 = get_seconds();
#pragma omp parallel for
    for (long Qi=0; Qi<QTail; Qi++) {
      long v = Q[Qi];		
      long adj1 = verPtr[v]; 
      long adj2 = verPtr[v+1];
      for(long k = adj1; k < adj2; k++) {
	long x = verInd[k];
	//Has this neighbor been visited?
	if ( __sync_fetch_and_add(&visited[x], 1) == 0 ) {
	  //Not visited: add it to the Queue
	  long whereInQ = __sync_fetch_and_add(&QtmpTail, 1);
	  Qtmp[whereInQ] = x;	  
	}
      } //End of for loop on k: the neighborhood of v
    } //End of processing Q
    // Also end of the parallel region
    // Swap the two queues:      
    Qswap = Q;
    Q = Qtmp; //Q now points to the second vector
    Qtmp = Qswap;
    QTail = QtmpTail; //Number of elements
    QtmpTail = 0; //Symbolic emptying of the second queue
    nLoops++;
    time2  = get_seconds() - time2;
    totalTime += time2;
  } //end of while ( !Q.empty() )
    //printf("Number of iterations in while loop: %d\n",NumLoops);
  printf("***********************************************\n");
  printf("Number of iterations       : %d     \n",    nLoops);
  printf("Time for Initialization    : %lf sec\n", time1);
  printf("Time for Graph-traversal   : %lf sec\n", totalTime);
  printf("Total Time                 : %lf sec\n", totalTime+time1);
  printf("***********************************************\n");
  //Clean Up:
  free(Q);
  free(Qtmp);
  free(visited);
  
} //End of algoEdgeApproxDominatingEdgesLinearSearch

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////  DOMINATING EDGE ALGORITHM  ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
// Perform breadth-first search from vector Source
// SSize indicates the size of Source
void algoBreadthFirstSearchSerial( graph_t* G, long * Source, long SSize )
{
#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    if (tid == 0)
      printf("Number of threads: %d \n", nthreads);
  }
  
  double time1=0, time2=0, totalTime=0;
  //Get the iterators for the graph:
  long NVer         = G->n;
  long NEdge        = G->m;
  attr_id_t *verPtr = G->numEdges;   //Vertex Pointer: pointers to endV
  attr_id_t *verInd = G->endV;       //Vertex Index: destination id of an edge (src -> dest)
  //int *edgeWeight   = G->int_weight_e;   //Edge Weight - Int
  //long *edgeWeight      = G->l_weight_e;   //Edge Weight - Long
  //double *edgeWeight    = G->dbl_weight_e;   //Edge Weight
  printf("Vertices: %ld  Edges: %ld\n", NVer, NEdge/2);
  
  //Allocate Data Structures:
  long *visited  = (long *) malloc (NVer * sizeof(long));
  if( visited == NULL ) {
    printf("Not enough memory to allocate the internal variable HeaviestPointer \n");
    exit(1);
  }
  //Initialize the Vectors:
#pragma omp parallel for
  for (long i=0; i<NVer; i++)
    visited[i]= 0; //zero means not visited
  
  //The Queue Data Structure for the Dominating Set:
  //The Queues are important for synchornizing the concurrency:
  //Have two queues - read from one, write into another
  // at the end, swap the two.
  long *Q    = (long *) malloc (NVer * sizeof(long));
  long *Qtmp = (long *) malloc (NVer * sizeof(long));
  long *Qswap;    
  if( (Q == NULL) || (Qtmp == NULL) ) {
    printf("Not enough memory to allocate for the two queues \n");
    exit(1);
  }
#pragma omp parallel for
  for (long i=0; i<NVer; i++) {
    Q[i]= -1; 
    Qtmp[i]= -1;
  }
  long QTail=0; //Tail of the queue (implicitly will represent the size)
  long QtmpTail=0; //Tail of the queue (implicitly will represent the size)
  /////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  //Build the initial Frontier Set
  if ( SSize > NVer ) {
    printf("The initial frontier set is larger than number of vertices\n");
    exit(1);
  }
  time1 = get_seconds();
#pragma omp parallel for 
  for (long v=0; v <SSize ; v++ ) {
    Q[v] = Source[v];
    visited[v]= 1;
    __sync_fetch_and_add(&QTail, 1); //Increment the queue tail
  }
  time1  = get_seconds() - time1;
  if ( SSize != QTail ) {
    printf("The initial frontier not correctly added to the queue\n");
    exit(1);
  }
  //The size of Q1 is now QTail+1; the elements are contained in Q1[0] through Q1[Q1Tail]
  /////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  int nLoops=0; //Count number of iterations in the while loop
  while ( /*!Q.empty()*/ QTail > 0 ) {      
    printf("Depth %d, QSize= %ld\n",nLoops,QTail);
    //KEY IDEA: Process all the members of the queue concurrently:
    time2 = get_seconds();
    for (long Qi=0; Qi<QTail; Qi++) {
      long v = Q[Qi];		
      long adj1 = verPtr[v]; 
      long adj2 = verPtr[v+1];
      for(long k = adj1; k < adj2; k++) {
	long x = verInd[k];
	//Has this neighbor been visited?
	if ( visited[x] == 0 ) {
	  //Not visited: add it to the Queue	  
	  Qtmp[QtmpTail] = x;
	  QtmpTail++;
	  visited[x] = 1;
	}
      } //End of for loop on k: the neighborhood of v
    } //End of processing Q
    // Also end of the parallel region
    // Swap the two queues:      
    Qswap = Q;
    Q = Qtmp; //Q now points to the second vector
    Qtmp = Qswap;
    QTail = QtmpTail; //Number of elements
    QtmpTail = 0; //Symbolic emptying of the second queue
    nLoops++;
    time2  = get_seconds() - time2;
    totalTime += time2;
  } //end of while ( !Q.empty() )
    //printf("Number of iterations in while loop: %d\n",NumLoops);
  printf("***********************************************\n");
  printf("Number of iterations       : %d     \n",    nLoops);
  printf("Time for Initialization    : %lf sec\n", time1);
  printf("Time for Graph-traversal   : %lf sec\n", totalTime);
  printf("Total Time                 : %lf sec\n", totalTime+time1);
  printf("***********************************************\n");
  //Clean Up:
  free(Q);
  free(Qtmp);
  free(visited);
  
} //End of algoEdgeApproxDominatingEdgesLinearSearch


//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////  FIRST TOUCH - GRAPH TRAVERSAL ////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
// Perform graph traversal for first-touch effects
void algoFirstTouch( graph_t* G )
{
#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    if (tid == 0)
      printf("Number of threads: %d \n", nthreads);
  }
  
  double timeT=0;
  long NVer         = G->n;
  long NEdge        = G->m;
  attr_id_t *verPtr = G->numEdges;   //Vertex Pointer: pointers to endV
  attr_id_t *verInd = G->endV;       //Vertex Index: destination id of an edge (src -> dest)
  int *edgeWeight   = G->int_weight_e;   //Edge Weight - Int
  //long *edgeWeight      = G->l_weight_e;   //Edge Weight - Long
  //double *edgeWeight    = G->dbl_weight_e;   //Edge Weight
  printf("Vertices: %ld  Edges: %ld\n", NVer, NEdge/2);
  
  //Allocate Data Structures:
  long *visited  = (long *) malloc (NVer * sizeof(long));
  if( visited == NULL ) {
    printf("Not enough memory to allocate the internal variable HeaviestPointer \n");
    exit(1);
  }
  
  //KEY IDEA: Process all the members of the queue concurrently:
  timeT = get_seconds();
#pragma omp parallel for
  for (long v=0; v<NVer; v++) {
    long adj1 = verPtr[v]; 
    long adj2 = verPtr[v+1];
    for(long k = adj1; k < adj2; k++) {
      long x = verInd[k];
      double wt = edgeWeight[k];
    } //End of for loop on k: the neighborhood of v
  } //End of processing Q
  timeT  = get_seconds() - timeT;
  //printf("Number of iterations in while loop: %d\n",NumLoops);
  printf("***********************************************\n");
  printf("Time for Graph-traversal   : %lf sec\n", timeT);
  printf("***********************************************\n");
  
} //End of algoEdgeApproxDominatingEdgesLinearSearch
