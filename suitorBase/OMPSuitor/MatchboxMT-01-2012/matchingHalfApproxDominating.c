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
void algoEdgeApproxDominatingEdgesLinearSearch( graph_t* G, long *Mate)
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
  long * HeaviestPointer = (long *) malloc (NVer * sizeof(long));
  if( HeaviestPointer == NULL ) {
    printf("Not enough memory to allocate the internal variable HeaviestPointer \n");
    exit(1);
  }
  //Initialize the Vectors:
#pragma omp parallel for
  for (long i=0; i<NVer; i++)
    HeaviestPointer[i]= -1;
  
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
  //Compute the Initial Matching Set:
  time1 = get_seconds();
  //#pragma omp parallel  default(none), shared(HeaviestPointer, Mate, QTail, Q, verPtr, edgeWeight, verInd)

  //#pragma omp for firstprivate(NVer)
#pragma omp parallel for 
  for (long v=0; v < NVer; v++ ) {
    //Start: COMPUTE_CANDIDATE_MATE(v)
    long adj1 = verPtr[v];
    long adj2 = verPtr[v+1];
    long w = -1;
    double heaviestEdgeWt = MilanRealMin; //Assign the smallest Value possible first LDBL_
    for(long k = adj1; k < adj2; k++ ) {
      if ( Mate[verInd[k]] == -1 ) { //Process only if unmatched
	if( (edgeWeight[k] > heaviestEdgeWt) || 
	    ((edgeWeight[k] == heaviestEdgeWt)&&(w<verInd[k])) ) {
	  heaviestEdgeWt = edgeWeight[k];
	  w = verInd[k];
	}
      }//End of if (Mate == -1)
    } //End of for loop
    HeaviestPointer[v] = w; // c(v) <- Hsv(v)
  } //End of for loop for setting the pointers
  //Check if two vertices point to each other:
  //#pragma omp for firstprivate(NVer)
#pragma omp parallel for 
  for (long v=0; v < NVer; v++ ) {
    //If found a dominating edge:
    if ( HeaviestPointer[v] >= 0 )
      if ( HeaviestPointer[HeaviestPointer[v]] == v ) {
	Mate[v] = HeaviestPointer[v];
	//Q.push_back(u,w);
	long whereInQ = __sync_fetch_and_add(&QTail, 1);
	//printf("WhereInQ= %d\t", whereInQ);
	//#pragma omp critical
	//{
	//  whereInQ = QTail++;
	//}
	Q[whereInQ] = v;
      }//End of if(Pointer(Pointer(v))==v)
  }//End of for(int v=0; v < NVer; v++ )

  time1  = get_seconds() - time1;
  //The size of Q1 is now QTail+1; the elements are contained in Q1[0] through Q1[Q1Tail]
  /////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////// PART 2 ////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  int nLoops=0; //Count number of iterations in the while loop
  while ( /*!Q.empty()*/ QTail > 0 )
    {      
      printf("Loop %d, QSize= %ld\n",nLoops,QTail);
      //KEY IDEA: Process all the members of the queue concurrently:
      time2 = get_seconds();
#pragma omp parallel for
      for (long Qi=0; Qi<QTail; Qi++) {
	//Q.pop_front();
	long v = Q[Qi];		
	long adj1 = verPtr[v]; 
	long adj2 = verPtr[v+1];
	for(long k = adj1; k < adj2; k++) {
	  long x = verInd[k];
	  if ( Mate[x] != -1 )   // x in Sv \ {c(v)}
	    continue;
	  if ( HeaviestPointer[x] == v ) {
	    //Start: PROCESS_EXPOSED_VERTEX(x)
	    //Start: COMPUTE_CANDIDATE_MATE(x)
	    long adj11 = verPtr[x];
	    long adj12 = verPtr[x+1];
	    long w = -1;
	    double heaviestEdgeWt = MilanRealMin; //Assign the smallest Value possible first LDBL_MIN
	    for(int k1 = adj11; k1 < adj12; k1++ ) {
	      if( Mate[verInd[k1]] != -1 ) // Sx <- Sx \ {v}
		continue;
	      if( (edgeWeight[k1] > heaviestEdgeWt) || 
		  ((edgeWeight[k1] == heaviestEdgeWt)&&(w<verInd[k1])) ) {
		heaviestEdgeWt = edgeWeight[k1];
		w = verInd[k1];
	      }
	    }//End of for loop on k1
	    HeaviestPointer[x] = w; // c(x) <- Hsv(x)
	    //End: COMPUTE_CANDIDATE_MATE(v)
	    //If found a dominating edge:
	    if ( HeaviestPointer[x] >= 0 ) 
	      if ( HeaviestPointer[HeaviestPointer[x]] == x ) {
		Mate[x] = HeaviestPointer[x];
		Mate[HeaviestPointer[x]] = x;
		//Q.push_back(u);
		long whereInQ = __sync_fetch_and_add(&QtmpTail, 2);
		//#pragma omp critical
		//{
		//  whereInQ = QtmpTail;
		//  QtmpTail += 2; //Add two elements to the queue
		//}
		Qtmp[whereInQ] = x;                    //add u
		Qtmp[whereInQ+1] = HeaviestPointer[x]; //add w
	      } //End of if found a dominating edge
	  } //End of if ( HeaviestPointer[x] == v )
	} //End of for loop on k: the neighborhood of v
      } //End of for loop on i: the number of vertices in the Queue
      ///Also end of the parallel region
      //Swap the two queues:
      
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
  printf("Time for Phase-1           : %lf sec\n", time1);
  printf("Time for Phase-2           : %lf sec\n", totalTime);
  printf("Total Time                 : %lf sec\n", totalTime+time1);
  printf("***********************************************\n");
  //Clean Up:
  free(Q);
  free(Qtmp);
  free(HeaviestPointer);

} //End of algoEdgeApproxDominatingEdgesLinearSearch


