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

#include "defs.h"
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////  DOMINATING EDGE ALGORITHM  ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
void algoEdgeApproxDominatingEdgesLinearSearchNew( graph *G, long *Mate)
{
   int nthreads;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  printf("Approx Bipartite Matching: Number of threads: %d \n", nthreads);
  double time1=0, time2=0, totalTime=0;
  //Get the iterators for the graph:
  long NVer     = G->numVertices;
  long NS       = G->sVertices;
  long NT       = NVer - NS;
  long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
  long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
  printf("NS= %ld  NT=%ld  NE=%ld\n", NS, NT, NEdge);

  //Allocate Data Structures:
  long *HeaviestPointer = (long *) malloc (NVer * sizeof(long));
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
    Q[i]    = -1; 
    Qtmp[i] = -1;
  }
  long QTail   =0; //Tail of the queue (implicitly will represent the size)
  long QtmpTail=0; //Tail of the queue (implicitly will represent the size)
  /////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////// PART 1 ////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  //Compute the Initial Matching Set:
  time1 = timer();
#pragma omp parallel for 
  for (long v=0; v < NVer; v++ ) {
    //Start: COMPUTE_CANDIDATE_MATE(v)
    long adj1 = verPtr[v];
    long adj2 = verPtr[v+1];
    long w = -1;
    double heaviestEdgeWt = MilanRealMin; //Assign the smallest Value possible first LDBL
    for(long k = adj1; k < adj2; k++ ) {
      if ( Mate[verInd[k].tail] == -1 ) { //Process only if unmatched
	if( (verInd[k].weight > heaviestEdgeWt) ||
	    ((verInd[k].weight == heaviestEdgeWt)&&(w<verInd[k].tail)) ) {
	  heaviestEdgeWt = verInd[k].weight;
	  w = verInd[k].tail;
	}
      }//End of if (Mate == -1)
    } //End of for loop
    HeaviestPointer[v] = w; // c(v) <- Hsv(v)
  } //End of for loop for setting the pointers
  //Check if two vertices point to each other:
#pragma omp parallel for 
  for (long v=0; v < NVer; v++ ) {
    //If found a dominating edge:
    if ( HeaviestPointer[v] >= 0 )
      if ( HeaviestPointer[HeaviestPointer[v]] == v ) {
	Mate[v] = HeaviestPointer[v];
	//Q.push_back(u,w);
	long whereInQ = __sync_fetch_and_add(&QTail, 1);
	Q[whereInQ] = v;
      }//End of if(Pointer(Pointer(v))==v)
  }//End of for(int v=0; v < NVer; v++ )
  time1  = timer() - time1;
  //The size of Q1 is now QTail+1; the elements are contained in Q1[0] through Q1[Q1Tail]
  /////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////// PART 2 ////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  int nLoops=0; //Count number of iterations in the while loop
  while ( /*!Q.empty()*/ QTail > 0 )  {      
    printf("Loop %d, QSize= %ld\n", nLoops, QTail);
    //KEY IDEA: Process all the members of the queue concurrently:
    time2 = timer();
#pragma omp parallel for
    for (long Qi=0; Qi<QTail; Qi++) {
      //Q.pop_front();
      long v = Q[Qi];		
      long adj1 = verPtr[v]; 
      long adj2 = verPtr[v+1];
      for(long k = adj1; k < adj2; k++) {
	long x = verInd[k].tail;
	if ( Mate[x] != -1 )   // x in Sv \ {c(v)}
	  continue;
	if ( HeaviestPointer[x] == v ) {
	  //Start: PROCESS_EXPOSED_VERTEX(x)
	  //Start: COMPUTE_CANDIDATE_MATE(x)
	  long adj11 = verPtr[x];
	  long adj12 = verPtr[x+1];
	  long w = -1;
	  double heaviestEdgeWt = MilanRealMin; //Assign the smallest Value possible first LDBL_MIN
	  for(long k1 = adj11; k1 < adj12; k1++ ) {
	    if( Mate[verInd[k1].tail] != -1 ) // Sx <- Sx \ {v}
	      continue;
	    if( (verInd[k1].weight > heaviestEdgeWt) || 
		((verInd[k1].weight == heaviestEdgeWt)&&(w < verInd[k1].tail)) ) {
	      heaviestEdgeWt = verInd[k1].weight;
	      w = verInd[k1].tail;
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
    time2  = timer() - time2;
    totalTime += time2;
  } //end of while ( !Q.empty() )
  //printf("Number of iterations in while loop: %d\n",NumLoops);
  printf("***********************************************\n");
  printf("Number of iterations       : %d     \n", nLoops);
  printf("Time for Phase-1           : %lf sec\n", time1);
  printf("Time for Phase-2           : %lf sec\n", totalTime);
  printf("Total Time                 : %lf sec\n", totalTime+time1);
  printf("***********************************************\n");
  //Clean Up:
  free(Q);
  free(Qtmp);
  free(HeaviestPointer);
  
} //End of algoEdgeApproxDominatingEdgesLinearSearch


/* 
This Function Finds the Initial Extreme Matching as suggested in 
1. Algorithm 548 - Solution of the Assignment Problem [H] by 
Giorgio Carpaneto and Paolo Toth, 
ACM Transactions on Mathematical Software, Vol 6, No. 1 March 1980, Pages 104-111

2. On Algorithms for permuting large entries to the diagonal of a sparse matrix, by
I. S. Duff and J. Koster,
SIAM J. MATRIX ANAL. APPL. Vol 22, No. 4, pp. 973-996

The Columns are numbered 0-(Nc-1) and Rows: Nc-(Nr+Nc-1)

The output is loaded in the Mate and Dual vectors
*/

#define  PRINT_STATISTICS_
void algoEdgeApproxInitialExtremeMatchingBipartiteSerial( graph *G, long *Mate )
{
  printf("Within algoEdgeApproxInitialExtremeMatchingBipartiteSerial()\n");
  double time1=0, time2=0, totalTime=0;
  //Get the iterators for the graph:
  long NVer     = G->numVertices;
  long NS       = G->sVertices;
  long NT       = NVer - NS;
  long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
  long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
  printf("NS= %ld  NT=%ld  NE=%ld\n", NS, NT, NEdge);
  
  //Vector DUAL: the first NC elements hold the Column side duals
  //the the next NR elements hold the Row side duals
  long i=0,k=0,k1=0,u=0,u1=0, w=0,w1=0; //Temporary Variables
  long adj1=0, adj2=0, adj11=0, adj12=0, cardinality=0;
  double RowMaximum=0.0f;
  int pathFound = 0;

  //Step-1: Store local maximum for each S vertex  
  double *RMax = (double *) malloc (NS * sizeof(double));
  if( RMax == NULL ) {
    printf("Not enough memory to allocate for RMax \n");
    exit(1);
  }  
  for( i=0; i<NS; i++ ) {
    adj1 = verPtr[i];
    adj2 = verPtr[i+1];
    RowMaximum = MilanRealMin;
    for( k = adj1; k < adj2; k++ )
      if( RowMaximum < verInd[k].weight )
	RowMaximum = verInd[k].weight;
    RMax[i] = RowMaximum;
  }  

  //Compute matchings from edges that are locally maximum (tight)
  
  //Step-2: Find augmenting paths of length one
  time1 = timer();
  for( i=0; i<NS; i++ ) {
    adj1 = verPtr[i];
    adj2 = verPtr[i+1];
    //Scan the neighborhood for an eligible edge
    for( k = adj1; k < adj2; k++ ) {
      w = verInd[k].tail;
      if( verInd[k].weight == RMax[i] ) { //Tight edge?
	if( Mate[w] == -1)  { //Not matched before
#ifdef PRINT_STATISTICS_
	  cardinality++;
#endif
	  Mate[i] = w;         //Set the Mate array
	  Mate[w] = i;
	  break;
	}
      } //End of if
    } //End of for inner loop
  } //End of for outer loop
  time1  = timer() - time1;
  
#ifdef PRINT_STATISTICS_
  printf("\n Results after Stage 1: \n");
  printf("Cardinality: %ld\n", cardinality);
#endif
  
  //STEP 3: Find Short augmenting paths from unmatched nodes
  time2 = timer();
  for( u=0; u<NS; u++ ) {
    if( Mate[u] >= 0 ) //Ignore matched S vertices
      continue;
    
    pathFound = 0;
    adj1 = verPtr[u];
    adj2 = verPtr[u+1];
    //Process all the neighbors
    for(k = adj1; k < adj2; k++) {
      if ( pathFound == 1 )
	break;
      w = verInd[k].tail;
      //Consider neighbors: tight AND matched
      if( (verInd[k].weight == RMax[u])&&(Mate[w] >= 0) ) { 
	u1 = Mate[w]; //Get the other end of matched edge	
	//Check for an unmatched row node
	adj11 = verPtr[u1]; 
	adj12 = verPtr[u1+1];
	for(k1 = adj11; k1 < adj12; k1++) {
	  w1 = verInd[k1].tail;
	  //Look if the row node is matched: Tight AND unmatched edge
	  if( (verInd[k1].weight == RMax[u1])&&(Mate[w1] == -1) ) {
	    //AUGMENT:
	    Mate[u] = w;
	    Mate[w] = u;
	    
	    Mate[u1] = w1;
	    Mate[w1] = u1;
	    
#ifdef PRINT_STATISTICS_
	    cardinality++;	
#endif	    
	    pathFound = 1;
	    break;
	  }//End of if()
	}//End of inner for loop
      }//End of if
    } //End of outer for loop 
  } // End of outer for loop
  time2  = timer() - time2;
#ifdef PRINT_STATISTICS_
  printf("\n Results after Stage 2: \n");
  printf("Cardinality: %ld\n", cardinality);
#endif
  printf("***********************************************\n");
  printf("Time for Phase-1           : %lf sec\n", time1);
  printf("Time for Phase-2           : %lf sec\n", time2);
  printf("Total Time                 : %lf sec\n", time1+time2);
  printf("***********************************************\n");

  //Cleanip
  free(RMax);
} // End of InitialExtremeMatching()


void algoEdgeApproxInitialExtremeMatchingBipartiteParallel( graph *G, long *Mate )
{
  int nthreads;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  printf("Within algoEdgeApproxInitialExtremeMatchingBipartiteSerial() -- %d threads\n", nthreads);
  double time1=0, time2=0, totalTime=0;
  //Get the iterators for the graph:
  long NVer     = G->numVertices;
  long NS       = G->sVertices;
  long NT       = NVer - NS;
  long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
  long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
  printf("NS= %ld  NT=%ld  NE=%ld\n", NS, NT, NEdge);
  
  //Vector DUAL: the first NC elements hold the Column side duals
  //the the next NR elements hold the Row side duals
  long cardinality=0;
  
  //Step-1: Store local maximum for each S vertex  
  double *RMax      = (double *) malloc (NS * sizeof(double));
  if( RMax == NULL ) {
    printf("Not enough memory to allocate for RMax \n");
    exit(1);
  }
#pragma omp parallel for
  for(long i=0; i<NS; i++ ) {
    long adj1 = verPtr[i];
    long adj2 = verPtr[i+1];
    double RowMaximum = MilanRealMin;
    for(long k = adj1; k < adj2; k++ )
      if( RowMaximum < verInd[k].weight )
	RowMaximum = verInd[k].weight;
    RMax[i] = RowMaximum;
  }
  long *Visited = (long *)   malloc (NT * sizeof(long));
  if( Visited == NULL ) {
    printf("Not enough memory to allocate for Processed \n");
    exit(1);
  }
#pragma omp parallel for
  for(long i=0; i<NT; i++ ) {
    Visited[i] = 0;
  }

  //Compute matchings from edges that are locally maximum (tight)
  
  //Step-2: Find augmenting paths of length one
  time1  = timer();
#pragma omp parallel for
  for(long i=0; i<NS; i++ ) {
    long adj1 = verPtr[i];
    long adj2 = verPtr[i+1];
    //Scan the neighborhood for an eligible edge
    for(long k = adj1; k < adj2; k++ ) {
      long w = verInd[k].tail;
      //If the neighbor is tight and unmatched
      if( (verInd[k].weight == RMax[i])&&(Mate[w] == -1) ) {
	//Ignore if processed by another vertex
	if ( __sync_fetch_and_add(&Visited[w-NS], 1) == 0 ) {
#ifdef PRINT_STATISTICS_
	  __sync_fetch_and_add(&cardinality, 1);
#endif
	  Mate[i] = w;         //Set the Mate array
	  Mate[w] = i;
	  break;
	} //End of if(visited)
      } //End of if(Tight and unmatched)
    } //End of for inner loop
  } //End of for outer loop
  time1  = timer() - time1;
#ifdef PRINT_STATISTICS_
  printf("\n Results after Stage 1: \n");
  printf("Cardinality: %ld\n", cardinality);
#endif

  //Reuse Visited array
#pragma omp parallel for
  for(long i=0; i<NT; i++ ) {
    Visited[i] = 0;
  }
    
  //STEP 3: Find Short augmenting paths from unmatched nodes
  time2  = timer();
#pragma omp parallel for
  for(long u=0; u<NS; u++ ) {
    if( Mate[u] >= 0 ) //Ignore matched vertices
      continue;
    
    //If u is unmatched, find an augmenting path of length 3
    int pathFound = 0;
    long adj1 = verPtr[u];
    long adj2 = verPtr[u+1];
    //Process all the neighbors
    for(long k = adj1; k < adj2; k++) {
      if ( pathFound == 1 )
	break;
      long w = verInd[k].tail;
      //Consider neighbors: tight AND matched
      if( (verInd[k].weight == RMax[u])&&(Mate[w] >= 0) ) {
	long u1 = Mate[w]; //Get the other end of matched edge
	//Check for an unmatched row node
	long adj11 = verPtr[u1]; 
	long adj12 = verPtr[u1+1];
	for(long k1 = adj11; k1 < adj12; k1++) {
	  long w1 = verInd[k1].tail;
	  //Look if the row node is matched: Tight AND unmatched edge
	  if( (verInd[k1].weight == RMax[u1])&&(Mate[w1] == -1) ) {
	    //!!!! WARNING: The logic is not validated yet
	    //Ignore if processed by another vertex
	    if ( __sync_fetch_and_add(&Visited[w1-NS], 1) == 0 ) {
	      //AUGMENT:
	      Mate[u] = w;
	      Mate[w] = u;
	      
	      Mate[u1] = w1;
	      Mate[w1] = u1;
	      
#ifdef PRINT_STATISTICS_
	      __sync_fetch_and_add(&cardinality, 1);	
#endif	    
	      pathFound = 1;
	      break;
	    }//End of if(Visited)
	  }//End of if()
	}//End of inner for loop
      }//End of if
    } //End of outer for loop 
  } // End of outer for loop
  time2  = timer() - time2;
#ifdef PRINT_STATISTICS_
  printf("\n Results after Stage 2: \n");
  printf("Cardinality: %ld\n", cardinality);
#endif
  
  printf("***********************************************\n");
  printf("Time for Phase-1           : %lf sec\n", time1);
  printf("Time for Phase-2           : %lf sec\n", time2);
  printf("Total Time                 : %lf sec\n", time1+time2);
  printf("***********************************************\n");

  //Cleanip
  free(RMax);
  free(Visited);

} // End of algoEdgeApproxInitialExtremeMatchingBipartiteParallel()


void algoEdgeApproxInitialExtremeMatchingBipartiteParallel2( graph *G, long *Mate )
{
  int nthreads;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  printf("Within algoEdgeApproxInitialExtremeMatchingBipartiteParallel2() -- %d threads\n", nthreads);
  double time1=0, time2=0, totalTime=0;
  //Get the iterators for the graph:
  long NVer     = G->numVertices;
  long NS       = G->sVertices;
  long NT       = NVer - NS;
  long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
  long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
  printf("NS= %ld  NT=%ld  NE=%ld\n", NS, NT, NEdge);
  
  //Vector DUAL: the first NC elements hold the Column side duals
  //the the next NR elements hold the Row side duals
  long cardinality=0;
  
    long *Visited = (long *)   malloc (NT * sizeof(long));
  if( Visited == NULL ) {
    printf("Not enough memory to allocate for Processed \n");
    exit(1);
  }
#pragma omp parallel for
  for(long i=0; i<NT; i++ ) {
    Visited[i] = 0;
  }

  //Step-1: Store local maximum for each S vertex  
  time1  = timer();
  double *RMax      = (double *) malloc (NS * sizeof(double));
  if( RMax == NULL ) {
    printf("Not enough memory to allocate for RMax \n");
    exit(1);
  }
#pragma omp parallel for
  for(long i=0; i<NS; i++ ) {
    long adj1 = verPtr[i];
    long adj2 = verPtr[i+1];
    double RowMaximum = MilanRealMin;
    for(long k = adj1; k < adj2; k++ )
      if( RowMaximum < verInd[k].weight )
	RowMaximum = verInd[k].weight;
    RMax[i] = RowMaximum;
  }

  //Compute matchings from edges that are locally maximum (tight)  
  //Step-2: Find augmenting paths of length one  
#pragma omp parallel for
  for(long i=0; i<NS; i++ ) {
    long adj1 = verPtr[i];
    long adj2 = verPtr[i+1];
    //Scan the neighborhood for an eligible edge
    for(long k = adj1; k < adj2; k++ ) {
      long w = verInd[k].tail;
      //If the neighbor is tight and unmatched
      if( (verInd[k].weight == RMax[i])&&(Mate[w] == -1) ) {
	//Ignore if processed by another vertex
	if ( __sync_fetch_and_add(&Visited[w-NS], 1) == 0 ) {
#ifdef PRINT_STATISTICS_
	  __sync_fetch_and_add(&cardinality, 1);
#endif
	  Mate[i] = w;         //Set the Mate array
	  Mate[w] = i;
	  break;
	} //End of if(visited)
      } //End of if(Tight and unmatched)
    } //End of for inner loop
  } //End of for outer loop
  time1  = timer() - time1;
#ifdef PRINT_STATISTICS_
  printf("Results after Stage 1: \n");
  printf("Cardinality: %ld\n", cardinality);
#endif

  printf("***********************************************\n");
  printf("Time for Phase-1           : %lf sec\n", time1);
  printf("***********************************************\n");

  //Cleanip
  free(RMax);
  free(Visited);
} // End of InitialExtremeMatching()

void algoEdgeApproxInitialExtremeMatchingBipartiteParallel3( graph *G, long *Mate )
{
  int nthreads;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  printf("Within algoEdgeApproxInitialExtremeMatchingBipartiteParallel2() -- %d threads\n", nthreads);
  double time1=0, time2=0, totalTime=0;
  //Get the iterators for the graph:
  long NVer     = G->numVertices;
  long NS       = G->sVertices;
  long NT       = NVer - NS;
  long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
  long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
  printf("NS= %ld  NT=%ld  NE=%ld\n", NS, NT, NEdge);
  
  //Vector DUAL: the first NC elements hold the Column side duals
  //the the next NR elements hold the Row side duals
  long cardinality=0;
  
    long *Visited = (long *)   malloc (NT * sizeof(long));
  if( Visited == NULL ) {
    printf("Not enough memory to allocate for Processed \n");
    exit(1);
  }
#pragma omp parallel for
  for(long i=0; i<NT; i++ ) {
    Visited[i] = 0;
  }

  //Step-1: Store local maximum for each S vertex  
  time1  = timer();
  double *RMax      = (double *) malloc (NVer * sizeof(double));
  if( RMax == NULL ) {
    printf("Not enough memory to allocate for RMax \n");
    exit(1);
  }
#pragma omp parallel for
  for(long i=0; i<NVer; i++ ) {
    long adj1 = verPtr[i];
    long adj2 = verPtr[i+1];
    double RowMaximum = MilanRealMin;
    for(long k = adj1; k < adj2; k++ )
      if( RowMaximum < verInd[k].weight )
	RowMaximum = verInd[k].weight;
    RMax[i] = RowMaximum;
  }

  //Compute matchings from edges that are locally maximum (tight)  
  //Step-2: Find augmenting paths of length one  
#pragma omp parallel for
  for(long i=0; i<NS; i++ ) {
    long adj1 = verPtr[i];
    long adj2 = verPtr[i+1];
    //Scan the neighborhood for an eligible edge
    for(long k = adj1; k < adj2; k++ ) {
      long w = verInd[k].tail;
      //If the neighbor is tight and unmatched
      if( (verInd[k].weight == RMax[i])&&(verInd[k].weight == RMax[w])&&(Mate[w] == -1) ) {
	//Ignore if processed by another vertex
	if ( __sync_fetch_and_add(&Visited[w-NS], 1) == 0 ) {
#ifdef PRINT_STATISTICS_
	  __sync_fetch_and_add(&cardinality, 1);
#endif
	  Mate[i] = w;         //Set the Mate array
	  Mate[w] = i;
	  break;
	} //End of if(visited)
      } //End of if(Tight and unmatched)
    } //End of for inner loop
  } //End of for outer loop
  time1  = timer() - time1;
#ifdef PRINT_STATISTICS_
  printf("Results after Stage 1: \n");
  printf("Cardinality: %ld\n", cardinality);
#endif

  printf("***********************************************\n");
  printf("Time for Phase-1           : %lf sec\n", time1);
  printf("***********************************************\n");

  //Cleanip
  free(RMax);
  free(Visited);
} // End of InitialExtremeMatching()

