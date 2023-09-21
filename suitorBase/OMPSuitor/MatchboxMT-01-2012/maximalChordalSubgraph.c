/*---------------------------------------------------------------------------*/
/*                                                                           */
/*                          Mahantesh Halappanavar                           */
/*                        High Performance Computing                         */
/*                Pacific Northwest National Lab, Richland, WA               */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*                                                                           */
/* Copyright (C) 2011 Mahantesh Halappanavar                                 */
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
////////////////////////// MAXIMAL CHORDAL SUBGRAPH    ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
// isChordal: a vector of size 2*|E| that says if an edge is chordal or not 
//            indicated from (u-->v) only
void algoMaximalChordalSubGraph( graph_t *G, int *isChordal )
{
  printf("Within algoMaximalChordalSubGraph()\n");    
#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int nprocs   = omp_get_num_procs();
    int tid = omp_get_thread_num();
    if (tid == 0)
      printf("Number of threads: %d\n", nthreads );
      //printf("Number of threads: %d\n Number of procs: %d\n", nthreads, nprocs);      
  }
    double time1=0, time2=0, totalTime=0;
    long NVer        = G->n;
    long NEdge       = G->m;
    attr_id_t *verPtr     = G->numEdges; //Vertex Pointer: pointers to endV
    attr_id_t *verInd    = G->endV;     //Vertex Index: destination id of an edge (src -> dest)
    printf("Vertices:%ld  Edges:%ld\n", NVer,NEdge);
    
    //Queue for the storing the vertices in conflict
    int *Q    = (int *) malloc (NVer * sizeof(int));
    int *Qtmp = (int *) malloc (NVer * sizeof(int));
    int *Qswap;    
    if( (Q == NULL) || (Qtmp == NULL) ) {
	printf("Not enough memory to allocate for the two queues.\n");
	exit(1);
    }
    int QTail=0;    //Tail of the queue (implicitly will represent the size)
    int QtmpTail=0; //Tail of the queue (implicitly will represent the size)
    
#pragma omp parallel for
    for (int i=0; i<NVer; i++)
	Qtmp[i]= -1;
    
    int *visited     = (int *) malloc (NVer * sizeof(int));   //Check if needs to be added to the queue
    int *leastParent = (int *) malloc (NVer * sizeof(int));   //Holds the least parent of v
    if( (visited == NULL) || (leastParent == NULL) ) {
	printf("Not enough memory to allocate for datastructures.\n");
	exit(1);
    }  
#pragma omp parallel for
    for (int i=0; i<NVer; i++)
	visited[i]= 0; //Visited is false

    int numChordalEdges = 0; //Number of chordal edges    
    ///////////////////////////////////////// PART 1 ////////////////////////////////////////
    // Find the minimum parent for each vertex
    time1 = get_seconds();
#pragma omp parallel for
    for (int i=0; i<NVer; i++) {	    	    
	int adj1 = verPtr[i];
	int adj2 = verPtr[i+1];	    
	int minParent = i; //Largest identity
	for(int j = adj1; j < adj2; j++ ) {
	    if (verInd[j] < minParent)
		minParent = verInd[j];	    
	}//End of inner for(j)
	if ( minParent < i ) {
	    leastParent[i] = minParent; //Set the smallest parent
	    //Add this parent to the queue
	    if ( __sync_fetch_and_add(&visited[minParent], 1) == 0 )
		Q[__sync_fetch_and_add(&QTail, 1)] = minParent; //Add if not already in the queue
	}
	else
	    leastParent[i] = -1; //This vertex has no eligible parents
    }//End of outer for(i)
    time1      = get_seconds() - time1;
    totalTime += time1;
    printf("Time to set smallest parent for each vertex:  %9.6lf sec.\n", time1);

    /*
    printf("Q:\n");    
    for(int i=0; i<QTail; i++)
	printf("%d ",Q[i]+1);
    printf("\n");
    */
    ///////////////////////////////////////// PART 2 ////////////////////////////////////////
    time1 = get_seconds();
    int *countSubset = (int *) malloc (NVer * sizeof(int));   //Check if needs to be added to the queue
    int *chordalSet  = (int *) malloc (2*NEdge * sizeof(int));   //Holds the least parent of v
    if( (countSubset == NULL) || (chordalSet == NULL) ) {
	printf("Not enough memory to allocate for datastructures.\n");
	exit(1);
    }
#pragma omp parallel for
    for (int i=0; i<NVer; i++) {
	countSubset[i]= 0; //Could how many neighbors have been visited for each vertex (C(v))
    }
#pragma omp parallel for
    for (int i=0; i<2*NEdge; i++) {
	chordalSet[i]= 0; //Maintain chordal set C(v)-- Each edge is stored twice (u--> and v-->u)
    }
    time1      = get_seconds() - time1;
    totalTime += time1;
    printf("Time to set up memory for chordal sets :  %9.6lf sec.\n", time1);

    int nLoops = 0;     //Number of rounds of conflict resolution        
    while (QTail > 0) {
	nLoops++;	
	printf("******************************************\n");
	printf("**Iteration: %d, |Q|= %d  \n", nLoops, QTail);   
	time1 = get_seconds();
	//Reset the visited vector to zero for next round of queuing
#pragma omp parallel for
	for (int i=0; i<QTail; i++)
	    visited[Q[i]] = 0;

#pragma omp parallel for
	for (int Qi=0; Qi<QTail; Qi++) {	    
	    int v = Q[Qi]; //Q.pop_front();
	    int adj1 = verPtr[v];
	    int adj2 = verPtr[v+1];	    	    	     	    
	    //Browse the adjacency set of vertex v
	    for(int k = adj1; k < adj2; k++ ) {	
		int u =  verInd[k];
		if ( leastParent[u] == v ) {
		    //printf("Parent= %d  -- Child= %d\n", v+1, u+1);		    
		    //Check if C(u) SUBSET C(v)
		    int isProperSubset = 0; //False
		    if ( countSubset[u] == 0 ) //Empty set for u: C(u) = NULL
			isProperSubset = 1;
		    else {
			isProperSubset = 1; //Assume C(u) is subset of C(v)
			int adj11 = verPtr[u];
			int adj12 = adj11 + countSubset[u];
			
			int adj21 = verPtr[v];
			int adj22 = adj21 + countSubset[v];
			for (int ci=adj11; ci<adj12; ci++) {
			    int Cu_i = chordalSet[ci];
			    int found = 0; //Check if C(u) SUBSET C(v)
			    for (int cj=adj21; cj<adj22; cj++) {
				if (chordalSet[cj] == Cu_i) {
				    //printf("Chordal Check %d == %d\n", chordalSet[cj]+1, Cu_i+1 );
 				    found = 1;
				    break;			    
				}
			    }//End of for(cj)
			    if (found == 0) {
				isProperSubset = 0; //C(u) is not a subset of C(v)
				break;
			    }	    
			}//End of for(ci)			
		    }    
		    if ( isProperSubset == 1 ) {
			//C(u) <-- C(u) U {v}
			chordalSet[verPtr[u] + __sync_fetch_and_add(&countSubset[u], 1)] = v;			
			//Mark edge(u,v) as chordal
			//NOTE: Chordal edges are marked from u --> v ONLY
			isChordal[k] = 1; //Mark the edge as chordal
			//printf("Chordal edge: (%d, %d)\n",v+1,u+1);			
			__sync_fetch_and_add(&numChordalEdges, 1);		    
			//Find the next smallest parent, w, of u:
			int adjm1 = verPtr[u];
			int adjm2 = verPtr[u+1];
			int minParent = u; //Largest identity
			for(int mj = adjm1; mj < adjm2; mj++ ) {
			    //Find a parent that is larger than the current parent
			    if ( (verInd[mj] < minParent)&&(verInd[mj] > v) )
				minParent = verInd[mj];
			}//End of inner for(j)
			//If w is not NULL, add it to Qtmp:
			if ( minParent < u ) {
			    //printf("New Parent of %d is %d\n", u+1, minParent+1);
			    leastParent[u] = minParent; //Set the smallest parent
			    //Add this parent to the queue
			    if ( __sync_fetch_and_add(&visited[minParent], 1) == 0 )
				Qtmp[__sync_fetch_and_add(&QtmpTail, 1)] = minParent; //Add if not already in the queue
			}
			else
			    leastParent[u] = -1; //This vertex has no eligible parents
		    }//End of if ( isProperSubset == 1 )
		}//End of if ( leastParent == v )
	    }//End of for(k)
	}//End of for (Qi)
	
	//Swap the two queues:
	Qswap = Q;
	Q = Qtmp;         //Q now points to the second vector
	Qtmp = Qswap;     //Swap the queues
	QTail = QtmpTail; //Number of elements
	QtmpTail = 0;     //Symbolic emptying of the second queue
	
	time1      = get_seconds() - time1;
	totalTime += time1;	
	
    }//End of while (QTail > 0)  
    printf("***********************************************\n");
    printf("Number of chordal edges    : %d\n", numChordalEdges);
    printf("Number of iterations       : %d \n", nLoops);
    printf("Total Time                 : %lf sec.\n", totalTime);
    printf("***********************************************\n");
    
    //Clean Up:
    free(Q);    
    free(Qtmp); 
    free(countSubset);
    free(chordalSet);  
    free(visited);
    free(leastParent);
    
} //End of algoMaximalChordalSubGraph()



//////////////////////////////////////////////////////////////////////////////////////
////////////////////////// MAXIMAL CHORDAL SUBGRAPH    ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
// isChordal: a vector of size 2*|E| that says if an edge is chordal or not 
//            indicated from (u-->v) only
/// OPTIMIZATIONS:
// 1. Neighbors are stored in a sorted order
// 2. Chordal set is stored in a sorted order (for finding intersection) 
void algoMaximalChordalSubGraphOpt( graph_t *G, int *isChordal )
{
  printf("Within algoMaximalChordalSubGraphOpt()\n");    
  printf("Warning: Assumes neighbors are stoerd in a sorted (ascending) order.\n");
#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int nprocs   = omp_get_num_procs();
    int tid = omp_get_thread_num();
    if (tid == 0)
      printf("Number of threads: %d\n", nthreads );
    //printf("Number of threads: %d\n Number of procs: %d\n", nthreads, nprocs);
  }

    double time1=0, time2=0, totalTime=0;
    long NVer        = G->n;
    long NEdge       = G->m;
    attr_id_t *verPtr     = G->numEdges; //Vertex Pointer: pointers to endV
    attr_id_t *verInd    = G->endV;     //Vertex Index: destination id of an edge (src -> dest)
    printf("Vertices:%ld  Edges:%ld\n", NVer,NEdge);
    
    //Queue for the storing the vertices in conflict
    int *Q    = (int *) malloc (NVer * sizeof(int));
    int *Qtmp = (int *) malloc (NVer * sizeof(int));
    int *Qswap;    
    if( (Q == NULL) || (Qtmp == NULL) ) {
	printf("Not enough memory to allocate for the two queues.\n");
	exit(1);
    }
    int QTail=0;    //Tail of the queue (implicitly will represent the size)
    int QtmpTail=0; //Tail of the queue (implicitly will represent the size)
    
#pragma omp parallel for
    for (int i=0; i<NVer; i++)
	Qtmp[i]= -1;
    
    int *visited     = (int *) malloc (NVer * sizeof(int));   //Check if it needs to be added to the queue
    int *leastParent = (int *) malloc (NVer * sizeof(int));   //Holds the least parent of v
    if( (visited == NULL) || (leastParent == NULL) ) {
	printf("Not enough memory to allocate for datastructures.\n");
	exit(1);
    }
#pragma omp parallel for
    for (int i=0; i<NVer; i++)
	visited[i]= 0; //Visited is false

    int numChordalEdges = 0; //Number of chordal edges
    ///////////////////////////////////////// PART 1 ////////////////////////////////////////
    // Find the minimum parent for each vertex
    // NOTE: Assumes neighbors are in sorted order.
    time1 = get_seconds();
#pragma omp parallel for
    for (int i=0; i<NVer; i++) {	    	    
	int adj1 = verPtr[i];
	int adj2 = verPtr[i+1];	    
	int minParent = i; //Largest identity
	for(int j = adj1; j < adj2; j++ ) {
	    if (verInd[j] < minParent) {
		minParent = verInd[j];
		break; //Neighbors are sorted, stop at the first eligible node
	    }
	    else {
		if (verInd[j] >= i) //No self loops
		    break; //Neighbors are sorted, no need in processing vertices larger than itself
	    }
	}//End of inner for(j)
	if ( minParent < i ) {
	    leastParent[i] = minParent; //Set the smallest parent
	    //Add this parent to the queue
	    if ( __sync_fetch_and_add(&visited[minParent], 1) == 0 )
		Q[__sync_fetch_and_add(&QTail, 1)] = minParent; //Add if not already in the queue
	}
	else
	    leastParent[i] = -1; //This vertex has no eligible parents
    }//End of outer for(i)
    time1      = get_seconds() - time1;
    totalTime += time1;
    printf("Time to set smallest parent for each vertex:  %9.6lf sec.\n", time1);

    /*
    printf("Q:\n");    
    for(int i=0; i<QTail; i++)
	printf("%d ",Q[i]+1);
    printf("\n");
    */
    ///////////////////////////////////////// PART 2 ////////////////////////////////////////
    time1 = get_seconds();
    int *countSubset = (int *) malloc (NVer * sizeof(int));    //Check if needs to be added to the queue
    int *chordalSet  = (int *) malloc (2*NEdge * sizeof(int)); //Maintain the chordal subset for each vertex
    if( (countSubset == NULL) || (chordalSet == NULL) ) {
	printf("Not enough memory to allocate for datastructures.\n");
	exit(1);
    }
#pragma omp parallel for
    for (int i=0; i<NVer; i++) {
	countSubset[i]= 0; //Could how many neighbors have been visited for each vertex (C(v))
    }
#pragma omp parallel for
    for (int i=0; i<2*NEdge; i++) {
	chordalSet[i]= 0; //Maintain chordal set C(v)-- Each edge is stored twice (u-->v and v-->u)
    }
    time1      = get_seconds() - time1;
    totalTime += time1;
    printf("Time to set up memory for chordal sets :  %9.6lf sec.\n", time1);

    int nLoops = 0;     //Number of rounds of conflict resolution    
    while (QTail > 0) {
	nLoops++;	
	printf("******************************************\n");
	printf("**Iteration: %d, |Q|= %d  \n", nLoops, QTail);   
	time1 = get_seconds();
	//Reset the visited vector to zero for next round of queuing
#pragma omp parallel for
	for (int i=0; i<QTail; i++)
	    visited[Q[i]] = 0;
	
#pragma omp parallel for
	for (int Qi=0; Qi<QTail; Qi++) {	    
	    int v = Q[Qi]; //Q.pop_front();
	    int adj1 = verPtr[v];
	    int adj2 = verPtr[v+1];	    	    	     	    
	    //Browse the adjacency set of vertex v
	    for(int k = adj1; k < adj2; k++ ) {	
		int u =  verInd[k];
		if ( leastParent[u] == v ) {
		    //printf("Parent= %d  -- Child= %d\n", v+1, u+1);		    
		    //Check if C(u) SUBSET C(v)
		    int isProperSubset = 0; //False
		    if ( countSubset[u] == 0 ) //Empty set for u: C(u) = NULL
			isProperSubset = 1;
		    else {
			isProperSubset = 1; //Assume C(u) is subset of C(v)
			int adj11 = verPtr[u];
			int adj12 = adj11 + countSubset[u];
			
			int adj21 = verPtr[v];
			int adj22 = adj21 + countSubset[v];
			for (int ci=adj11; ci<adj12; ci++) {
			    int Cu_i = chordalSet[ci];
			    int found = 0; //Check if C(u) SUBSET C(v)
			    //Assumes that Chordal sets are stored in a sorted order
			    for (int cj=adj21; cj<adj22; cj++) {
				if ( chordalSet[cj] < Cu_i )
				    adj21++; //Do not have to compare it again for other nodes in C(u)
				else {
				    if ( chordalSet[cj] > Cu_i ) {
					break; //C(u) is not a subset of C(v)
				    }
				    else { //chordalSet[cj] == Cu_i)
					//printf("Chordal Check %d == %d\n", chordalSet[cj]+1, Cu_i+1 );
					found = 1;
					adj21++; //Sored in ascending order, do not have to check for next element in C(u)
					break;
				    }//End of inner else
				}//End of outer else				
			    }//End of for(cj)
			    if (found == 0) {
				isProperSubset = 0; //C(u) is not a subset of C(v)
				break;
			    }
			}//End of for(ci)
		    }
		    if ( isProperSubset == 1 ) {
			//C(u) <-- C(u) U {v}
			chordalSet[verPtr[u] + __sync_fetch_and_add(&countSubset[u], 1)] = v;
			//Since the least parent is chosen at each step, the list will be in a sorted order
			//Mark edge(u,v) as chordal
			//NOTE: Chordal edges are marked from u --> v ONLY
			isChordal[k] = 1; //Mark the edge as chordal
			//printf("Chordal edge: (%d, %d)\n",v+1,u+1);
			__sync_fetch_and_add(&numChordalEdges, 1);    
			//Find the next smallest parent, w, of u:
			int adjm1 = verPtr[u];
			int adjm2 = verPtr[u+1];
			int minParent = u; //Largest identity
			for(int mj = adjm1; mj < adjm2; mj++ ) {
			    //Find a parent that is larger than the current parent			    
			    if ( (verInd[mj] > v)&&(verInd[mj] < minParent) ) {
				minParent = verInd[mj];
				break; //Neighbors are sorted, stop at the first eligible node				
			    }
			    else {
				if (verInd[mj] >= u) //No self loops
				    break; //Neighbors are sorted, no need to process vertices larger than itself
			    }
			}//End of inner for(j)
			//If w is not NULL, add it to Qtmp:
			if ( minParent < u ) {
			    //printf("New Parent of %d is %d\n", u+1, minParent+1);
			    leastParent[u] = minParent; //Set the smallest parent
			    //Add this parent to the queue
			    if ( __sync_fetch_and_add(&visited[minParent], 1) == 0 )
			      Qtmp[__sync_fetch_and_add(&QtmpTail, 1)] = minParent; //Add if not already in the queue
			}
			else
			    leastParent[u] = -1; //This vertex has no eligible parents
		    }//End of if ( isProperSubset == 1 )
		}//End of if ( leastParent == v )
	    }//End of for(k)
	}//End of for (Qi)
	
	//Swap the two queues:
	Qswap = Q;
	Q = Qtmp;         //Q now points to the second vector
	Qtmp = Qswap;     //Swap the queues
	QTail = QtmpTail; //Number of elements
	QtmpTail = 0;     //Symbolic emptying of the second queue
	
	time1      = get_seconds() - time1;
	totalTime += time1;	
	
    }//End of while (QTail > 0)  
    printf("***********************************************\n");
    printf("Number of chordal edges    : %d\n", numChordalEdges);
    printf("Number of iterations       : %d \n", nLoops);
    printf("Total Time                 : %lf sec.\n", totalTime);
    printf("***********************************************\n");
    
    //Clean Up:
    free(Q);    
    free(Qtmp); 
    free(countSubset);
    free(chordalSet);  
    free(visited);
    free(leastParent);
    
} //End of algoMaximalChordalSubGraph()

