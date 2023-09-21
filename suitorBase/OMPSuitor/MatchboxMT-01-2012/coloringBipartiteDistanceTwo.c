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

#include "defs.h"

//#define MaxDegree 1024
//////////////////////////////////////////////////////////////////////////////////////
////////////////////// BIPARTITE DISTANCE TWO COLORING      //////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
// CSC: Compressed Sparse Column
// CSR: Compressed Sparse Row
// vtxColor: Color the column vertices only
// MaxDegree: An estimate of the maximum degree/color
void algoDistanceTwoVertexColoring( matrix_CSC *X, matrix_CSR *Y, 
				    int *vtxColor, int *numColors, int MaxDegree )
{
#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int nprocs   = omp_get_num_procs();
    int tid = omp_get_thread_num();
    if (tid == 0)
      printf("Coloring Rouinte: Number of threads: %d\n Number of procs: %d\n", nthreads, nprocs);
  }
  double time1=0, time2=0, totalTime=0;
  //Get the iterators for the CSC graph:
  int NRows       = X->nRows;
  int NCols       = X->nCols;
  int NNnz        = X->nNNZ;
  int *rowPtr     = X->RowPtr;  //Row Pointer
  int *rowInd     = X->RowInd;  //Row Index 
  
  //Get the iterators for the CSR graph:
  int *colPtr     = Y->ColPtr;  //Row Pointer
  int *colInd     = Y->ColInd;  //Row Index    
  
  //Build a vector of random numbers
  time1 = timer();
  double *randValues = (double*) malloc (NCols * sizeof(double));
  if( randValues == NULL ) {
    printf("Not enough memory to allocate for random numbers \n");
    exit(1);
  }
  prand(NCols, randValues); //Generate n random numbers
  time1 = timer() - time1;
  printf("Time taken for random number generation:  %9.6lf sec.\n", time1);  
  
  //Queue for the storing the vertices in conflict
  int *Q    = (int *) malloc (NCols * sizeof(int));
  int *Qtmp = (int *) malloc (NCols * sizeof(int));
  int *Qswap;    
  if( (Q == NULL) || (Qtmp == NULL) ) {
    printf("Not enough memory to allocate for the two queues \n");
    exit(1);
  }
  int QTail=0;    //Tail of the queue (implicitly will represent the size)
  int QtmpTail=0; //Tail of the queue (implicitly will represent the size)
  
#pragma omp parallel for
  for (int i=0; i<NCols; i++) {
    Q[i]= i; //Natural order
    Qtmp[i]= -1;
  }
  QTail = NCols;       
  
  /////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////// START THE WHILE LOOP ///////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////    
  int nLoops = 0;     //Number of rounds of conflict resolution
  int nTotalConflicts = 0; //Total number of conflicts
  int *Mark = (int *) malloc ( MaxDegree * NCols * sizeof(int) );
  printf("Size of Mark: %d\n",MaxDegree * NCols);
  if( Mark == NULL ) {
    printf("Not enough memory to allocate for Mark \n");
    exit(1);
  }

  printf("Results from parallel coloring (using Bits):\n");
  printf("***********************************************\n");
  do {
    ///////////////////////////////////////// PART 1 ////////////////////////////////////////
    //Color the vertices in parallel - do not worry about conflicts
    //printf("Phase 1: Color vertices, ignore conflicts.\n");    
    printf("** Iteration : %d (|Q|=%d)\n", nLoops,QTail);
    time1 = timer();
#pragma omp parallel for
    for (int Qi=0; Qi<QTail; Qi++) {
      int v = Q[Qi]; //Q.pop_front();      
      int StartIndex = v*MaxDegree; //Location in Mark
      for (int i=0; i<MaxDegree; i++)
	Mark[StartIndex+i]= -1;      
      int adj1 = rowPtr[v];
      int adj2 = rowPtr[v+1];	    
      int maxColor = -1;
      int adjColor = -1;
      //Browse the adjacency set of vertex v
      for(int k = adj1; k < adj2; k++ ) {	
	int u = rowInd[k]; //The row index:
	int adj11 = colPtr[u];
	int adj12 = colPtr[u+1];
	for(int kk = adj11; kk < adj12; kk++ ) {	
	  if ( colInd[kk] == v ) //Self-loop
	    continue;
	  adjColor =  vtxColor[colInd[kk]];
	  if ( adjColor >= 0 ) {
	    Mark[StartIndex+adjColor] = v;
	    //Find the largest color in the neighborhood
	    if ( adjColor > maxColor )
	      maxColor = adjColor;
	  }
	}
      } //End of for loop to traverse adjacency of v
      int myColor;
      for (myColor=0; myColor<=maxColor; myColor++) {
	if ( Mark[StartIndex+myColor] != v )
	  break;
      }
      if (myColor == maxColor)
	myColor++; /* no available color with # less than cmax */
      vtxColor[v] = myColor; //Color the vertex
    } //End of outer for loop: for each vertex
    time1      = timer() - time1;
    totalTime += time1;
    //printf("Time taken for Coloring:  %9.6lf sec.\n", time1);
    
    ///////////////////////////////////////// PART 2 ////////////////////////////////////////
    //Detect Conflicts:
    time2 = timer();
#pragma omp parallel for
    for (int Qi=0; Qi<QTail; Qi++) {
      int v = Q[Qi]; //Q.pop_front();
      int adj1 = rowPtr[v];
      int adj2 = rowPtr[v+1];
      //Browse the adjacency set of vertex v
      for(int k = adj1; k < adj2; k++ ) {
	int u = rowInd[k]; //The row index:
	int adj11 = colPtr[u];
	int adj12 = colPtr[u+1];
	for(int kk = adj11; kk < adj12; kk++ ) {
	  int w = colInd[kk];  //The col index:
	  //printf("v= %d - w= %d (%d)\n", v, w, adj12-adj11);
	  if ( (w == v) || (vtxColor[v] == -1) ) //Link back/preprocess
	    continue;
	  if ( vtxColor[v] == vtxColor[w] ) {
	    //Q.push_back(v or w) 
	    if ( (randValues[v] < randValues[w]) || 
		 ((randValues[v] == randValues[w])&&(v < w)) ) {
	      int whereInQ = __sync_fetch_and_add(&QtmpTail, 1);
	      Qtmp[whereInQ] = v;//Add to the queue
	      vtxColor[v] = -1;  //Will prevent v from being in conflict in another pairing
	      break;
	    } //If rand values			
	  } //End of if( vtxColor[v] == vtxColor[verInd[k]] )
	}//End of inner for loop on kk		
      } //End of middle for loop on k
    } //End of outer for loop on Qi
    time2  = timer() - time2;
    totalTime += time2;
    nLoops++;
    nTotalConflicts += QtmpTail;
    //printf("Conflicts : %d \n", QtmpTail);
    //printf("Time for detection     : %9.6lf sec.\n", time2);
    
    //Swap the two queues:
    Qswap    = Q;
    Q        = Qtmp;     //Q now points to the second vector
    Qtmp     = Qswap;    //Swap the queues
    QTail    = QtmpTail; //Number of elements
    QtmpTail = 0;        //Symbolic emptying of the second queue
  } while (QTail > 0);
  
  int max_color = -1;
  for (int i = 0; i < NCols; i++)
    if (vtxColor[i] > max_color) max_color = vtxColor[i];
  *numColors = max_color+1; //Number of colors will be one greater than the max, since zero is a valid color
  printf("***********************************************\n");
  printf("Number of Columns          : %d\n",  NCols);
  printf("Number of Colors Used      : %d\n",  max_color + 1);
  printf("Number of conflicts overall: %d \n", nTotalConflicts);
  printf("Number of rounds           : %d \n", nLoops);
  printf("Total Time                 : %9.6lf sec.\n", totalTime);
  printf("Average columns per color  : %9.6lf\n", (double)NCols/(double)(max_color + 1));
  printf("***********************************************\n");
  /////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////// VERIFY THE COLORS /////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  nTotalConflicts = 0;
#pragma omp parallel for
  for (int v=0; v < NCols; v++ ) {
    int adj1 = rowPtr[v]; 
    int adj2 = rowPtr[v+1];
    //Browse the adjacency set of vertex v
    for(int k = adj1; k < adj2; k++ ) {
      int u = rowInd[k]; //The row index:
      int adj11 = colPtr[u];
      int adj12 = colPtr[u+1];
      for(int kk = adj11; kk < adj12; kk++ ) {
	int w = colInd[kk];	    
	if ( w == v ) //Link back
	  continue;		    
	if ( vtxColor[v] == vtxColor[w] ) {		    
	  __sync_fetch_and_add(&nTotalConflicts,1); //increment the counter
	}
      }//End of for loop on kk	    
    }//End of for loop on 
  }//End of for loop on v
  nTotalConflicts = nTotalConflicts / 2; //Have counted each conflict twice
  if (nTotalConflicts > 0)
    printf("Check - WARNING: Number of conflicts detected after resolution: %d \n",nTotalConflicts);
  else
    printf("Check - SUCCESS: No conflicts exist.\n");
  
  //Clean Up:
  free(Q);
  free(Qtmp);
  free(Mark); 
  free(randValues);
} //End of algoEdgeApproxDominatingEdgesLinearSearch
