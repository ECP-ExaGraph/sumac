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
void algoEdgeApproxDominatingEdgesSuitorNew( graph* G, long *Mate)
{
    int nthreads;
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    printf("algoEdgeApproxDominatingEdgesSuitorNew(): Number of threads: %d\n", nthreads);
    
    double time1=0, time2=0;
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
    omp_lock_t *nlocks = (omp_lock_t *) malloc (NVer * sizeof(omp_lock_t));
    long *s            = Mate;
    double *ws         = (double *) malloc (NVer * sizeof(double));
    if( (ws==NULL)||(nlocks==NULL) ) {
        printf("Not enough memory to allocate the internal variables  nlocks and ws\n");
        exit(1);
    }
    //Initialize the Vectors:
#pragma omp parallel for
    for (long i=0; i<NVer; i++) {
        ws[i]= 0.0f;               //Set current weight of best suitor to zero
        omp_init_lock(&nlocks[i]); //Initialize locks
    }
    time1  = get_seconds() - time1;
    
    time2 = get_seconds();
#pragma omp parallel for
    for(long x=0; x<NVer; x++) {  // Loop over vertices
        long i = x;
        int done = 0; //FALSE;
        while (!done) {
            double heaviest = ws[i];
            long partner     =  s[i];
            long next_vertex;
            //printf("Processing node %d \n",i);
            long adj1 = verPtr[i];
            long adj2 = verPtr[i+1];
            for(long j=adj1; j<adj2; j++) { // Loop over neighbors of vertex i
                long y = verInd[j].tail;    // y is the current neighbor of i
                double edgeWeight = verInd[j].weight; //weight of y
                if( (edgeWeight < heaviest)||(edgeWeight < ws[y]) )
                    continue;
                if( (edgeWeight == heaviest)&&(y < partner) )
                    continue;
                if( (edgeWeight == ws[y])&&(i < s[y]) )
                    continue;
                // Check if w(i,y) is the best so far, and if it is a better option for y
                heaviest = edgeWeight;      // Store the best so far
                partner = y;
            } // loop over neighbors
            done = 1; //TRUE;
            if ( heaviest > 0 ) {
                omp_set_lock(&nlocks[partner]);    // Locking partner
                if( (heaviest > ws[partner])||( (heaviest == ws[partner])&&(i>s[partner]) ) ) {
                    if (s[partner] >= 0 ) {
                        next_vertex = s[partner];
                        done = 0; //FALSE;
                    }
                    s[partner]  = i;         // i is now the current suitor of s
                    ws[partner] = heaviest;  // The weight of the edge (i,partner)
                }
                else {   // can no longer use the result for node i, must find new partner
                    done = 0; // FALSE;
                    next_vertex = i;
                }
                omp_unset_lock(&nlocks[partner]); // Unlocking partner
            }
            if( !done ) { // Continue with the next vertex
                i = next_vertex;
            }
        } // while not done
    } // loop over vertices
    time2  = get_seconds() - time2;
    
    printf("***********************************************\n");
    printf("Time for Initialization    : %lf sec\n", time1);
    printf("Time for Matching          : %lf sec\n", time2);
    printf("Total Time                 : %lf sec\n", time1+time2);
    printf("***********************************************\n");
    
    //Clean Up:
    free(ws);
#pragma omp parallel for
    for (long i=0; i<NVer; i++) {
        omp_destroy_lock(&nlocks[i]); //Initialize locks
    }
    //free(nlocks);
    
} //End of algoEdgeApproxDominatingEdgesLinearSearch


