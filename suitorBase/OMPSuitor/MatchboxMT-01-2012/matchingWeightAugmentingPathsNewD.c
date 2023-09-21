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
#include "defs.h"

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////  AUGMENTING PATHS OF THREE  ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
void algoAugmentingPathThreeWithLossSerialNew( graph* G, long *Mate, double loss)
{
    printf("Within function algoAugmentingPathThreeWithLossSerialNew()\n");
    //Get the iterators for the graph:
    //Get the iterators for the graph:
    long NVer     = G->numVertices;
    long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
    long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
    edge *verInd  = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
    printf("NV= %ld  NE=%ld\n", NVer, NEdge);
    
    //Allocate Data Structures:
    double weight1 = 0, weight2 = 0, weight3 = 0, heaviest=0;
    double totalTime=0;
    short foundMate = 0;
    totalTime = get_seconds();
    for(long v=0; v<NVer; v++) {  // Loop over vertices
        if (Mate[v] != -1) //Already matched
            continue;
        long adj1 = verPtr[v];
        long adj2 = verPtr[v+1];
        long bestW = -1, bestX = -1, bestY = -1;
        double bestGain = -1; //Start with a negative gain
        foundMate = 0;
        for(long k = adj1; k < adj2; k++) // Loop over nighbors of v
        {
            long w = verInd[k].tail;
            weight1 = verInd[k].weight;
            assert(Mate[w] != -1); //Sanity check
            long x = Mate[w];
            //Find the weight of the matching edge
            long adjM1 = verPtr[w];
            long adjM2 = verPtr[w+1];
            weight2 = -1; //For sanity check
            for(long kM = adjM1; kM < adjM2; kM++) // Loop over nighbors of v
            {
                if (verInd[kM].tail == x) {
                    weight2 = verInd[kM].weight;
                    break;
                }
            }//End of for(k-M)
            assert(weight2 != -1); //Sanity check
            /// Now, find an augmenting path of length three
            long adj11 = verPtr[x];
            long adj12 = verPtr[x+1];
            for(long kk = adj11; kk < adj12; kk++) // Loop over nighbors of x
            {
                long y = verInd[kk].tail;
                if (Mate[y] != -1) //Already matched
                    continue;
                if (y == v) //Get rid of triangles --- blossoms :-)
                    continue;
                weight3 = verInd[kk].weight;
                double gain = (double)((double)(weight1 + weight3)/(double)weight2);
                if( gain >= loss) //Check if the loss is acceptable
                {
                    //printf("Found an augmenting path: (%ld, %ld, -- %ld, %ld)\n",v,w,x,y);
                    //printf("Weights: (%lf + %lf) vs. %lf\n\n", weight1, weight3, weight2);
                    //Store the best values seen so far:
                    if(gain > bestGain) {
                        bestGain = gain;
                        bestW = w;
                        bestX = x;
                        bestY = y;
                    }
                    foundMate = 1; //Found at least one good option to augment
                }//End of if()
            }//End of for(kk)
        }//End of for(k)
        if(foundMate) {
            Mate[v] = bestW;
            Mate[bestW] = v;
            Mate[bestX] = bestY;
            Mate[bestY] = bestX;
        }//End of if(foundMate)
    }//End of for(v)
    totalTime  = get_seconds() - totalTime;
    
    printf("***********************************************\n");
    printf("Time for augmentation      : %lf sec\n", totalTime);
    printf("***********************************************\n");
    
} //End of algoAugmentingPathThreeWithLossSerialNew

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////  WEIGHT AUGMENTING CYCLES OF FOUR  ////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
void algoWeightAugmentingCyclesFourSerialNew( graph* G, long *Mate)
{
/*
    printf("Within function algoWeightAugmentingCyclesFourSerialNew()\n");
    //Get the iterators for the graph:
    //Get the iterators for the graph:
    long NVer     = G->numVertices;
    long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
    long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
    edge *verInd  = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
    printf("NV= %ld  NE=%ld\n", NVer, NEdge);
    
    //Allocate Data Structures:
    double weight1 = 0, weight2 = 0, weight3 = 0, weight4=0;
    double totalTime=0;
    short foundMate = 0;
    totalTime = get_seconds();
    for(long v=0; v<NVer; v++) {  // Loop over vertices
        if (Mate[v] != -1) //Already matched
            continue;
        long adj1 = verPtr[v];
        long adj2 = verPtr[v+1];
        long bestW = -1, bestX = -1, bestY = -1;
        double bestGain = -1; //Start with a negative gain
        foundMate = 0;
        for(long k = adj1; k < adj2; k++) // Loop over nighbors of v
        {
            long w = verInd[k].tail;
            weight1 = verInd[k].weight;
            assert(Mate[w] != -1); //Sanity check
            long x = Mate[w];
            //Find the weight of the matching edge
            long adjM1 = verPtr[w];
            long adjM2 = verPtr[w+1];
            weight2 = -1; //For sanity check
            for(long kM = adjM1; kM < adjM2; kM++) // Loop over nighbors of v
            {
                if (verInd[kM].tail == x) {
                    weight2 = verInd[kM].weight;
                    break;
                }
            }//End of for(k-M)
            assert(weight2 != -1); //Sanity check
            /// Now, find an augmenting path of length three
            long adj11 = verPtr[x];
            long adj12 = verPtr[x+1];
            for(long kk = adj11; kk < adj12; kk++) // Loop over nighbors of x
            {
                long y = verInd[kk].tail;
                if (Mate[y] != -1) //Already matched
                    continue;
                if (y == v) //Get rid of triangles --- blossoms :-)
                    continue;
                weight3 = verInd[kk].weight;
                double gain = (double)((double)(weight1 + weight3)/(double)weight2);
                if( gain >= loss) //Check if the loss is acceptable
                {
                    //printf("Found an augmenting path: (%ld, %ld, -- %ld, %ld)\n",v,w,x,y);
                    //printf("Weights: (%lf + %lf) vs. %lf\n\n", weight1, weight3, weight2);
                    //Store the best values seen so far:
                    if(gain > bestGain) {
                        bestGain = gain;
                        bestW = w;
                        bestX = x;
                        bestY = y;
                    }
                    foundMate = 1; //Found at least one good option to augment
                }//End of if()
            }//End of for(kk)
        }//End of for(k)
        if(foundMate) {
            Mate[v] = bestW;
            Mate[bestW] = v;
            Mate[bestX] = bestY;
            Mate[bestY] = bestX;
        }//End of if(foundMate)
    }//End of for(v)
    totalTime  = get_seconds() - totalTime;
    
    printf("***********************************************\n");
    printf("Time for augmentation      : %lf sec\n", totalTime);
    printf("***********************************************\n");
 
 */
} //End of algoAugmentingPathThreeWithLossSerialNew

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////  DOMINATING EDGE ALGORITHM  ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//Assumes Edges are sorted in descending order
void algoAugmentingPathThreeWithLossSerialNewSorted( graph* G, long *Mate, double loss)
{
    printf("Within function algoAugmentingPathThreeWithLossSerialNew()\n");
    //Get the iterators for the graph:
    //Get the iterators for the graph:
    long NVer     = G->numVertices;
    long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
    long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
    edge *verInd  = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
    printf("NV= %ld  NE=%ld\n", NVer, NEdge);
    
    //Allocate Data Structures:
    double weight1 = 0, weight2 = 0, weight3 = 0, heaviest=0;
    double totalTime=0;
    short foundMate = 0;
    totalTime = get_seconds();
    for(long v=0; v<NVer; v++) {  // Loop over vertices
        if (Mate[v] != -1) //Already matched
            continue;
        long adj1 = verPtr[v];
        long adj2 = verPtr[v+1];
        foundMate = 0;
        for(long k = adj1; k < adj2; k++) // Loop over nighbors of v
        {
            long w = verInd[k].tail;
            weight1 = verInd[k].weight;
            assert(Mate[w] != -1); //Sanity check
            long x = Mate[w];
            //Find the weight of the matching edge
            long adjM1 = verPtr[w];
            long adjM2 = verPtr[w+1];
            weight2 = -1; //For sanity check
            for(long kM = adjM1; kM < adjM2; kM++) // Loop over nighbors of v
            {
                if (verInd[kM].tail == x) {
                    weight2 = verInd[kM].weight;
                    break;
                }
            }//End of for(k-M)
            assert(weight2 != -1); //Sanity check
            /// Now, find an augmenting path of length three
            long adj11 = verPtr[x];
            long adj12 = verPtr[x+1];
            for(long kk = adj11; kk < adj12; kk++) // Loop over nighbors of x
            {
                long y = verInd[kk].tail;
                if (Mate[y] != -1) //Already matched
                    continue;
                if (y == v) //Get rid of triangles --- blossoms :-)
                    continue;
                weight3 = verInd[kk].weight;
                if((double)((double)(weight1 + weight3)/(double)weight2) >= loss) //Check if the loss is acceptable
                {
                    //printf("Found an augmenting path: (%ld, %ld, -- %ld, %ld)\n",v,w,x,y);
                    //printf("Weights: (%lf + %lf) vs. %lf\n\n", weight1, weight3, weight2);
                    //Augment the matching:
                    Mate[v] = w;
                    Mate[w] = v;
                    Mate[x] = y;
                    Mate[y] = x;
                    foundMate = 1;
                    break;
                }//End of if()
            }//End of for(kk)
            if(foundMate)
                break; //Found a mate for v
        }//End of for(k)
    }//End of for(v)
    totalTime  = get_seconds() - totalTime;
    
    printf("***********************************************\n");
    printf("Time for augmentation      : %lf sec\n", totalTime);
    printf("***********************************************\n");
    
} //End of algoEdgeApproxDominatingEdgesLinearSearch

