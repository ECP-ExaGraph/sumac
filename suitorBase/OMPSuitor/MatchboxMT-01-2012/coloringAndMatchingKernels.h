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

#ifndef _coloringAndMatching_
#define _coloringAndMatching_

//Matchbox headers:
#include "defs.h"

//SNAP headers:
#include "graph_defs.h"
#include "graph_kernels.h"
#include "sprng.h"
#include "utils.h"

//Coloring Kernels:
void algoDistanceOneVertexColoring(graph_t* G, long *vtxColor);
void algoDistanceOneVertexColoringSerial(graph_t* G, long *vtxColor);

//Matching Kernels:
void algoEdgeApproxDominatingEdgesLinearSearch( graph_t* G, long *Mate);
void algoEdgeApproxDominatingEdgesLinearSearchNew( graph *G, long *Mate);
void algoEdgeApproxDominatingEdgesLinearSearchSerial( graph_t* G, long *Mate);
//Augmenting paths of length three
void algoAugmentingPathThreeWithLossSerialNew( graph* G, long *Mate, double loss);
void algoAugmentingPathThreeWithLossSerialNewSorted( graph* G, long *Mate, double loss);
//Weight augmenting cycles of four
void algoWeightAugmentingCyclesFourSerialNew( graph* G, long *Mate);

//Maximal Chordal Subgraph:
//void algoEdgeApproxDominatingEdgesLinearSearch( graph_t* G, long *Mate);

#endif
