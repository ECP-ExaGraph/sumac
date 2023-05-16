#include <cmath>
#include "graph.h"
#include "rand.h"
#include "io.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cassert>
#include <chrono>
#include <cstring>
#include <omp.h>
#include <limits.h>
#include <float.h>
#include <algorithm>


long eval_linGapArr(graph* g)
{
  long linGap = 0; 
  #pragma omp parallel for reduction(+: linGap)
  for (int i=0;i<g->num_verts;i++){
      long* adjs = out_vertices(g,i);
      long degree = out_degree(g,i);
      long* tempAdjsList = new long[degree];
      #pragma omp parallel for
      for (int j=0;j<degree;j++)
        tempAdjsList[j] = adjs[j];
        //printf("%d\n",tempAdjsList[j]);}
      #pragma omp parallel for
      for (int j=0;j<degree;j++){
        tempAdjsList[j] = g->label_map[tempAdjsList[j]];
      }
      std::sort(tempAdjsList,tempAdjsList+degree);
      #pragma omp parallel for reduction(+: linGap)
      for (int j=1;j<degree;j++)
      {
        linGap += tempAdjsList[j] - tempAdjsList[j-1];
      }
      
  }
  return linGap;
}
double eval_logGapArr(graph* g)
{
  double logGap = 0; 
  #pragma omp parallel for reduction(+: logGap)
  for (int i=0;i<g->num_verts;i++){
      long* adjs = out_vertices(g,i);
      long degree = out_degree(g,i);
      long* tempAdjsList = new long[degree];
      #pragma omp parallel for
      for (int j=0;j<degree;j++)
        tempAdjsList[j] = adjs[j];
      #pragma omp parallel for
      for (int j=0;j<degree;j++){
        tempAdjsList[j] = g->label_map[tempAdjsList[j]];
      }
      std::sort(tempAdjsList,tempAdjsList+degree);
      #pragma omp parallel for reduction(+: logGap)
      for (int j=1;j<degree;j++)
      {
        if(tempAdjsList[j]-tempAdjsList[j-1] == 0)
          continue;
        logGap += log(tempAdjsList[j] - tempAdjsList[j-1]);
      }
      
  }
  return logGap;
}


long eval_linGapArrSingle(graph* g, long v1){
  long linGap = 0; 
  long* adjs = out_vertices(g,v1);
  long degree = out_degree(g,v1);
  long* tempAdjsList = new long[degree];
  #pragma omp parallel for
  for (int j=0;j<degree;j++)
    tempAdjsList[j] = adjs[j];
  #pragma omp parallel for
  for (int j=0;j<degree;j++){
    tempAdjsList[j] = g->label_map[tempAdjsList[j]];
  }
  std::sort(tempAdjsList,tempAdjsList+degree);
  #pragma omp parallel for reduction(+: linGap)
  for (int j=1;j<degree;j++)
  {
    linGap += tempAdjsList[j] - tempAdjsList[j-1];
  } 
  delete [] tempAdjsList;
  return linGap;
}
long eval_logGapArrSingle(graph* g, long v1){
  double logGap = 0; 
  long* adjs = out_vertices(g,v1);
  long degree = out_degree(g,v1);
  long* tempAdjsList = new long[degree];
  #pragma omp parallel for
  for (int j=0;j<degree;j++)
    tempAdjsList[j] = adjs[j];
  #pragma omp parallel for
  for (int j=0;j<degree;j++){
    tempAdjsList[j] = g->label_map[tempAdjsList[j]];
  }
  std::sort(tempAdjsList,tempAdjsList+degree);
  #pragma omp parallel for reduction(+: logGap)
  for (int j=1;j<degree;j++)
  {
    if(tempAdjsList[j]-tempAdjsList[j-1] == 0)
      continue;
    logGap += log(tempAdjsList[j] - tempAdjsList[j-1]);
  } 
  delete [] tempAdjsList;
  return logGap;
}

long eval_linGapArrLocal(graph* g, long v1, long v2){
  long verts[2] = {v1,v2};
  long linGap = 0; 
  long neighborGaps = 0;
  #pragma omp parallel for reduction(+: linGap)
  for (int i=0;i<2;i++){
      long* adjs = out_vertices(g,verts[i]);
      long degree = out_degree(g,verts[i]);
      long* tempAdjsList = new long[degree];
      #pragma omp parallel for
      for (int j=0;j<degree;j++)
        tempAdjsList[j] = adjs[j];
      #pragma omp parallel for reduction(+: neighborGaps)
      for (int j=0;j<degree;j++){
        if(adjs[j]!=v1 and adjs[j]!=v2)
          neighborGaps+=eval_linGapArrSingle(g,adjs[j]);
      }
      #pragma omp parallel for
      for (int j=0;j<degree;j++){
        tempAdjsList[j] = g->label_map[tempAdjsList[j]];
      }
      std::sort(tempAdjsList,tempAdjsList+degree);
      #pragma omp parallel for reduction(+: linGap)
      for (int j=1;j<degree;j++)
      {
        linGap += tempAdjsList[j] - tempAdjsList[j-1];
      } 
      delete [] tempAdjsList;
  }
  //printf("v: %ld, %ld  lg: %ld ng: %ld\n",v1,v2,linGap,neighborGaps);
  return linGap + neighborGaps;
}







double eval_logGapArrLocal(graph* g, long v1, long v2){
  long verts[2] = {v1,v2};
  double logGap = 0; 
  double neighborGaps = 0;
  #pragma omp parallel for reduction(+: logGap)
  for (int i=0;i<2;i++){
      long* adjs = out_vertices(g,verts[i]);
      long degree = out_degree(g,verts[i]);
      long* tempAdjsList = new long[degree];
      #pragma omp parallel for
      for (int j=0;j<degree;j++)
        tempAdjsList[j] = adjs[j];
      #pragma omp parallel for reduction(+: neighborGaps)
      for (int j=0;j<degree;j++){
        if(adjs[j]!=v1 and adjs[j]!=v2)
          neighborGaps+=eval_logGapArrSingle(g,adjs[j]);
      }
      #pragma omp parallel for
      for (int j=0;j<degree;j++){
        tempAdjsList[j] = g->label_map[tempAdjsList[j]];
      }
      std::sort(tempAdjsList,tempAdjsList+degree);
      #pragma omp parallel for reduction(+: logGap)
      for (int j=1;j<degree;j++)
      {
        if(tempAdjsList[j] - tempAdjsList[j-1]==0)
          continue;
        logGap += log(tempAdjsList[j] - tempAdjsList[j-1]);
      } 
      delete [] tempAdjsList;
  }
  return logGap + neighborGaps;
}


long eval_linGapArrSingleSwap(graph* g, long v1, long v1swap, long v2swap){
  long verts[2] = {v1swap,v2swap};
  long linGap = 0; 
  long* adjs = out_vertices(g,v1);
  long degree = out_degree(g,v1);
  long* tempAdjsList = new long[degree];
  #pragma omp parallel for
  for (int j=0;j<degree;j++)
    tempAdjsList[j] = adjs[j];
  #pragma omp parallel for
  for (int j=0;j<degree;j++){
    if(tempAdjsList[j] == verts[0])
      tempAdjsList[j] = g->label_map[verts[1]];
    else if(tempAdjsList[j] == verts[1])
      tempAdjsList[j] = g->label_map[verts[0]];
    else
      tempAdjsList[j] = g->label_map[tempAdjsList[j]];
  }
  std::sort(tempAdjsList,tempAdjsList+degree);
  #pragma omp parallel for reduction(+: linGap)
  for (int j=1;j<degree;j++)
  {
    linGap += tempAdjsList[j] - tempAdjsList[j-1];
  } 
  delete [] tempAdjsList;
  return linGap;
}


double eval_logGapArrSingleSwap(graph* g, long v1, long v1swap, long v2swap){
  long verts[2] = {v1swap,v2swap};
  double logGap = 0; 
  long* adjs = out_vertices(g,v1);
  long degree = out_degree(g,v1);
  long* tempAdjsList = new long[degree];
  #pragma omp parallel for
  for (int j=0;j<degree;j++)
    tempAdjsList[j] = adjs[j];
  #pragma omp parallel for
  for (int j=0;j<degree;j++){
    if(tempAdjsList[j] == verts[0])
      tempAdjsList[j] = g->label_map[verts[1]];
    else if(tempAdjsList[j] == verts[1])
      tempAdjsList[j] = g->label_map[verts[0]];
    else
      tempAdjsList[j] = g->label_map[tempAdjsList[j]];
  }
  std::sort(tempAdjsList,tempAdjsList+degree);
  #pragma omp parallel for reduction(+: logGap)
  for (int j=1;j<degree;j++)
  {
    if(tempAdjsList[j] - tempAdjsList[j-1]==0)
      continue;
    logGap += log(tempAdjsList[j] - tempAdjsList[j-1]);
  } 
  delete [] tempAdjsList;
  return logGap;
}



long eval_linGapArrLocalSwap(graph* g, long v1, long v2){
  long verts[2] = {v1,v2};
  long linGap = 0; 
  long neighborGaps = 0;
  #pragma omp parallel for reduction(+: linGap)
  for (int i=0;i<2;i++){
      long* adjs = out_vertices(g,verts[i]);
      long degree = out_degree(g,verts[i]);
      long* tempAdjsList = new long[degree];
      #pragma omp parallel for
      for (int j=0;j<degree;j++)
        tempAdjsList[j] = adjs[j];
      #pragma omp parallel for reduction(+: neighborGaps)
      for (int j=0;j<degree;j++){
        if(adjs[j]!=v1 and adjs[j]!=v2)
          neighborGaps+=eval_linGapArrSingleSwap(g,adjs[j],v1,v2);
      }
      #pragma omp parallel for
      for (int j=0;j<degree;j++){
        if(tempAdjsList[j] == verts[0])
          tempAdjsList[j] = g->label_map[verts[1]];
        else if(tempAdjsList[j] == verts[1])
          tempAdjsList[j] = g->label_map[verts[0]];
        else
          tempAdjsList[j] = g->label_map[tempAdjsList[j]];
      }
      std::sort(tempAdjsList,tempAdjsList+degree);
      #pragma omp parallel for reduction(+: linGap)
      for (int j=1;j<degree;j++)
      {
        linGap += tempAdjsList[j] - tempAdjsList[j-1];
      } 
      delete [] tempAdjsList;
  }
  return linGap + neighborGaps;
}

double eval_logGapArrLocalSwap(graph* g, long v1, long v2){
  long verts[2] = {v1,v2};
  double logGap = 0; 
  double neighborGaps = 0;
  #pragma omp parallel for reduction(+: logGap)
  for (int i=0;i<2;i++){
      long* adjs = out_vertices(g,verts[i]);
      long degree = out_degree(g,verts[i]);
      long* tempAdjsList = new long[degree];
      #pragma omp parallel for
      for (int j=0;j<degree;j++)
        tempAdjsList[j] = adjs[j];
      #pragma omp parallel for reduction(+: neighborGaps)
      for (int j=0;j<degree;j++){
        if(adjs[j]!=v1 and adjs[j]!=v2)
          neighborGaps+=eval_logGapArrSingleSwap(g,adjs[j],v1,v2);
      }
      #pragma omp parallel for
      for (int j=0;j<degree;j++){
        if(tempAdjsList[j] == verts[0])
          tempAdjsList[j] = g->label_map[verts[1]];
        else if(tempAdjsList[j] == verts[1])
          tempAdjsList[j] = g->label_map[verts[0]];
        else
          tempAdjsList[j] = g->label_map[tempAdjsList[j]];
      }
      std::sort(tempAdjsList,tempAdjsList+degree);
      #pragma omp parallel for reduction(+: logGap)
      for (int j=1;j<degree;j++)
      {
        if(tempAdjsList[j] - tempAdjsList[j-1]==0)
          continue;
        logGap += log(tempAdjsList[j] - tempAdjsList[j-1]);
      } 
      delete [] tempAdjsList;
  }
  return logGap + neighborGaps;
}
long eval_linGapArrSwap(graph* g, long v1, long v2){
  long verts[2] = {v1,v2};
  long linGap = 0; 
  #pragma omp parallel for reduction(+: linGap)
  for (int i=0;i<g->num_verts;i++){
      long* adjs = out_vertices(g,verts[i]);
      long degree = out_degree(g,verts[i]);
      long* tempAdjsList = new long[degree];
      #pragma omp parallel for
      for (int j=0;j<degree;j++)
        tempAdjsList[j] = adjs[j];
      #pragma omp parallel for
      for (int j=0;j<degree;j++){
        if(tempAdjsList[j] == verts[0])
          tempAdjsList[j] = verts[1];
        else if(tempAdjsList[j] == verts[1])
          tempAdjsList[j] = verts[0];
        else
          tempAdjsList[j] = g->label_map[tempAdjsList[j]];
      }
      std::sort(tempAdjsList,tempAdjsList+degree);
      #pragma omp parallel for reduction(+: linGap)
      for (int j=1;j<degree;j++)
      {
        linGap += tempAdjsList[j] - tempAdjsList[j-1];
      } 
      delete [] tempAdjsList;
  }
  return linGap;
}
double eval_logGapArrSwap(graph* g, long v1, long v2){
  long verts[2] = {v1,v2};
  double logGap = 0; 
  #pragma omp parallel for reduction(+: logGap)
  for (int i=0;i<g->num_verts;i++){
      long* adjs = out_vertices(g,verts[i]);
      long degree = out_degree(g,verts[i]);
      long* tempAdjsList = new long[degree];
      #pragma omp parallel for
      for (int j=0;j<degree;j++)
        tempAdjsList[j] = adjs[j];
      #pragma omp parallel for
      for (int j=0;j<degree;j++){
        if(tempAdjsList[j] == verts[0])
          tempAdjsList[j] = verts[1];
        else if(tempAdjsList[j] == verts[1])
          tempAdjsList[j] = verts[0];
        else
          tempAdjsList[j] = g->label_map[tempAdjsList[j]];
      }
      std::sort(tempAdjsList,tempAdjsList+degree);
      #pragma omp parallel for reduction(+: logGap)
      for (int j=1;j<degree;j++)
      {
        if(tempAdjsList[j] - tempAdjsList[j-1]==0)
          continue;
        logGap += log(tempAdjsList[j] - tempAdjsList[j-1]);
      } 
      delete [] tempAdjsList;
  }
  return logGap;
}