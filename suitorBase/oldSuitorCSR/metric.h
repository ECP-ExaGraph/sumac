#ifndef _METRIC_H_
#define _METRIC_H_

long eval_linGapArr(graph* g);
double eval_logGapArr(graph* g);
long eval_linGapArrLocal(graph* g, long v1, long v2);
double eval_logGapArrLocal(graph* g, long v1, long v2);
long eval_linGapArrLocalSwap(graph* g, long v1, long v2);
double eval_logGapArrLocalSwap(graph* g, long v1, long v2);
long eval_linGapArrSwap(graph* g, long v1, long v2);
long eval_linGapArrSingle(graph* g, long v1);
long eval_linGapArrSingleSwap(graph* g, long v1, long v1swap, long v2swap);
long eval_logGapArrSingle(graph* g, long v1);
#endif