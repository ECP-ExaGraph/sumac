
#include "defs.h"
#include "omp.h"
/**
 * Run a small matching problem
 * n the number of nodes
 * m the number of nodes
 * nedges the number of edges
 * v1 is the source for each of the nedges 
 * v2 is the target for each of the nedges
 * weight is the weight of each of the nedges
 * mi is a vector saying which of v1 and v2 are used, length >= nedges
 * 
 * Written by Ying Wang
 */
void intmatch(long n, long m, long nedges, long *v1, long *v2, double *weight, long *mi)
{
    double ret, al;
    double *l1, *l2, *w;
    long *match1, *match2;
    long i, j, k, p, q, r, t1, t2;
    long *s, *t, *deg, *offset, *list, *index;
    long cardinality=0;
    
    double time1=0; 
    // allocate memory for problem
    l1 = new double[n];
    l2 = new double[n+m];
    s = new long[n+m];
    t = new long[n+m];
    offset = new long[n];
    deg = new long[n];
    list = new long[nedges + n];
    index = new long[nedges+n];
    w = new double[nedges + n];
    match1 = new long[n];
    match2 = new long[n+m];
    
    // track modifications to t
    long *tmod, ntmod=0;
    tmod = new long[n+m];
    
    //Build CSR format from the edge list:
    for (i = 0; i < n; i++) {
      offset[i] = 0;
      deg[i] = 1;
    }
    for (i = 0; i < nedges; i++) deg[v1[i]]++;
    for (i = 1; i < n; i++) offset[i] = offset[i-1] + deg[i-1];
    for (i = 0; i < n; i++) deg[i] = 0;
    for (i = 0; i < nedges; i++) {
      list[offset[v1[i]] + deg[v1[i]]] = v2[i];
      w[offset[v1[i]] + deg[v1[i]]] = weight[i];
      index[offset[v1[i]] + deg[v1[i]]] = i;
      deg[v1[i]]++;
    }
    for (i = 0; i < n; i++) {
      list[offset[i] + deg[i]] = m + i;
      w[offset[i] + deg[i]] = 0;
      index[offset[i] + deg[i]] = -1;
      deg[i]++;
    }
    for (i = 0; i < n; i++) {
      l1[i] = 0;
      for (j = 0; j < deg[i]; j++) {
	if (w[offset[i]+j] > l1[i]) l1[i] = w[offset[i] + j];
      }
    }

    //Compute Matching:
    time1 = omp_get_wtime();
    // initialize the primal match
    for (i = 0; i < n; i++) {
      match1[i] = -1;
    }
    // initialize the dual variables l2
    for (i = 0; i < n + m; i++) {
      l2[i] = 0;
      match2[i] = -1;
    }
    // initialize t once
    for (j=0; j < n+m; j++) {
      t[j] = -1;
    }
    
    for (i = 0; i < n; i++) {
      for (j=0; j<ntmod; j++) {
	t[tmod[j]] = -1;
      }
      ntmod = 0;
      
      // clear the queue and add i to the head
      s[p = q = 0] = i;
      for(; p <= q; p++) {
	if (match1[i] >= 0) break;
	k = s[p];
	for (r = 0; r < deg[k]; r++) {
	  j = list[offset[k] + r];
	  if (w[offset[k] + r] < l1[k] + l2[j] - 1e-8) continue;
	  if (t[j] < 0) {
	    s[++q] = match2[j];
	    t[j] = k;
	    tmod[ntmod]=j; // save our modification to t
	    ntmod++;
	    if (match2[j] < 0) {
	      for(; j>=0 ;) {
		k = match2[j] = t[j];
		// reusing p here is okay because we'll
		// stop below
		p = match1[k];
		match1[k] = j;
		j = p;
	      }
	      break; // we found an alternating path and updated
	    }
	  }
	}
      }
      if (match1[i] < 0) {
	al = 1e20;
	for (j = 0; j < p; j++) {
	  t1 = s[j];
	  for (k = 0; k < deg[t1]; k++) {
	    t2 = list[offset[t1] + k];
	    if (t[t2] < 0 && l1[t1] + l2[t2] - w[offset[t1] + k] < al) {
	      al = l1[t1] + l2[t2] - w[offset[t1] + k];
	    }
	  }
	}
	for (j = 0; j < p; j++) l1[s[j]] -= al;
	//for (j = 0; j < n + m; j++) if (t[j] >= 0) l2[j] += al;
	for (j=0; j<ntmod; j++) { l2[tmod[j]] += al; }
	i--;
	continue;
      }
    }
    time1  = omp_get_wtime() - time1;
    
    ret = 0;
    cardinality = 0;
    for (i = 0; i < n; i++) {
      for (j = 0; j < deg[i]; j++) {
	if (list[offset[i] + j] == match1[i]) {
	  ret += w[offset[i] + j];
	  cardinality++;
	}
      }
    }        
    
    // build the matching indicator 
    for (i=0; i<nedges; i++) {
        mi[i] = 0;
    }
    for (i=0; i<n; i++) {
      if (match1[i] < m) {
	for (j = 0; j < deg[i]; j++) {
	  if (list[offset[i] + j] == match1[i]) {
	    mi[index[offset[i]+j]] = 1;
	  }
	}
      }
    }
    
    delete[] index;    
    delete[] l1;
    delete[] l2;
    delete[] s;
    delete[] t;
    delete[] offset;
    delete[] deg;
    delete[] list;
    delete[] w;
    delete[] match1;
    delete[] match2;
    delete[] tmod;
    
    printf("******************************************\n");
    printf("Exact Matching Results:\n");
    printf("******************************************\n");
    printf("Cardinality  = %ld\n", cardinality/2);
    printf("Weight       = %g\n",   ret);
    printf("Time         = %lf sec\n", time1);
    printf("******************************************\n");
}

/** Holder function for previous code, should be removed and converted as soon as possible
 */
void exact_match(graph* G)
{
   //Get the iterators for the graph:
  long NVer     = G->numVertices;
  long NS       = G->sVertices;
  long NT       = NVer - NS;
  long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
  long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
  printf("NS= %ld  NT=%ld  NE=%ld\n", NS, NT, NEdge);

  //Build edge list: 
  long m = NS;
  long n = NT;
  long nedges = NEdge;
  
  long *v1    = new long[nedges];
  long *v2    = new long[nedges];
  double *wt  = new double[nedges];
  long *ind   = new long[nedges]; 
  
#pragma omp parallel for
  for (long i=0; i<NEdge; i++) {
    v1[i] = verInd[i].head;
    v2[i] = verInd[i].tail;
    wt[i] = verInd[i].weight;
    ind[i]= 0;
  }
  
  intmatch(m, n, nedges, v1, v2, wt, ind);
  
  delete[] v1;
  delete[] v2;
  delete[] ind;
  delete[] wt;
  
}
