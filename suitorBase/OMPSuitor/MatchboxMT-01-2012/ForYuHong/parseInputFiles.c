#include "defs.h"

/* Since graph is undirected, sort each edge head --> tail AND tail --> head */
void SortEdgesUndirected2(long NV, long NE, edge *list1, edge *list2, long *ptrs) {

  for (long i = 0; i < NV + 2; i++) 
    ptrs[i] = 0;
  ptrs += 2;

  /* Histogram key values */
  for (long i = 0; i < NE; i++) {
    ptrs[list1[i].head]++;
    ptrs[list1[i].tail]++;
  }
  /* Compute start index of each bucket */
  for (long i = 1; i < NV; i++) 
    ptrs[i] += ptrs[i-1];
  ptrs --;

  /* Move edges into its bucket's segment */
  for (long i = 0; i < NE; i++) {
    long head   = list1[i].head;
    long index          = ptrs[head]++;
    //list2[index].id     = list1[i].id;
    list2[index].head   = list1[i].head;
    list2[index].tail   = list1[i].tail;
    list2[index].weight = list1[i].weight;

    long tail   = list1[i].tail;
    index               = ptrs[tail]++;
    //list2[index].id     = list1[i].id;
    list2[index].head   = list1[i].tail;
    list2[index].tail   = list1[i].head;
    list2[index].weight = list1[i].weight;
  } 
}//End of SortEdgesUndirected2()

/* Sort each node's neighbors by tail from smallest to largest. */
void SortNodeEdgesByIndex2(long NV, edge *list1, edge *list2, long *ptrs) {
  
  for (long i = 0; i < NV; i++) {
    edge *edges1 = list1 + ptrs[i];
    edge *edges2 = list2 + ptrs[i];
    long size    = ptrs[i+1] - ptrs[i];

    /* Merge Sort */
    for (long skip = 2; skip < 2 * size; skip *= 2) {
      for (long sect = 0; sect < size; sect += skip)  {
	long j = sect;
	long l = sect;
	long half_skip = skip / 2;
	long k = sect + half_skip;
	
	long j_limit = (j + half_skip < size) ? j + half_skip : size;
	long k_limit = (k + half_skip < size) ? k + half_skip : size;
	
	while ((j < j_limit) && (k < k_limit)) {
	  if   (edges1[j].tail < edges1[k].tail) {edges2[l] = edges1[j]; j++; l++;}
	  else                                   {edges2[l] = edges1[k]; k++; l++;}
	}
	
	while (j < j_limit) {edges2[l] = edges1[j]; j++; l++;}
	while (k < k_limit) {edges2[l] = edges1[k]; k++; l++;}
      }

      edge *tmp = edges1;
      edges1 = edges2;
      edges2 = tmp;
    }
    // result is in list2, so move to list1
    if (edges1 == list2 + ptrs[i])
      for (long j = ptrs[i]; j < ptrs[i+1]; j++) list1[j] = list2[j];
  } 
}//End of SortNodeEdgesByIndex2()

/*-------------------------------------------------------*
 * This function reads a MATRIX MARKET file and builds the graph
 *-------------------------------------------------------*/
void parse_MatrixMarket(graph * G, char *fileName) {
  printf("Parsing a Matrix Market File...\n");
  int nthreads;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();    
  }
  printf("parse_MatrixMarket: Number of threads: %d\n", nthreads);

  double time1, time2;
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  
  /* -----      Read File in Matrix Market Format     ------ */
  //Parse the first line:
  char line[1024];
  fgets(line, 1024, file);
  char  LS1[25], LS2[25], LS3[25], LS4[25], LS5[25];    
  if (sscanf(line, "%s %s %s %s %s", LS1, LS2, LS3, LS4, LS5) != 5) {
    printf("parse_MatrixMarket(): bad file format - 01");
    exit(1);
  }
  printf("%s %s %s %s %s\n", LS1, LS2, LS3, LS4, LS5);
  if ( strcmp(LS1,"%%MatrixMarket") != 0 ) {
    printf("Error: The first line should start with %%MatrixMarket word \n");
    exit(1);
  }
  if ( !( strcmp(LS2,"matrix")==0 || strcmp(LS2,"Matrix")==0 || strcmp(LS2,"MATRIX")==0 ) ) {
    printf("Error: The Object should be matrix or Matrix or MATRIX \n");
    exit(1);
  }
  if ( !( strcmp(LS3,"coordinate")==0 || strcmp(LS3,"Coordinate")==0 || strcmp(LS3,"COORDINATE")==0) ) {
    printf("Error: The Object should be coordinate or Coordinate or COORDINATE \n");
    exit(1);
  }
  int isComplex = 0;    
  if ( strcmp(LS4,"complex")==0 || strcmp(LS4,"Complex")==0 || strcmp(LS4,"COMPLEX")==0 ) {
    isComplex = 1;
    printf("Warning: Will only read the real part. \n");
  }
  int isPattern = 0;
  if ( strcmp(LS4,"pattern")==0 || strcmp(LS4,"Pattern")==0 || strcmp(LS4,"PATTERN")==0 ) {
    isPattern = 1;
    printf("Note: Matrix type is Pattern. Will set all weights to 1.\n");
    //exit(1);
  }
  int isSymmetric = 0, isGeneral = 0;
  if ( strcmp(LS5,"general")==0 || strcmp(LS5,"General")==0 || strcmp(LS5,"GENERAL")==0 )
    isGeneral = 1;
  else {
    if ( strcmp(LS5,"symmetric")==0 || strcmp(LS5,"Symmetric")==0 || strcmp(LS5,"SYMMETRIC")==0 ) {
      isSymmetric = 1;
      printf("Note: Matrix type is Symmetric: Converting it into General type. \n");
    }
  }	
  if ( (isGeneral==0) && (isSymmetric==0) ) 	  {
    printf("Warning: Matrix type should be General or Symmetric. \n");
    exit(1);
  }
  
  /* Parse all comments starting with '%' symbol */
  do {
    fgets(line, 1024, file);
  } while ( line[0] == '%' );
  
  /* Read the matrix parameters */
  long NS=0, NT=0, NV = 0;
  long NE=0;
  if (sscanf(line, "%ld %ld %ld",&NS, &NT, &NE ) != 3) {
    printf("parse_MatrixMarket(): bad file format - 02");
    exit(1);
  }
  NV = NS + NT;
  printf("|S|= %ld, |T|= %ld, |E|= %ld \n", NS, NT, NE);

  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /* S vertices: 0 to NS-1                                               */
  /* T vertices: NS to NS+NT-1                                           */
  /*---------------------------------------------------------------------*/
  //Allocate for Edge Pointer and keep track of degree for each vertex
  long  *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
#pragma omp parallel for
  for (long i=0; i <= NV; i++)
    edgeListPtr[i] = 0; //For first touch purposes
  
  edge *edgeListTmp; //Read the data in a temporary list
  long newNNZ = 0;    //New edges because of symmetric matrices
  long Si, Ti;
  double weight = 1;
  if( isSymmetric == 1 ) {
    printf("Matrix is of type: Symmetric Real or Complex\n");
    printf("Weights will be converted to positive numbers.\n");
    edgeListTmp = (edge *) malloc(2 * NE * sizeof(edge));
    for (long i = 0; i < NE; i++) {
      if (isPattern == 1)
	fscanf(file, "%ld %ld", &Si, &Ti);
      else
	fscanf(file, "%ld %ld %lf", &Si, &Ti, &weight);
      Si--; Ti--;            // One-based indexing
      weight = fabs(weight); //Make it positive  : Leave it as is
      if ( Si == Ti ) {
	edgeListTmp[i].head = Si;       //The S index
	edgeListTmp[i].tail = NS+Ti;    //The T index 
	edgeListTmp[i].weight = weight; //The value
	edgeListPtr[Si+1]++;
	edgeListPtr[NS+Ti+1]++;
      }
      else { //an off diagonal element: Also store the upper part
	//LOWER PART:
	edgeListTmp[i].head = Si;       //The S index 
	edgeListTmp[i].tail = NS+Ti;    //The T index 
	edgeListTmp[i].weight = weight; //The value
	edgeListPtr[Si+1]++;
	edgeListPtr[NS+Ti+1]++;
	//UPPER PART:
	edgeListTmp[NE+newNNZ].head = Ti;       //The S index
	edgeListTmp[NE+newNNZ].tail = NS+Si;    //The T index
	edgeListTmp[NE+newNNZ].weight = weight; //The value
	newNNZ++; //Increment the number of edges
	edgeListPtr[Ti+1]++;
	edgeListPtr[NS+Si+1]++;
      }
    }
  } //End of Symmetric
    /////// General Real or Complex ///////
  else {
    printf("Matrix is of type: Unsymmetric Real or Complex\n");
    printf("Weights will be converted to positive numbers.\n");
    edgeListTmp = (edge *) malloc( NE * sizeof(edge));
    for (long i = 0; i < NE; i++) {
      if (isPattern == 1)
	fscanf(file, "%ld %ld", &Si, &Ti);
      else
	fscanf(file, "%ld %ld %lf", &Si, &Ti, &weight);
      //printf("(%d, %d) %lf\n",Si, Ti, weight);
      Si--; Ti--;            // One-based indexing
      weight = fabs(weight); //Make it positive    : Leave it as is
      edgeListTmp[i].head = Si;       //The S index
      edgeListTmp[i].tail = NS+Ti;    //The T index
      edgeListTmp[i].weight = weight; //The value
      edgeListPtr[Si+1]++;
      edgeListPtr[NS+Ti+1]++;
    }
  } //End of Real or Complex

  fclose(file); //Close the file
  printf("Done reading from file.\n");
  if( isSymmetric ) {
    printf("Modified the number of edges from %d ",&NE);
    NE += newNNZ; //#NNZ might change
    printf("to %ld \n",&NE);
  }

  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = timer();
  for (long i=0; i<=NV; i++) {
    edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = timer();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: 2|E| = %ld, edgeListPtr[NV]= %ld\n", NE*2, edgeListPtr[NV]);

  /*---------------------------------------------------------------------*/
  /* Allocate memory for G & Build it                                    */
  /*---------------------------------------------------------------------*/    
  time1 = timer();
  edge *edgeList = (edge *) malloc( 2*NE * sizeof(edge)); //Every edge stored twice
  time2 = timer();
  printf("Time for allocating memory for marks and edgeList = %lf\n", time2 - time1);
  
  time1 = timer();
  //Keep track of how many edges have been added for a vertex:
  long  *added    = (long *)  malloc( NV  * sizeof(long));
#pragma omp parallel for
  for (long i = 0; i < NV; i++) 
    added[i] = 0;
  
  printf("About to build edgeList...\n");
  //Build the edgeList from edgeListTmp:
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    long head  = edgeListTmp[i].head;
    long tail  = edgeListTmp[i].tail;
    double weight      = edgeListTmp[i].weight;
    
    long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);   
    edgeList[Where].head = head;
    edgeList[Where].tail = tail;
    edgeList[Where].weight = weight;
    //added[head]++;
    //Now add the counter-edge:
    Where = edgeListPtr[tail] + __sync_fetch_and_add(&added[tail], 1);
    edgeList[Where].head = tail;
    edgeList[Where].tail = head;
    edgeList[Where].weight = weight;
    //added[tail]++;
  }
  time2 = timer();
  printf("Time for building edgeList = %lf\n", time2 - time1);
  
  G->sVertices    = NS;
  G->numVertices  = NV;
  G->numEdges     = NE;
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;
  
  free(edgeListTmp);
  free(added);
}

/*-------------------------------------------------------*
 * This function reads a Matrix Market file and build a 
 * matrix in CSR format
 *-------------------------------------------------------*/
void parse_MatrixMarket_CSC(matrix_CSC * M, char *fileName) {
  printf("Parsing a Matrix Market File...\n");
  int nthreads;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  printf("Coloring Rouinte: Number of threads: %d\n", nthreads);
  double time1, time2;
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  /* -----      Read File in Matrix Market Format     ------ */
  //Parse the first line:
  char line[1024];
  fgets(line, 1024, file);
  char  LS1[25], LS2[25], LS3[25], LS4[25], LS5[25];    
  if (sscanf(line, "%s %s %s %s %s", LS1, LS2, LS3, LS4, LS5) != 5) {
    printf("parse_MatrixMarket_CSC(): bad file format - 01\n");
    exit(1);
  }
  printf("%s %s %s %s %s\n", LS1, LS2, LS3, LS4, LS5);
  if ( strcmp(LS1,"%%MatrixMarket") != 0 ) {
    printf("Error: The first line should start with %%MatrixMarket word \n");
    exit(1);
  }
  if ( !( strcmp(LS2,"matrix")==0 || strcmp(LS2,"Matrix")==0 || strcmp(LS2,"MATRIX")==0 ) ) {
    printf("Error: The Object should be matrix or Matrix or MATRIX \n");
    exit(1);
  }
  if ( !( strcmp(LS3,"coordinate")==0 || strcmp(LS3,"Coordinate")==0 || strcmp(LS3,"COORDINATE")==0) ) {
    printf("Error: The Object should be coordinate or Coordinate or COORDINATE \n");
    exit(1);
  }
  int isComplex = 0;    
  if ( strcmp(LS4,"complex")==0 || strcmp(LS4,"Complex")==0 || strcmp(LS4,"COMPLEX")==0 ) {
    isComplex = 1;
    printf("Warning: Will only read the real part. \n");
  }
  if ( strcmp(LS4,"pattern")==0 || strcmp(LS4,"Pattern")==0 || strcmp(LS4,"PATTERN")==0 ) {
    printf("Error: Cannot handle if data type is Pattern \n");
    exit(1);
  }
  int isSymmetric = 0, isGeneral = 0;
  if ( strcmp(LS5,"general")==0 || strcmp(LS5,"General")==0 || strcmp(LS5,"GENERAL")==0 )
    isGeneral = 1;
  else {
    if ( strcmp(LS5,"symmetric")==0 || strcmp(LS5,"Symmetric")==0 || strcmp(LS5,"SYMMETRIC")==0 ) {
      isSymmetric = 1;
      printf("Note: Matrix type is Symmetric: Converting it into General type. \n");
    }
  }
  if ( (isGeneral==0) && (isSymmetric==0) ) 	  {
    printf("Warning: Matrix type should be General or Symmetric. \n");
    exit(1);
  }

  /* Parse all comments starting with '%' symbol */
  do {
    fgets(line, 1024, file);
  } while ( line[0] == '%' );
  
  /* Read the matrix parameters */
  long NS=0, NT=0, NV = 0;
  long NE=0;
  if (sscanf(line, "%ld %ld %ld",&NS, &NT, &NE ) != 3) {
    printf("parse_MatrixMarket_CSC(): bad file format - 02\n");
    exit(1);
  }
  NV = NS + NT;
  printf("|S|= %ld, |T|= %ld, |E|= %ld \n", NS, NT, NE);
  
  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /*---------------------------------------------------------------------*/
  //Allocate for Edge Pointer and keep track of degree for each column
  long  *rowPtr = (long *)  malloc((NT+1) * sizeof(long));
#pragma omp parallel for
  for (long i=0; i <= NT; i++)
    rowPtr[i] = 0;
  
  edge *edgeListTmp; //Read the data in a temporary list
  long newNNZ = 0;    //New edges because of symmetric matrices
  long Si, Ti;
  double weight;
  if( isSymmetric == 1 ) {
    printf("Matrix is of type: Symmetric Real or Complex\n");
    edgeListTmp = (edge *) malloc(2 * NE * sizeof(edge));
    for (long i = 0; i < NE; i++) {
      fscanf(file, "%ld %ld %lf", &Si, &Ti, &weight);
      Si--; Ti--;            // One-based indexing
      //weight = fabs(weight); //Make it positive  : Leave it as is
      if ( Si == Ti ) {
	edgeListTmp[i].head = Si;     //The S index = Row
	edgeListTmp[i].tail = Ti;     //The T index = Col 
	edgeListTmp[i].weight = weight; //The value
	rowPtr[Ti+1]++; //Increment for Column
      }
      else { //an off diagonal element: Also store the upper part
	//LOWER PART:
	edgeListTmp[i].head = Si;       //The S index 
	edgeListTmp[i].tail = Ti;    //The T index 
	edgeListTmp[i].weight = weight; //The value
	rowPtr[Ti+1]++;
	//UPPER PART:
	edgeListTmp[NE+newNNZ].head = Ti;    //The S index
	edgeListTmp[NE+newNNZ].tail = Si;    //The T index
	edgeListTmp[NE+newNNZ].weight = weight; //The value
	newNNZ++; //Increment the number of edges
	rowPtr[Si+1]++;
      }
    }
  } //End of Symmetric
  /////// General Real or Complex ///////
  else {
    printf("Matrix is of type: Unsymmetric Real or Complex\n");
    edgeListTmp = (edge *) malloc( NE * sizeof(edge));
    for (long i = 0; i < NE; i++) {
      fscanf(file, "%ld %ld %lf", &Si, &Ti, &weight);
      Si--; Ti--;            // One-based indexing
      //weight = fabs(weight); //Make it positive    : Leave it as is
      edgeListTmp[i].head = Si;    //The S index = Row
      edgeListTmp[i].tail = Ti;    //The T index = Col
      edgeListTmp[i].weight = weight; //The value
      rowPtr[Ti+1]++;
    }
  } //End of Real or Complex

  fclose(file); //Close the file
  printf("Done reading from file.\n");
  if( isSymmetric ) {
    printf("Modifying number of edges from %d ",&NE);
    NE += newNNZ; //#NNZ might change
    printf("to %ld \n",&NE);
  }
  
  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = timer();
  for (long i=0; i<NT; i++) {
    rowPtr[i+1] += rowPtr[i]; //Prefix Sum
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = timer();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: |E| = %ld, rowPtr[NV]= %ld\n", NE, rowPtr[NT]);
  
  /*---------------------------------------------------------------------*/
  /* Allocate memory for G & Build it                                    */
  /*---------------------------------------------------------------------*/    
  time1 = timer();
  //Every edge stored ONLY ONCE!
  long *rowIndex   = (long*)    malloc (NE * sizeof(long)); 
  double *weights  = (double*)  malloc (NE * sizeof(double));
  time2 = timer();
  printf("Time for allocating memory for rowIndex and edgeList = %lf\n", time2 - time1);

  time1 = timer();
  //Keep track of how many edges have been added for a vertex:
  long  *added    = (long *)  malloc (NT  * sizeof(long));
#pragma omp parallel for
  for (long i = 0; i < NT; i++) 
    added[i] = 0;
  
  printf("Building edgeList...\n");
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    long head      = edgeListTmp[i].head;   //row id
    long tail      = edgeListTmp[i].tail;   //col id
    double weight = edgeListTmp[i].weight; //weight
    long Where     = rowPtr[tail] + __sync_fetch_and_add(&added[tail], 1);
    rowIndex[Where] = head;  //Add the row id
    weights[Where]  = weight;
    //added[tail]++;
  }
  time2 = timer();
  printf("Time for building edgeList = %lf\n", time2 - time1);
  
  M->nRows    = NS;
  M->nCols    = NT;
  M->nNNZ     = NE;
  M->RowPtr   = rowPtr;
  M->RowInd   = rowIndex;
  M->Weights  = weights;    
  
  free(edgeListTmp);
  free(added);
}

/*-------------------------------------------------------*
 * This function reads a Simple Matrix Market file and builds a 
 * matrix in CSR format
 *-------------------------------------------------------*/
void parse_Simple_CSC(matrix_CSC * M, char *fileName) {
  printf("Parsing a Simple Matrix Market File...\n");
  int nthreads;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();    
  }
  printf("File I/O: Number of threads: %d\n", nthreads);
  double time1, time2;      
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  /* -----      Read File in Matrix Market Format     ------ */
  //Parse the first line:
  char line[1024];
  fgets(line, 1024, file);
  /* Read the matrix parameters */
  long NS=0, NT=0, NV = 0;
  long NE=0;
  if (sscanf(line, "%ld %ld %ld",&NS, &NT, &NE ) != 3) {
    printf("parse_MatrixMarket_CSC(): bad file format - 02\n");
    exit(1);
  }
  NV = NS + NT;
  printf("|S|= %ld, |T|= %ld, |E|= %ld \n", NS, NT, NE);
  
  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /*---------------------------------------------------------------------*/
  //Allocate for Edge Pointer and keep track of degree for each column
  long *rowPtr = (long *)  malloc((NT+1) * sizeof(long));
#pragma omp parallel for
  for (long i=0; i <= NT; i++)
    rowPtr[i] = 0;
  
  edge *edgeListTmp; //Read the data in a temporary list
  long Si, Ti;
  double weight;
  /////// General Real or Complex ///////
  printf("Matrix is of type: Unsymmetric Real or Complex\n");
  edgeListTmp = (edge *) malloc( NE * sizeof(edge));
  for (long i = 0; i < NE; i++) {
    fscanf(file, "%ld %ld %lf", &Si, &Ti, &weight);
    Si--; Ti--;            // One-based indexing
    //weight = fabs(weight); //Make it positive    : Leave it as is
    edgeListTmp[i].head = Si;    //The S index = Row
    edgeListTmp[i].tail = Ti;    //The T index = Col
    edgeListTmp[i].weight = weight; //The value
    rowPtr[Ti+1]++;
  }
  fclose(file); //Close the file
  printf("Done reading from file.\n");
   
  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = timer();
  for (long i=0; i<NT; i++) {
    rowPtr[i+1] += rowPtr[i]; //Prefix Sum
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = timer();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: |E| = %ld, rowPtr[NV]= %ld\n", NE, rowPtr[NT]);
  
  /*---------------------------------------------------------------------*/
  /* Allocate memory for G & Build it                                    */
  /*---------------------------------------------------------------------*/    
  time1 = timer();
  //Every edge stored ONLY ONCE!
  long *rowIndex   = (long*)    malloc (NE * sizeof(long)); 
  double *weights = (double*) malloc (NE * sizeof(double));
  time2 = timer();
  printf("Time for allocating memory for marks and edgeList = %lf\n", time2 - time1);
  
  time1 = timer();
  //Keep track of how many edges have been added for a vertex:
  long *added    = (long *)  malloc( NT  * sizeof(long));
#pragma omp parallel for
  for (long i = 0; i < NT; i++) 
    added[i] = 0;
  
  printf("Building edgeList...\n");
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    long head      = edgeListTmp[i].head;   //row id
    long tail      = edgeListTmp[i].tail;   //col id
    double weight = edgeListTmp[i].weight; //weight
    long Where     = rowPtr[tail] + __sync_fetch_and_add(&added[tail], 1);
    rowIndex[Where] = head;  //Add the row id
    weights[Where]  = weight;
  }
  time2 = timer();
  printf("Time for building edgeList = %lf\n", time2 - time1);

  M->nRows    = NS;
  M->nCols    = NT;
  M->nNNZ     = NE;
  M->RowPtr   = rowPtr;
  M->RowInd   = rowIndex;
  M->Weights  = weights;    
  
  free(edgeListTmp);
  free(added);
}

/*-------------------------------------------------------*
 * This function reads a file and builds the vector y
 * Comments are in Matrix-Market style
 *-------------------------------------------------------*/
void parse_YVector(int *Y, long sizeY, char *fileName) {
  printf("Parsing for vector Y..\n");
  int nthreads;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  printf("Coloring Rouinte: Number of threads: %d\n", nthreads);
  double time1, time2;
  
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  
  /* Parse all comments starting with '%' symbol */
  char line[1024];
  do {
    fgets(line, 1024, file);
  } while ( line[0] == '%' );
  
  long numY=0;
  if (sscanf(line, "%ld",&numY ) != 1) {
    printf("Read Y: bad file format");
    exit(1);
  }
  printf("|Y|= %ld\n", numY);
  
  /*---------------------------------------------------------------------*/
  /* Read Y list  (row_id, y_value)                                      */
  /*---------------------------------------------------------------------*/
  printf("Reading the Y values...\n");
  time1 = timer();
  long row_id;
  int y_value;
  for (long i = 0; i < numY; i++) {      
    fscanf(file, "%ld %d", &row_id, &y_value);
    //printf("%d - %d\n", row_id, y_value);
    Y[row_id-1] = y_value; // One-based indexing
  }
  time2 = timer();    
  printf("Done reading from file. It took %lf seconds.\n", time2-time1);

  fclose(file); //Close the file
}//End of parse_YVector()

/*-------------------------------------------------------*
 * This function reads a MATRIX MARKET file and build the graph
 * graph is nonbipartite: each diagonal entry is a vertex, and 
 * each non-diagonal entry becomes an edge. Assume structural and 
 * numerical symmetry.
 *-------------------------------------------------------*/
void parse_MatrixMarket_Sym_AsGraph(graph * G, char *fileName) {
  printf("Parsing a Matrix Market File as a general graph...\n");
  int nthreads = 0;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  printf("parse_MatrixMarket: Number of threads: %d\n ", nthreads);

  double time1, time2;
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  /* -----      Read File in Matrix Market Format     ------ */
  //Parse the first line:
  char line[1024];
  fgets(line, 1024, file);
  char  LS1[25], LS2[25], LS3[25], LS4[25], LS5[25];    
  if (sscanf(line, "%s %s %s %s %s", LS1, LS2, LS3, LS4, LS5) != 5) {
    printf("parse_MatrixMarket(): bad file format - 01");
    exit(1);
  }
  printf("%s %s %s %s %s\n", LS1, LS2, LS3, LS4, LS5);
  if ( strcmp(LS1,"%%MatrixMarket") != 0 ) {
    printf("Error: The first line should start with %%MatrixMarket word \n");
    exit(1);
  }
  if ( !( strcmp(LS2,"matrix")==0 || strcmp(LS2,"Matrix")==0 || strcmp(LS2,"MATRIX")==0 ) ) {
    printf("Error: The Object should be matrix or Matrix or MATRIX \n");
    exit(1);
  }
  if ( !( strcmp(LS3,"coordinate")==0 || strcmp(LS3,"Coordinate")==0 || strcmp(LS3,"COORDINATE")==0) ) {
    printf("Error: The Object should be coordinate or Coordinate or COORDINATE \n");
    exit(1);
  }
  int isComplex = 0;    
  if ( strcmp(LS4,"complex")==0 || strcmp(LS4,"Complex")==0 || strcmp(LS4,"COMPLEX")==0 ) {
    isComplex = 1;
    printf("Warning: Will only read the real part. \n");
  }
  int isPattern = 0;
  if ( strcmp(LS4,"pattern")==0 || strcmp(LS4,"Pattern")==0 || strcmp(LS4,"PATTERN")==0 ) {
    isPattern = 1;
    printf("Note: Matrix type is Pattern. Will set all weights to 1.\n");
    //exit(1);
  }
  int isSymmetric = 0, isGeneral = 0;
  if ( strcmp(LS5,"general")==0 || strcmp(LS5,"General")==0 || strcmp(LS5,"GENERAL")==0 )
    isGeneral = 1;
  else {
    if ( strcmp(LS5,"symmetric")==0 || strcmp(LS5,"Symmetric")==0 || strcmp(LS5,"SYMMETRIC")==0 ) {
      isSymmetric = 1;
      printf("Note: Matrix type is Symmetric: Converting it into General type. \n");
    }
  }	
  if ( isSymmetric==0 ) 	  {
    printf("Warning: Matrix type should be Symmetric for this routine. \n");
    exit(1);
  }
  
  /* Parse all comments starting with '%' symbol */
  do {
    fgets(line, 1024, file);
  } while ( line[0] == '%' );
  
  /* Read the matrix parameters */
  long NS=0, NT=0, NV = 0;
  long NE=0;
  if (sscanf(line, "%ld %ld %ld",&NS, &NT, &NE ) != 3) {
    printf("parse_MatrixMarket(): bad file format - 02");
    exit(1);
  }
  NV = NS;
  printf("|S|= %ld, |T|= %ld, |E|= %ld \n", NS, NT, NE);

  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /* S vertices: 0 to NS-1                                               */
  /* T vertices: NS to NS+NT-1                                           */
  /*---------------------------------------------------------------------*/
  //Allocate for Edge Pointer and keep track of degree for each vertex
  long *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
#pragma omp parallel for
  for (long i=0; i <= NV; i++)
    edgeListPtr[i] = 0; //For first touch purposes
  
  edge *edgeListTmp; //Read the data in a temporary list
  long newNNZ = 0;    //New edges because of symmetric matrices
  long Si, Ti;
  double weight = 1;
  printf("Matrix is of type: Symmetric Real or Complex\n");
  printf("Weights will be converted to positive numbers.\n");
  edgeListTmp = (edge *) malloc(2 * NE * sizeof(edge));
  for (long i = 0; i < NE; i++) {
    if (isPattern == 1)
      fscanf(file, "%ld %ld", &Si, &Ti);
    else
      fscanf(file, "%ld %ld %lf", &Si, &Ti, &weight);
    Si--; Ti--;            // One-based indexing
    weight = fabs(weight); //Make it positive  : Leave it as is
    if ( Si == Ti ) {
      //Do nothing...
    }
    else { //an off diagonal element: store the edge
      //LOWER PART:
      edgeListTmp[newNNZ].head = Si;       //The S index 
      edgeListTmp[newNNZ].tail = Ti;       //The T index 
      edgeListTmp[newNNZ].weight = weight; //The value
      edgeListPtr[Si+1]++;
      edgeListPtr[Ti+1]++;
      newNNZ++;
    }//End of Else
  }//End of for loop
  fclose(file); //Close the file
  //newNNZ = newNNZ / 2;
  printf("Done reading from file.\n");
  printf("Modified the number of edges from %ld ", NE);
  NE = newNNZ; //#NNZ might change
  printf("to %ld \n", NE);

  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = timer();
  for (long i=0; i<=NV; i++) {
    edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = timer();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: 2|E| = %ld, edgeListPtr[NV]= %ld\n", NE*2, edgeListPtr[NV]);

  /*---------------------------------------------------------------------*/
  /* Allocate memory for G & Build it                                    */
  /*---------------------------------------------------------------------*/
  time1 = timer();
  edge *edgeList = (edge *) malloc( 2*NE * sizeof(edge)); //Every edge stored twice
  if ( edgeList== NULL) {
    printf("Not enough memory for added\n");
    exit(1);
  }
  time2 = timer();
  printf("Time for allocating memory for edgeList = %lf\n", time2 - time1);

  time1 = timer();
  //Keep track of how many edges have been added for a vertex:
  long  *Counter = (long *) malloc (NV  * sizeof(long));
  if (Counter == NULL) {
    printf("Not enough memory for Counter\n");
    exit(1);
  }

#pragma omp parallel for
  for (long i = 0; i < NV; i++) {
    Counter[i] = 0;
  }

  printf("About to build edgeList...\n");
  //Build the edgeList from edgeListTmp:
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    long head     = edgeListTmp[i].head;
    long tail     = edgeListTmp[i].tail;
    double weight = edgeListTmp[i].weight;
    //printf("%ld  %ld  %lf\n", head, tail, weight);

    long Where    = edgeListPtr[head] + __sync_fetch_and_add(&Counter[head], 1);
    edgeList[Where].head = head;
    edgeList[Where].tail = tail;
    edgeList[Where].weight = weight;

    //Now add the edge the other way:
    Where                  = edgeListPtr[tail] + __sync_fetch_and_add(&Counter[tail], 1);
    edgeList[Where].head   = tail;
    edgeList[Where].tail   = head;
    edgeList[Where].weight = weight;
  }
  time2 = timer();
  printf("Time for building edgeList = %lf\n", time2 - time1);

  printf("About to set pointers to graph data structure:\n");
  G->sVertices    = NV;
  G->numVertices  = NV;
  G->numEdges     = NE;
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;

  printf("Freeing up memory:\n");

  //free(edgeListTmp);
  //free(Counter);
  printf("Getting out of parse_MatrixMarket_Sym_AsGraph()\n");

}//End of parse_MatrixMarket_Sym_AsGraph()

/* 
-------------------------------------------------------------------------
INPUT FORMAT FOR WMATCH:
-------------------------------------------------------------------------
   Graph I/O is performed by a generic graph library package, 
   so some of the fields are ignored by the "wmatch" code (but 
   you must include dummy fields in the input). 

   There are three types of lines: the first line, vertex lines, 
   and edge lines. The fields in each line type are as follows. 

   First line-> size edges U
      size: integer giving number of vertices
      edges: integer giving number of edges 
      U: character ``U'' or ``u'' specifying an undirected graph

   Vertex lines->  degree vlabel xcoord ycoord
      degree: edge degree of the vertex
      vlabel: vertex label (ignored--vertices are referred to by index)
      xcoord: integer x-coordinate location (ignored)
      ycoord: integer y-coordinate location (ignored) 

      *****Each vertex line is followed immediately by the lines 
      for all its adjacent edges (thus each edge appears twice, 
      once for each vertex).******

   Edge lines-> adjacent  weight
      adjacent: index (not vlabel) of the adjacent vertex
      weight: integer edge weight 
-------------------------------------------------------------------------
*/
void parse_Dimacs1Format(graph * G, char *fileName) {
  printf("Parsing a DIMACS-1 formatted file as a general graph...\n");
  int nthreads = 0;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  printf("parse_Dimacs1Format: Number of threads: %d\n ", nthreads);
  
  double time1, time2;
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  /* -----      Read File in Matrix Market Format     ------ */  
  /* Read the matrix parameters */
  long NV = 0, NE=0;
  char line[1024], LS1[25];
  fgets(line, 1024, file);
  //Parse the first line:
  if (sscanf(line, "%ld %ld %s",&NV, &NE, LS1) != 3) {
    printf("parse_Dimacs1(): bad file format - 01");
    exit(1);
  }
  assert((LS1[0] == 'U')||(LS1[0] == 'u')); //graph is undirected
  //NE = NE / 2; //Each edge is stored twice, but counted only once, in the file
  printf("|V|= %ld, |E|= %ld \n", NV, NE);
  
  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /* S vertices: 0 to NS-1                                               */
  /* T vertices: NS to NS+NT-1                                           */
  /*---------------------------------------------------------------------*/
  //Allocate for Edge Pointer and keep track of degree for each vertex
  time1 = timer();
  long *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
  assert(edgeListPtr != NULL);
  edge *edgeList = (edge *) malloc( 2*NE * sizeof(edge)); //Every edge stored twice
  assert( edgeList != NULL);
  time2 = timer();
  printf("Time for allocating memory for storing graph = %lf\n", time2 - time1);
#pragma omp parallel for
  for (long i=0; i <= NV; i++)
    edgeListPtr[i] = 0; //For first touch purposes
  
  long Degree, Ti;
  double weight = 1;
  int Twt=0, label, xCoord, yCoord;
  long nE = 0;
  printf("Weights will be converted to positive integers.\n");
  
  time1 = timer();
  for (long i = 0; i < NV; i++) {
    //Vertex lines:  degree vlabel xcoord ycoord
    fscanf(file, "%ld %d %d %d", &Degree, &label, &xCoord, &yCoord);
    edgeListPtr[i+1] = Degree;
    for (long j=0; j<Degree; j++) {
      fscanf(file, "%ld %d", &Ti, &Twt);
      edgeList[nE].head   = i;       //The S index 
      edgeList[nE].tail   = Ti-1;    //The T index: One-based indexing
      edgeList[nE].weight = fabs((double)Twt); //Make it positive and cast to Double      
      nE++;
    }//End of inner for loop
  }//End of outer for loop
  fclose(file); //Close the file
  time2 = timer(); 
  printf("Done reading from file: nE= %ld. Time= %lf\n", nE, time2-time1);
  assert(NE == nE/2);
  
  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = timer();
  for (long i=0; i<=NV; i++) {
    edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = timer();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: 2|E| = %ld, edgeListPtr[NV]= %ld\n", NE*2, edgeListPtr[NV]);

  G->sVertices    = NV;
  G->numVertices  = NV;
  G->numEdges     = NE;
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;

}//End of parse_Dimacs1Format()
