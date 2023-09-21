#include "defs.h"
#include "sprng.h"

/* Generate a list of random numbers */
void prand(int howMany, double *randList) {
  
  int *stream, seed;
  /* Initialize RNG stream */
  seed = 786;
  stream = init_sprng(0, 0, 1, seed, SPRNG_DEFAULT);
  
  /* Generate random numbers */
  //#pragma omp parallel for
  for (int i=0; i<howMany; i++) {
    randList[i] = sprng(stream);
  }
  
  free_sprng(stream);
}
