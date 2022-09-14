#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "dataset.h"
#include <assert.h>

dataset generate_dataset(int size) {
  dataset out = malloc(size * sizeof(datum));
  assert(out);
  return out;
}
