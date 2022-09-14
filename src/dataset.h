#ifndef DATASET_H
#define DATASET_H



typedef struct datum_t {
  matrix *data;

  matrix *class;
} datum;

typedef datum * dataset;

dataset generate_dataset(int size);

#endif
