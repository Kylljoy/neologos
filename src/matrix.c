 /*
Attempt at writing a somewhat optimized gradient descent algorithm
in C.
*/
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct mat_t {
    unsigned int rows;
    unsigned int cols;
    double *values;
} matrix;

void print_matrix(matrix *a);

matrix *init_matrix(unsigned int rows, unsigned int cols) {
    matrix *output = malloc(sizeof(matrix));
    assert(output);
    output->rows = rows;
    output->cols = cols;
    output->values = malloc(sizeof(double) * rows * cols);
    assert(output->values);
    for (int i = 0; i < rows * cols; i++) {
      output->values[i] = 0;
    }
    return output;
}

matrix *add_matrix(matrix *a, matrix *b) {
    if (!a) {
        return NULL;
    }
    assert(a->rows == b->rows);
    assert(a->cols == b->cols);
    matrix *output = malloc(sizeof(matrix));
    output->rows = a->rows;
    output->cols = a->cols;
    register unsigned int row_size = a->cols;
    output->values = malloc(sizeof(double) * a->rows * a->cols);
    for (int i = 0; i < a->rows * a->cols; i++) {
            output->values[i] = a->values[i] + b->values[i];
    }
    return output;
}
matrix *subtract_matrix(matrix *a, matrix *b) {
  if (!a) {
    return NULL;
  }
  assert(a->rows == b->rows);
  assert(a->cols == b->cols);
  matrix *output = malloc(sizeof(matrix));
  output->rows = a->rows;
  output->cols = a->cols;
  register unsigned int row_size = a->cols;
  output->values = malloc(sizeof(double) * a->rows * a->cols);
  for (int i = 0; i < a->rows * a->cols; i++) {
    output->values[i] = a->values[i] - b->values[i];
  }
  return output;
}


void add_matrix_d(matrix *a, matrix *b, matrix *dest) {
    if (!a || !b || !dest) {
        return;
    }
    assert(a->rows == b->rows);
    assert(a->rows == dest->rows);
    assert(a->cols == dest->cols);
    assert(a->cols == b->cols);
    register unsigned int row_size = a->cols;
    for (int i = 0; i < a->rows * a->cols; i++) {
            dest->values[i] = a->values[i] + b->values[i];
    }
}
void *subtract_matrix_d(matrix *a, matrix *b, matrix *dest) {
  if (!a) {
    return NULL;
  }
  assert(a->rows == b->rows);
  assert(a->cols == b->cols);
  register unsigned int row_size = a->cols;
  for (int i = 0; i < a->rows * a->cols; i++) {
    dest->values[i] = a->values[i] - b->values[i];
  }
}


matrix *multiply_matrix(matrix *a, matrix *b) {
    if (!a) {
        return NULL;
    }
    assert(a->cols == b->rows);
    matrix *output = malloc(sizeof(matrix));
    output->rows = a->rows;
    output->cols = b->cols;
    output->values = malloc(sizeof(double) * a->rows * b->cols);
    register unsigned int a_row_size = a->cols;
    register unsigned int b_row_size = b->cols;
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            output->values[(i * b_row_size) + j] = a->values[(i * a_row_size)] * b->values[j];
            for (int k = 1; k < a->cols; k++) {
                output->values[(i * b_row_size) + j] += a->values[(i * a_row_size) + k] * b->values[(k * b_row_size) + j];
            }
        }
    }
    return output;
}

void multiply_matrix_d(matrix *a, matrix *b, matrix *dest) {
    if (!a || !b || !dest) {
        return;
    }
    assert(a->cols == b->rows);
    assert(a->rows == dest->rows);
    assert(b->cols == dest->cols);
    register unsigned int a_row_size = a->cols;
    register unsigned int b_row_size = b->cols;
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
          dest->values[(i * b_row_size) + j] = a->values[(i * a_row_size)] * b->values[j];

         for (int k = 1; k < a->cols; k++) {
              dest->values[(i * b_row_size) + j] += a->values[(i * a_row_size) + k] * b->values[(k * b_row_size) + j];
            }
        }
    }
    return;
}



matrix *scalar_multiply_matrix(matrix *a, double scalar) {
    if (!a) {
        return NULL;
    }
    register unsigned char matrix_size = a->cols * a->rows;
    matrix *output = malloc(sizeof(matrix));
    output->rows = a->rows;
    output->cols = a->cols;
    output->values = malloc(sizeof(double) * matrix_size);
    for (int i = 0; i < matrix_size; i++) {
        output->values[i] =  a->values[i] * scalar;
    }
    return output;
}

void scale_matrix(matrix *a, double scalar) {
  assert(a);
  for (int i = 0; i < a->cols * a->rows; i++) {
    a->values[i] *= scalar;
  }
}


void assign_value_matrix(matrix *a, int row, int col, double val) {
  a->values[(a->cols * row) + col] = val;
}

double get_value_matrix(matrix *a, int row, int col) {
  return a->values[(a->cols * row) + col];
}

matrix *map_matrix(matrix *a, double (*fun)(double)) {
  matrix *output = malloc(sizeof(matrix));
  output->rows = a->rows;
  output->cols = a->cols;
  output->values = malloc(sizeof(double) * output->cols * output->rows);
  for (int i = 0; i < a->rows * a->cols; i++) {
    (output->values)[i] = (*fun)(a->values[i]);
  }
  return output;
}

void map_matrix_d(matrix *a, double (*fun)(double), matrix *dest) {
  for (int i = 0; i < a->rows * a->cols; i++) {
    dest->values[i] = (*fun)(a->values[i]);
  }
}

void print_matrix(matrix *a) {
    if (!a) {
      printf("\n\t[ NULL ]\n");
      return;
    }
    printf("[%d x %d]\n", a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            printf("%5.5f  ", a->values[(a->cols * i) + j]);
        }
        printf("\n");
    }
}


matrix *copy_matrix(matrix *a) {
  matrix *output = init_matrix(a->rows, a->cols);
  for (int i = 0; i < a->rows * a->cols; i++) {
    output->values[i] = a->values[i];
  }
  return output;
}

void copy_matrix_into(matrix *a, matrix *b) {
  assert(a->rows == b->rows);
  assert(a->cols == b->cols);
  for (int i = 0; i < a->rows * a->cols; i++) {
    b->values[i] = a->values[i];
  }
}


double dp_matrix(matrix *a) {
  double out = 0.0;
  for (int i = 0; i < a->rows * a->cols; i++) {
    out += a->values[i];
  }
  return out;
}

matrix* transpose_matrix(matrix *a) {
  matrix *out = init_matrix(a->cols, a->rows);
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      out->values[(j * a->rows) + i] = a->values[(i * a->cols) + j];
    }
  }
  return out;
}

matrix *hadamard_matrix(matrix *a, matrix *b) {
  assert(a->rows == b->rows);
  assert(a->cols == b->cols);

  matrix *out = init_matrix(a->rows, a->cols);
  for (int i = 0; i < a->rows * a->cols; i++) {
      out->values[i] = a->values[i] * b->values[i];
  }
  return out;
}

void normalize_matrix(matrix *a) {
  double magnitude = 0;
  for (int i = 0; i < a->cols * a->rows; i++) {
    magnitude += pow(a->values[i], 2);
  }
  magnitude = pow(magnitude, 0.5);
  if (abs(magnitude) <= 0.0000001) {
    return;
  }
  for (int i = 0; i < a->cols * a->rows; i++) {
    a->values[i] /= magnitude;
  }
}

matrix *average_matrix(matrix *a, matrix *b, double favor) {
  assert(a->rows == b->rows);
  assert(a->cols == b->cols);
  matrix *out = init_matrix(a->rows, a->cols);
  for (int i = 0; i < a->rows * a->cols; i++) {
    out->values[i] = (a->values[i] * favor)+ ((1.0 - favor) * b->values[i]);
  }
  return out;
} 


void free_matrix (matrix *a) {
  free(a->values);
  free(a);
}
