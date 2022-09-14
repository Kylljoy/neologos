#ifndef MATRIX_H
#define MATRIX_H
typedef struct mat_t {
    unsigned int rows;
    unsigned int cols;
    double *values;
} matrix;

matrix *init_matrix(unsigned int rows, unsigned int cols);

matrix *add_matrix(matrix *a, matrix *b);

void add_matrix_d(matrix *a, matrix *b, matrix *dest);

matrix *subtract_matrix(matrix *a, matrix *b);

matrix *subtract_matrix_d(matrix *a, matrix *b, matrix *dest);

matrix *multiply_matrix(matrix *a, matrix *b);

void multiply_matrix_d(matrix *a, matrix *b, matrix *dest);

matrix *scalar_multiply_matrix(matrix *a, double scalar);

void print_matrix(matrix *a);

matrix *map_matrix(matrix *a, double (*fun)(double));

void map_matrix_d(matrix *a, double (*fun)(double), matrix *dest);

void assign_value_matrix(matrix *a, unsigned int row, unsigned int col, double val);

matrix *copy_matrix(matrix *a);

void copy_matrix_into(matrix *a, matrix *b);

double dp_matrix(matrix *a);

void free_matrix(matrix *a);

void scale_matrix(matrix *a, double scalar);

void normalize_matrix(matrix *a);

matrix *hadamard_matrix(matrix *a, matrix *b);

matrix *transpose_matrix(matrix *a);

matrix *average_matrix(matrix *a, matrix *b, double favor);
#endif
