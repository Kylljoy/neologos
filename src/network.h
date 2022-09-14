#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"
#include <math.h>

typedef struct layer_t {
    matrix *weights;
    matrix *activations;
    matrix *biases;
    matrix *output;

    unsigned short int size;

    struct layer_t *prev;
} layer;


typedef struct network_t {
    layer *input;
    layer *output;

    unsigned int length;
} network;

layer *generate_layer(int size, layer *prev);

network *generate_network(unsigned int length, int *sizes);

void print_network(network *n);

void propagate(network *n);

void propagate_recursive(layer *l);

double backpropagate_recursive(layer *l, matrix *error, double delt);

double backpropagate(network *n, matrix *error, double delt);

void populate_network(network *n);

void free_layer(layer *l);

void free_network(network *n);

network *merge_networks(network *a, network *b, double favor);

network *read_network(FILE *s);

void write_network(network *n, FILE *s);

#endif
