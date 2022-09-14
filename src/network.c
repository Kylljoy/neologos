/*
Neural Network
*/
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "network.h"

layer *generate_layer(int size, layer *prev) {
    layer *output = malloc(sizeof(layer));
    assert(output);
    output->output = init_matrix(size, 1);
    output->size = size;
    output->prev = NULL;
    if(prev) {
        output->activations = init_matrix(size, 1);
        output->weights = init_matrix(size, prev->size);
        output->prev = prev;
        output->biases = init_matrix(size, 1);
    } else {
        output->weights = NULL;
        output->biases = NULL;
    }
    return output;
}

double sigismoid(double value) {
  return 1.0 / (1.0 + exp(0.0 - value));
}


double sig_prime(double value) {
  return (0.0 - exp(0.0 - value)) / pow((1.0 + exp(0.0 - value)), 2);

}

double loss(double value) {
  return pow(value, 2);
}

double loss_prime(double value) {
  return 2.0 * value;
}

network *generate_network(unsigned int length, int *sizes) {
    if(length == 0) {
        return NULL;
    }
    network *output = malloc(sizeof(network));
    assert(output);
    output->length = length;
    output->input =  generate_layer(sizes[0], NULL);
    
    layer *prev = output->input;
    for (int i = 1; i < length; i++) {
        assert(prev);
        prev = generate_layer(sizes[i], prev);
    }
    output->output = prev;
    return output;
}

double random_double(double low, double high) {
  return  ((fmod((double) random(),  10000.0)) / 10000.0) * (high - low) + low;
}

double random_double_bound(double a) {
  return random_double(-0.5,0.5);
}

void propagate_recursive(layer *l) {
  if (l->prev) {
    propagate_recursive(l->prev);
    multiply_matrix_d(l->weights, l->prev->output, l->activations);
    add_matrix_d(l->activations, l->biases, l->activations);
    map_matrix_d(l->activations, &sigismoid, l->output);
  }
}

double backpropagate(network *n, matrix *target, double delt) {
  matrix *error = subtract_matrix(target, n->output->output);
  matrix *loss_m = copy_matrix(error);
  map_matrix_d(error, &loss_prime, error);
  double out = backpropagate_recursive(n->output, error, delt);
  map_matrix_d(loss_m, &loss, loss_m);
  //double out = dp_matrix(loss_m);
  free_matrix(loss_m);
  return out;
}

double backpropagate_recursive(layer *l, matrix *error, double delt) {
  assert(error->rows == l->size);
  //Error is dL / dS, where S is the non-sigismoid sum
  matrix *unwrap_sig = map_matrix(l->activations, &sig_prime);
  matrix *error_p = hadamard_matrix(unwrap_sig, error);
  free_matrix(error);
  error = error_p;
  double total_grad = 0.0;
  free_matrix(unwrap_sig);
  if (l->prev->prev) {
    //Calculate the error of the previous activations
    //Essentially, apply dS / dS-1
    //First, apply sig prime to the activations

    //Then, undo the multiplication
    matrix *weight_transpose = transpose_matrix(l->weights);
    matrix *next_error = multiply_matrix(weight_transpose, error);
    free_matrix(weight_transpose);
    total_grad = backpropagate_recursive(l->prev, next_error, delt);

  }
  //Calculate weight gradient

  matrix *activations_transpose = transpose_matrix(l->prev->output);
  matrix *weight_gradient = multiply_matrix(error, activations_transpose);
  total_grad = dp_matrix(weight_gradient);
  normalize_matrix(weight_gradient);
  scale_matrix(weight_gradient, delt);

  

  subtract_matrix_d(l->weights, weight_gradient, l->weights);

  free_matrix(weight_gradient);
  free_matrix(activations_transpose);

  //Calculate bias gradient
  matrix *bias_gradient = copy_matrix(error);
  total_grad = dp_matrix(bias_gradient);
  normalize_matrix(bias_gradient);
  scale_matrix(bias_gradient, delt * 0.1);

  
  
  subtract_matrix_d(l->biases, bias_gradient, l->biases);

  free_matrix(bias_gradient);
  free_matrix(error);
  return total_grad;
}

void print_network(network *n) {
    if(!n) {
        return;
    }
    layer *curr = n->output;
    for (int i = n->length; i > 0; i--) {

        printf("\n\nLAYER %d:\n\n\n", i);
        printf("OUTPUT:\n\n");
        print_matrix(curr->output);
        printf("\n\nBIASES:\n\n");
        print_matrix(curr->biases);
        printf("\n\nWEIGHTS:\n\n");
        print_matrix(curr->weights);
        curr = curr->prev;
    }
    printf("================\n");
}

void free_network(network *n) {
  free_layer(n->output);
  free(n);
}

void free_layer(layer *l) {
  if(l->prev) {
    free_layer(l->prev);
  }
  if (l->prev) {
    free_matrix(l->biases);
    free_matrix(l->weights);
    free_matrix(l->activations);
  }
  free_matrix(l->output);
  free(l);
}


layer *merge_layer_recursive_logos(layer *a, layer *b, double favor) {
  assert(a->size == b->size);
  if (a->prev) {
    layer *out = generate_layer(a->size, merge_layer_recursive_logos(a->prev, b->prev, favor));
    free_matrix(out->weights);
    free_matrix(out->biases);
    out->weights = average_matrix(a->weights, b->weights, favor);
    out->biases = average_matrix(a->biases, b->biases, favor);
    return out;
  }
  layer *out = generate_layer(a->size, NULL);
  return out;
}

layer *get_input_layer(layer *l) {
  if (!l->prev) {
    return l;
  }
  return get_input_layer(l->prev);
}

//Merge methods for stochastic systems
network *merge_networks(network *a, network *b, double favor) {
  assert(a->length == b->length);
  network *n = malloc(sizeof(network));
  assert(n);
  n->length = a->length;
  n->output = merge_layer_recursive_logos(a->output, b->output, favor);
  //printf("Layers merged...\n");
  n->input = get_input_layer(n->output);
  //rintf("Input accessed...\n");
  free_network(a);
  free_network (b);
  //printf("Layers freed...\n");
  return n;
}

void write_layer_size_recursive(layer *l, FILE *stream) {
  if (l->prev) {
     write_layer_size_recursive(l->prev, stream);
  }
  fwrite(&(l->size), sizeof(short int), 1, stream);
}

void write_network(network *n, FILE *stream) {
  //Size goes first
  assert(stream);
  fwrite(&(n->length), sizeof(unsigned int), 1, stream);
  
  //Then layer sizes. Back to front.
  write_layer_size_recursive(n->output, stream);
  //Then matrix details: weight then biases. Front to back. 
  layer *curr = n->output;
  while(curr->prev) {
    fwrite(curr->weights->values, sizeof(double), curr->size * curr->prev->size, stream);
    fwrite(curr->biases->values, sizeof(double), curr->size, stream);
    curr = curr->prev;
  }
  fflush(stream);
}

network *read_network(FILE *stream){
  assert(stream);
  network *out = malloc(sizeof(network));
  assert(out);
  //Length comes out first
  fread(&(out->length), sizeof(unsigned int), 1, stream);
  layer *l = NULL;
  int layer_size = 0;
  //Then layer sizes. Back to front.

  for (int i = 0; i < out->length; i++) {
    
    fread(&layer_size, sizeof(short int), 1, stream);
    l = generate_layer(layer_size, l);
    //printf("%d...\n", i);
  }

  out->output = l;
  //Then matrix details: weight then biases. Front to back.
  while (l->prev) {
    fread(l->weights->values, sizeof(double), l->size * l->prev->size, stream);

    fread(l->biases->values, sizeof(double), l->size, stream);

    l = l->prev;
  }
  out->input = l;;
  //Network is assembled
  return out;
}

void populate_network(network *n) {
  layer *l = n->output;
  assert(l);
  while (l->prev) {
    if (l->biases) {
      map_matrix_d(l->biases,&random_double_bound, l->biases);
    }
    if (l->weights) {
      map_matrix_d(l->weights, &random_double_bound, l->weights);
    }
    l = l->prev;
  }
}
