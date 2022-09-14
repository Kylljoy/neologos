#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "network.h"
#include "dataset.h"
#define DATASET_SIZE (150)
#define STOCHASTIC_FORKS (1)
#define DELTA (0.0001)
#include <assert.h>
#include <time.h>
#include <unistd.h>

dataset read_dataset() {
  dataset out = generate_dataset(DATASET_SIZE);
  FILE *fp = fopen("./iris.data", "r");
  assert(fp);
  double a = 0.0;
  for (int i = 0; i < DATASET_SIZE; i++) {
    out[i].data = init_matrix(4,1);
    out[i].class = init_matrix(3,1);

    fscanf(fp, "%lf,%lf,%lf,%lf;%lf,%lf,%lf\n", out[i].data->values, out[i].data->values + 1, out[i].data->values + 2, out[i].data->values + 3, out[i].class->values, out[i].class->values + 1, out[i].class->values + 2);
    normalize_matrix(out[i].data);
  }
  fclose(fp);
  return out;
}

int main() {
    //double delt = 0.5;
    int fds[2], status;
    status = 1;
    fds[0] = -1;
    fds[1] = -1;
    
    int forks_remaining = STOCHASTIC_FORKS;
    int read_end, write_end;
    //Create the shared dataset
    dataset d = read_dataset();


    //Create N + 1 processes
    while (forks_remaining > 1) {
      read_end = fds[0]; //Child takes read
      if (pipe(fds) > 0) {
        return 1;
      }
      //printf("r/w %d << %d\n", fds[0], fds[1]);
      
      status = fork();
      if (status < 0) {
        return 1;
      }
      if (status == 0) {
        //Parent takes write
        //printf("%d assigned write\n", forks_remaining);
        write_end = fds[1];
        break;
      }
      forks_remaining--;
      //sleep(0.1);
    
    }

    if (status != 0) {
      read_end = fds[0];
      write_end = -1;
    }
    
    
    //srand(time(0));
    //printf("Process %d; Read %d, Write %d\n", forks_remaining, read_end, write_end);
    

    network *n = generate_network(3, (int *) &((int[]){4, 30, 3}));
    
    populate_network(n);
    propagate_recursive(n->output);
    
  //  print_network(n);
    double average_loss = 0.0;
    for (int i = 1; i <= 200000; i++) {
      datum case_ = d[random() % DATASET_SIZE];
      copy_matrix_into(case_.data,n->input->output);
      propagate_recursive(n->output);
    //  print_matrix(n->output->output);
      backpropagate(n, case_.class, DELTA);

    }

    double successes = 0.0;
    for (int i = 0; i < 100; i++) {
      datum case_ = d[random() % DATASET_SIZE];
      copy_matrix_into(case_.data,n->input->output);
      propagate_recursive(n->output);
      if ((n->output->output->values[0] > n->output->output->values[1])
          && (n->output->output->values[0] > n->output->output->values[2])) {
            if (case_.class->values[0] > 0.5) {
              successes++;
            } else {
        //      successes--;
            }
      } else if ((n->output->output->values[1] > n->output->output->values[0])
          && (n->output->output->values[1] > n->output->output->values[2])) {
            if (case_.class->values[1] > 0.5) {
              successes++;
            } else {
          //    successes--;
            }
      } else {
        if (case_.class->values[2] > 0.5) {
          successes++;
        } else {
        //  successes--;
        }
      }
    //  print_matrix(composite->output->output);
     
    }

    successes /= 100.0;
    //printf("%lf\n", successes);

    network *composite = n;
    if (read_end != -1) {
      //Read the data
      FILE *read_file = fdopen(read_end, "r");
      double incoming_accuracy = 0.0;
      fread(&incoming_accuracy, sizeof(double), 1, read_file);
      double ratio = 0.0;
      
      if (incoming_accuracy == 0) {
        ratio = 1.0;
      } else {
        ratio = (successes / incoming_accuracy) / ((successes / incoming_accuracy) + 1.0);
      }
      //printf("Accepting parallel network...\n");
      network *incoming = read_network(read_file);
      //printf("Compositing networks...\n");

      composite = merge_networks(n, incoming, ratio);
      sleep(0.01);
      printf("process %d, and folded at %lf\n", forks_remaining, ratio);
      fclose(read_file);
    }


    //printf("Process %d complete\n", forks_remaining);
    if (write_end != -1) {
      //Send to the next member
      FILE *write_file = fdopen(write_end, "w");
      fwrite(&successes, sizeof(double), 1, write_file);
      write_network(composite, write_file);
      free_network(composite);
      fclose(write_file);
      //printf("Process %d (Loss: %lf) merging into ", forks_remaining, average_loss);
      //Cave Johnson, We're Done Here
      return 0;
    }
    average_loss = 0;
    printf("Process %d proceeding\n\n", forks_remaining);
    //print_network(composite);
    //Check the final network
    
    successes = 0.0;
    for (int i = 0; i < 100; i++) {
      datum case_ = d[random() % DATASET_SIZE];
      copy_matrix_into(case_.data,composite->input->output);
      propagate_recursive(composite->output);
      if ((composite->output->output->values[0] > composite->output->output->values[1])
          && (composite->output->output->values[0] > composite->output->output->values[2])) {
            if (case_.class->values[0] > 0.5) {
              successes++;
            } else {
        //      successes--;
            }
      } else if ((composite->output->output->values[1] > composite->output->output->values[0])
          && (composite->output->output->values[1] > composite->output->output->values[2])) {
            if (case_.class->values[1] > 0.5) {
              successes++;
            } else {
          //    successes--;
            }
      } else {
        if (case_.class->values[2] > 0.5) {
          successes++;
        } else {
        //  successes--;
        }
      }
    //  print_matrix(composite->output->output);
      average_loss += backpropagate(composite, case_.class, 0.0);
    }

    printf("Average loss: %f; Accuracy: %f\n", average_loss / 100.0, successes / 100.0);
    
    
    return 0;

}
