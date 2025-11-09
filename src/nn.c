#include <stdio.h>
#include <stdlib.h>

#include "numstruct.h"

/*
    NOTE: For this example we assume XOR training dataset as fixed, in the future we will make this variable

    A static 2D array representing the XOR truth table. Each row contains {x1, x2, y}.
    It uses float rather than int. That’s fine if you want float-based neural net inputs/targets.
*/
float TRAIN_DATA[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}};

// Macro to compute number of training rows at compile time.
#define TRAIN_COUNT (sizeof(TRAIN_DATA) / sizeof(TRAIN_DATA[0]))

// -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Neural Network structure.
typedef struct {
    size_t *layer;      // Number of neurons per layer, including input
    size_t layer_count; // Number of layers

    Matrix *weight;     // These are layer_count - 1
    Vector *bias;       // These are layer_count - 1
    Vector *activation; // These are layer_count, activation holds activation (output) vectors for each layer;
                        // activation[0] is the input layer, up to activation[layer_count-1].
} Network;

/*
    Note these functions take Network n by value. Since Network contains pointers, passing by value copies the struct (shallow copy of pointers).
    That is okay for read-only or for in-place modification of pointed memory, but could be misleading.
    Consider using Network *n to make ownership and mutability explicit.
    --> Using directly the struct instead of a pointer because the structure itself is lightweight (8*5 = 40 byte)
*/
Network network_alloc(size_t *layer, size_t layer_count);
void network_fill_rand(Network n, float high, float low);
Vector network_forward(Network n, Vector input);
float network_cost(Network n); // Assume cost with respect to XOR
void network_learn(Network n, float epsilon, float learning_rate);

void network_print(Network n);

// -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Allocates arrays for weight, bias, and activation.
Network network_alloc(size_t *layer, size_t layer_count) {
    Network n = {0};

    n.layer = layer; // NOTE: assume layer remain allocated throughout the lifetime of the program
    n.layer_count = layer_count;

    n.weight = malloc(sizeof(Matrix) * (layer_count - 1));
    n.bias = malloc(sizeof(Vector) * (layer_count - 1));
    n.activation = malloc(sizeof(Vector) * layer_count);

    n.activation[0] = vector_alloc(layer[0]);
    for (size_t i = 1; i < n.layer_count; i++) {
        n.weight[i - 1] = matrix_alloc(n.activation[i - 1].cols, layer[i]);
        n.bias[i - 1] = vector_alloc(layer[i]);
        n.activation[i] = vector_alloc(layer[i]);
    }

    return n;
}

// Initializes network values with random values. Note that activation is not part of th network parameters, thus it is used to store partial computations.
void network_fill_rand(Network n, float low, float high) {
    for (size_t i = 0; i < n.layer_count - 1; i++) {
        matrix_fill_rand(n.weight[i], low, high);
        vector_fill_rand(n.bias[i], low, high);
    }
}

// Print Neural Network parametrs
void network_print(Network n) {
    for (size_t i = 0; i < n.layer_count - 1; i++) {
        printf("Layer n. %d\n\n", (int)(i + 1));
        printf("  Weight matrix:\n");
        matrix_print(n.weight[i]);
        printf("\n\n");
        printf("  Bias vector:\n");
        vector_print(n.bias[i]);
        printf("\n\n");
    }
}

// Vector network_forward(Network n, Vector input) {
// }

// float network_cost(Network n) {
// }

// void network_learn(Network n, float epsilon, float learning_rate) {
// }

// -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

int main(void) {
    printf("Hello World!\n\n");

    /* XOR network with
     - input layer of 2 neurons
     - hidden layer of 2 neurons
     - output layer of 1 neuron
    */
    size_t xor_layer[] = {2, 2, 1};
    size_t xor_layer_count = sizeof(xor_layer) / sizeof(size_t);

    printf("xor_layer_count: %zu\n\n", xor_layer_count);

    Network nn = network_alloc(xor_layer, xor_layer_count);
    network_fill_rand(nn, 0.0f, 1.0f);
    network_print(nn);

    return 0;
}
