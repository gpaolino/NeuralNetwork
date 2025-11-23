#include <stdio.h>
#include <stdlib.h>

#include "numstruct.h"
#include "utils.h"

float** TRAIN_DATA;
size_t TRAIN_COUNT;

/*  NOTE: For this example we assume XOR training dataset as fixed, in the future we will make this variable

    A static 2D array representing the XOR truth table. Each row contains {x1, x2, y}.
    It uses float rather than int. That’s fine if you want float-based neural net inputs/targets.
*/
/*	XOR truth table:
char FNC_NAME[] = "XOR";
float TRAIN_DATA[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}};
*/

/*  TODO: extend the framework in order to manage large matrix as train data input
    char FNC_NAME[] = "PokerHand";
    float** TRAIN_DATA = load_train_data("data/poker+hand+normalized/test3.txt", 1000, 94);
*/

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

float** load_train_data(const char* filename, size_t* rows, size_t cols);

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
        n.weight[i - 1] = matrix_alloc(n.activation[i - 1].len, layer[i]);
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

// Print Neural Network parameters.
void network_print(Network n) {
    for (size_t i = 0; i < n.layer_count - 1; i++) {
        printf("==Layer n. %d\n\n", (int)(i + 1));
        printf("  Weight matrix:\n");
        matrix_print(n.weight[i]);
        printf("\n\n");
        printf("  Bias vector:\n");
        vector_print(n.bias[i]);
        printf("\n\n");
    }
}

// Load train dataset from file.
float** load_train_data(const char* filename, size_t* rows, size_t cols) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "[ERROR] - Could not open file %s in load_train_data()\n", filename);
        exit(1);
    }

    // Count the rows
    size_t count = 0;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), fp)) {
        count++;
    }

    *rows = count;

    // Back to the beginning of file
    fseek(fp, 0, SEEK_SET);

    // Dynamically allocate the table
    float** table = malloc(count * sizeof(float*));
    for (size_t i = 0; i < count; i++) {
        table[i] = malloc(cols * sizeof(float));
        for (size_t j = 0; j < cols; j++) {
            fscanf(fp, "%f", &table[i][j]);
        }
    }

    /* Test print
    printf("Loaded %zu rows:\n", *rows);

    for (size_t i = 0; i < *rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            printf("%f ", table[i][j]);
        }
        printf("\n");
    }
    */

    fclose(fp);
    return table;
}

// Feed an input into the network and compute an output.
Vector network_forward(Network n, Vector input) {
    // First, set the input into the network as the first input layer
    matrix_copy(vector_as_matrix(n.activation[0]), vector_as_matrix(input));

    // Iterate across all the network layers
    for (size_t i = 0; i < n.layer_count - 1; i++) {
        // NN core operations: matrix multiplication, bias vector sum, then apply the activation function.

        /*
            First multiply the previous activation vector with the
            weight matrix of the current layer. If this is the first hidden
            layer, the activation vector will be the input vector.
        */
        matrix_mult(vector_as_matrix(n.activation[i + 1]),
                    vector_as_matrix(n.activation[i]), n.weight[i]);

        /*
            Then, sum the bias vector of the current layer to the
            result of the multiplication.
        */
        matrix_sum(vector_as_matrix(n.activation[i + 1]),
                   vector_as_matrix(n.bias[i]));

        /*
            Finally, apply the activation function to prepare for the
            next iteration. This is where the network becomes non-linear.
        */
        matrix_apply(vector_as_matrix(n.activation[i + 1]), sigmoid);
    }


    return n.activation[n.layer_count - 1];
}

// Compute the cost of the current parameters choice.
float network_cost(Network n) {
    float cost = 0.0f;

    // TODO: parametrize the input construction
    Vector input = vector_alloc(2);
    Vector output;
    
    for (size_t i = 0; i < TRAIN_COUNT; i++) {
        input.data[0] = TRAIN_DATA[i][0];
        input.data[1] = TRAIN_DATA[i][1];
        output = network_forward(n, input);

        float y_expected = TRAIN_DATA[i][2];
        float y_obtained = output.data[0];

        float d = y_obtained - y_expected;

        /*
            Square to make sure that the cost function will have a
            continuous partial derivative. This is needed to implement
            gradient descent as a learning algorithm.
        */
       cost += d*d;
    }

    // Return the average cost
    cost /= TRAIN_COUNT;
    return cost;
}

// Uses finite differences to compute the gradient. Can improve it using backpropagation.
void network_learn(Network n, float epsilon, float learning_rate) {
    float base_cost = network_cost(n);

    for (size_t l = 0; l < n.layer_count - 1; l++) {

        // Finite differences on weights
        for (size_t row = 0; row < n.weight[l].rows; row++) {
            for (size_t col = 0; col < n.weight[l].cols; col++) {
                // Save original
                float original = MATRIX_AT(n.weight[l], row, col);

                // Perturb parameter
                MATRIX_AT(n.weight[l], row, col) += epsilon;

                // Compute new cost
                float new_cost = network_cost(n);

                // Estimate gradient
                float gradient = (new_cost - base_cost) / epsilon;

                // Update parameter through gradient descent
                MATRIX_AT(n.weight[l], row, col) = original - learning_rate * gradient;
            }
        }
        

        // Finite differences on bias
        for (size_t col = 0; col < n.bias[l].len; col++) {
            // Save original
            float original = n.bias[l].data[col];

            // Perturb parameter
            n.bias[l].data[col] += epsilon;

            // Compute new cost
            float new_cost = network_cost(n);

            // Estimate gradient
            float gradient = (new_cost - base_cost) / epsilon;

            // Update parameter through gradient descent
            n.bias[l].data[col] = original - learning_rate * gradient;
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

int main(void) {
    char FNC_NAME[] = "XOR";
	printf("======= %s =======\n", FNC_NAME);

    size_t rows;
    const size_t cols = 3;
    TRAIN_DATA = load_train_data("data/logic+gates/xor_truth_table.txt", &rows, cols);
    TRAIN_COUNT = rows;
	
    // FNC network with
    // - input layer of 2 neurons
    // - hidden layer of 3 neurons
    // - output layer of 1 neuron    
    size_t fnc_layer[] = {2, 3, 1};
    size_t fnc_layer_count = sizeof(fnc_layer) / sizeof(size_t);
    printf("fnc_layer_count: %zu\n\n", fnc_layer_count);

    Network nn = network_alloc(fnc_layer, fnc_layer_count);
    network_fill_rand(nn, 0.0f, 1.0f);

    float cost = network_cost(nn);
    printf("Original cost: %f\n", cost);
    printf("Training in progress...\n");

    float learning_rate = 1e-2;
    float epsilon = 1e-3;
    int epochs = 500 * 500;
    for (int i = 0; i < epochs; i++) {
        network_learn(nn, epsilon, learning_rate);
    }

    cost = network_cost(nn);
    printf("Cost after training: %f\n\n", cost);
    
	network_print(nn);

    printf("--------------------------------\n");

    for (size_t i = 0; i < TRAIN_COUNT; i++) {
        Vector input = vector_alloc(2);
        input.data[0] = TRAIN_DATA[i][0];
        input.data[1] = TRAIN_DATA[i][1];
        Vector output = network_forward(nn, input);
        printf("%f %s %f = %f\n", input.data[0], FNC_NAME, input.data[1], output.data[0]);
    }

    return 0;
}
