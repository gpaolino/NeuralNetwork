#include <stdio.h>

#include "numstruct.h"
#include "utils.h"

// Allocate and zero-initialize a Matrix.
Matrix matrix_alloc(size_t rows, size_t cols) {
    Matrix m = {0};
    m.rows = rows;
    m.cols = cols;

    m.data = malloc(sizeof(float) * rows * cols);
    if (m.data == NULL) {
        fprintf(stderr, "[ERROR] - Could not alloc in matrix_alloc()\n");
        exit(1);
    }

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            MATRIX_AT(m, i, j) = 0.0f;
        }
    }

    return m;
}

// Fill a matrix with random values between low and high.
void matrix_fill_rand(Matrix m, float low, float high) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MATRIX_AT(m, i, j) = rand_float_range(low, high);
        }
    }
}

// Print a matrix.
void matrix_print(Matrix m) {
    printf("(%zu, %zu)\n", m.rows, m.cols);
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            printf("%.2f ", MATRIX_AT(m, i, j));
        }
        printf("\n");
    }
}

// Allocates a Vector of length cols and zeroes it.
Vector vector_alloc(size_t cols) {
    Vector v = {0};

    v.cols = cols;
    v.data = malloc(sizeof(float) * cols);
    if (v.data == NULL) {
        fprintf(stderr, "[ERROR] - Could not alloc in vector_alloc()\n");
        exit(1);
    }

    for (size_t i = 0; i < cols; i++) {
        v.data[i] = 0.0f;
    }

    return v;
}

// Fill a vector with random values between low and high.
void vector_fill_rand(Vector v, float low, float high) {
    for (size_t i = 0; i < v.cols; i++) {
        v.data[i] = rand_float_range(low, high);
    }
}

// Print a vector.
void vector_print(Vector v) {
    for (size_t i = 0; i < v.cols; i++) {
        printf("%.2f ", v.data[i]);
    }
    printf("\n");
}
