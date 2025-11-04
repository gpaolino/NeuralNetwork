#include <stdlib.h>

#ifndef NUMSTRUCT_H
#define NUMSTRUCT_H

// Vector is a 1-D container.
typedef struct {
    size_t cols;
    float *data;
} Vector;

// Matrix stores number of rows and columns and a contiguous float *data in row-major order. The Matrix is stored in a linear way.
typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} Matrix;

// Macro to index into Matrix in row-major layout.
#define MATRIX_AT(m, i, j) (m).data[(i) * (m).cols + (j)]

Matrix matrix_alloc(size_t rows, size_t cols);
void matrix_fill_rand(Matrix m, float low, float high);
void matrix_print(Matrix m);
Vector vector_alloc(size_t cols);
void vector_fill_rand(Vector v, float low, float high);
void vector_print(Vector v);

#endif
