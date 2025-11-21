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

// Copy the src matrix values into the dst matrix.
void matrix_copy(Matrix dst, Matrix src) {
    if ((src.rows != dst.rows) || (src.cols != dst.cols)) {
        fprintf(stderr, "[ERROR] - Matrix dimensions do not match matrix_copy()\n");
        exit(1);
    }

    for (size_t i = 0; i < src.rows; i++) {
        for (size_t j = 0; j < src.cols; j++) {
            MATRIX_AT(dst, i, j) = MATRIX_AT(src, i, j);
        }
    }
}

// Perform matrix multiplication.
void matrix_mult(Matrix dst, Matrix a, Matrix b) {
    if (a.cols != b.rows) {
        fprintf(stderr, "[ERROR] - Matrix dimensions do not work out matrix_mult(): a, b\n");
        exit(1);
    }

    if ((dst.rows != a.rows) || (dst.cols != b.cols)) {
        fprintf(stderr, "[ERROR] - Matrix dimensions do not work out matrix_mult(): dst\n");
        exit(1);
    }

    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            // Compute dst[i][j]
            MATRIX_AT(dst, i, j) = 0.0f;
            for (size_t k = 0; k < a.cols; k++) {
                MATRIX_AT(dst, i, j) += MATRIX_AT(a, i, k) * MATRIX_AT(b, k, j);
            }
        }
    }
}

// Perform matrix sum.
void matrix_sum(Matrix dst, Matrix a) {
    if ((dst.rows != a.rows) || (dst.cols != a.cols)) {
        fprintf(stderr, "[ERROR] - Matrix dimensions do not work out matrix_sum()\n");
        exit(1);
    }

    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MATRIX_AT(dst, i, j) += MATRIX_AT(a, i, j);
        }
    }
}

// Apply act_fn to all the matrix values. Uses function pointers in order to parametrize the act_fn.
void matrix_apply(Matrix a, float (*act_fn)(float)) {
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < a.cols; j++) {
            MATRIX_AT(a, i, j) = act_fn(MATRIX_AT(a, i, j));
        }
    }
}

// Load a 0|1 matrix from a file.
Matrix matrix_load_from_file(size_t rows, size_t cols, const char *filename) {
    Matrix m = matrix_alloc(rows, cols);

    FILE *fp = fopen(filename, "r");
    if(!fp) {
        fprintf(stderr, "[ERROR] - Could not open file %s in matrix_load_from_file()\n", filename);
        exit(1);
    }

    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            if(fscanf(fp, "%f", &MATRIX_AT(m, i, j)) != 1) {
                fprintf(stderr, "[ERROR] - Could not read data from file %s in matrix_load_from_file()\n", filename);
                fclose(fp);
                exit(1);
            }
        }
    }

    fclose(fp);

    /* Print matrix
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++)
            printf("%f ", MATRIX_AT(m, i, j));
        printf("\n");
    }
    */

    return m;
}

// Allocates a Vector of length len and zeroes it.
Vector vector_alloc(size_t len) {
    Vector v = {0};

    v.len = len;
    v.data = malloc(sizeof(float) * len);
    if (v.data == NULL) {
        fprintf(stderr, "[ERROR] - Could not alloc in vector_alloc()\n");
        exit(1);
    }

    for (size_t i = 0; i < len; i++) {
        v.data[i] = 0.0f;
    }

    return v;
}

// Fill a vector with random values between low and high.
void vector_fill_rand(Vector v, float low, float high) {
    for (size_t i = 0; i < v.len; i++) {
        v.data[i] = rand_float_range(low, high);
    }
}

// Print a vector.
void vector_print(Vector v) {
    for (size_t i = 0; i < v.len; i++) {
        printf("%.2f ", v.data[i]);
    }
    printf("\n");
}

// Cast a Vector into a 1 x N Matrix.
Matrix vector_as_matrix(Vector v) {
    return (Matrix) {
        .rows = 1,
        .cols = v.len,
        .data = v.data
    };
};
