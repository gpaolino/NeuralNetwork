#include <stdlib.h>
#include <math.h>

#include "utils.h"

// Generates a random floating-point number within a specified range.
float rand_float_range(float low, float high) {
    return low + ((float)rand() / (float)RAND_MAX) * (high - low);
}

// Reduces exery possible x value to a value between 0 and 1.
float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}
