#include <stdlib.h>

#include "utils.h"

// Generates a random floating-point number within a specified range.
float rand_float_range(float low, float high) {
    return low + ((float)rand() / (float)RAND_MAX) * (high - low);
}
