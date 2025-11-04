#include <stdlib.h>
#include "utils.h"

float rand_float_range(float low, float high) {
    return low + ((float)rand() / (float)RAND_MAX) * (high - low);
}
