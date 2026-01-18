#include <stdlib.h>
#include <math.h>

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>

#include "utils.h"

// Generates a random floating-point number within a specified range.
float rand_float_range(float low, float high) {
    return low + ((float)rand() / (float)RAND_MAX) * (high - low);
}

// Reduces exery possible x value to a value between 0 and 1.
float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

/*
    Prints a formatted string centered within an 80-character width.
    The string is converted to uppercase before printing.
    The function is variadic, meaning it accepts a format string (fmt) followed by a variable number of arguments.
*/
void print_center(char *fmt, ...) {
    // A va_list is initialized using va_start to handle the variable arguments.
    va_list ap;
    va_start(ap, fmt);
    char mybuf[64];

    // The formatted string is written into a fixed-size buffer (mybuf, 64 bytes) using vsnprintf, which prevents buffer overflow by limiting the number of characters written.
    vsnprintf(mybuf, sizeof(mybuf), fmt, ap);

    for(size_t j = 0; j < sizeof(mybuf); j++) {
        // Every character in the buffer is converted to uppercase using toupper.
        mybuf[j] = toupper(mybuf[j]);
    }

    // The amount of left padding needed to center the text in an 80-column terminal is calculated
    size_t len = strlen(mybuf);
    size_t padding = (80 - len)/2;

    for (size_t j = 0; j < padding; j++) printf(" ");
    printf("%s\n", mybuf);

    // Finally, va_end is called to properly clean up the variable argument list.
    va_end(ap);
}
