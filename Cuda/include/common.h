#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdio.h>
#include <stdlib.h>

static void handleError(cudaError_t err, char const *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (handleError(err, __FILE__, __LINE__))

#define HANDLE_NULL(ptr) do {                                                \
    if (a == NULL) {                                                         \
        printf("Host memory failed in %s at line %d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while (0)

#endif // !_COMMON_H_
