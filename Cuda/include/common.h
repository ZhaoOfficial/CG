#ifndef _COMMON_H_
#define _COMMON_H_

#include <cstdio>
#include <cstdlib>
#include <limits>

static void handleError(cudaError_t err, char const *file, int line) {
    if (err != cudaSuccess) {
        std::printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (handleError(err, __FILE__, __LINE__))

#define HANDLE_NULL(ptr) do {                                                \
    if (a == NULL) {                                                         \
        std::printf("Host memory failed in %s at line %d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while (0)

template <typename T>
constexpr T pi = T(3.1415926535897932);
template <typename T>
constexpr T inf = std::numeric_limits<T>::max();

#endif // !_COMMON_H_
