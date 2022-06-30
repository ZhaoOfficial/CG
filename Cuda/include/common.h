#ifndef _COMMON_H_
#define _COMMON_H_

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

template <typename T>
constexpr T pi = T(3.1415926535897932);
template <typename T>
constexpr T inf = std::numeric_limits<T>::max();

static inline void handleError(cudaError_t err, char const *file, int line) {
    if (err != cudaSuccess) {
        std::printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (handleError(err, __FILE__, __LINE__))

#define HANDLE_NULL(ptr) do {                                                     \
    if (a == NULL) {                                                              \
        std::printf("Host memory failed in %s at line %d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                       \
    }                                                                             \
} while (0)

class PathChecker {
public:
    PathChecker() = delete;
    ~PathChecker() = delete;
    // Check number of arguments.
    static void checkNumArgs(int argc);
    // Check if the path exists.
    static void checkPath(int argc, char **argv);
    // Check if the file path with required extension exists.
    static void checkFilePath(int argc, char **argv, std::string_view ext);
};

class Bitmap {
public:
    Bitmap(std::size_t x_dim, std::size_t y_dim);
    ~Bitmap();

    // memory copy from gpu to cpu
    void memcpyDeviceToHost();
    // generate an image
    void toImage(std::string const& path) const;
    uint8_t const* data() const;
    std::size_t size() const;
    std::size_t numPixels() const;

public:
    std::size_t x_dim, y_dim;
    std::vector<uint8_t> bitmap;
    uint8_t* dev_bitmap;
};

#endif // !_COMMON_H_
