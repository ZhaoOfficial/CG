// 7.3 Simulating Heat Transfer
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "common.h"

constexpr float SPEED = 0.25f;
constexpr int DIM = 1024;
constexpr float MAX_TEMP = 1.0f;
constexpr float MIN_TEMP = 1e-4f;

class DataBlock {
public:
    DataBlock()
        : bitmap(DIM * DIM * 4) {
        HANDLE_ERROR(cudaMalloc((void**)&dev_in, bitmap.size()));
        HANDLE_ERROR(cudaMalloc((void**)&dev_out, bitmap.size()));
        HANDLE_ERROR(cudaMalloc((void**)&dev_const, bitmap.size()));
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
    }

    ~DataBlock() {
        HANDLE_ERROR(cudaFree(dev_in));
        HANDLE_ERROR(cudaFree(dev_out));
        HANDLE_ERROR(cudaFree(dev_const));
        HANDLE_ERROR(cudaEventDestroy(start));
        HANDLE_ERROR(cudaEventDestroy(stop));
    }

public:
    std::vector<uint8_t> bitmap;
    uint8_t *dev_bitmap;
    float * dev_in;
    float * dev_out;
    float * dev_const;
    cudaEvent_t start, stop;
};

__global__ void copyConstantKernel(float *in_ptr, float const *const_ptr, unsigned int x_dim, unsigned int y_dim) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int coord = x + y * x_dim;

    if (const_ptr[coord] != 0.0f) {
        in_ptr[coord] = const_ptr[coord];
    }
}

__global__ void blendKernel(float *out_ptr, float const *in_ptr, unsigned int x_dim, unsigned int y_dim) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int coord = x + y * x_dim;

    unsigned int left = coord - 1;
    unsigned int right = coord + 1;
    if (x == 0) ++left;
    if (x == x_dim - 1) --right;

    unsigned int top = coord - x_dim;
    unsigned int bottom = coord + x_dim;
    if (y == 0) top += x_dim;
    if (y == y_dim - 1) bottom -= x_dim;

    out_ptr[coord] = in_ptr[coord] + SPEED * (in_ptr[left] + in_ptr[right] + in_ptr[top] + in_ptr[bottom] - in_ptr[coord] * 4)

}

int main(int argc, char **argv) {
    DataBlock data;


    // Update equation: T_{new} = T_{old} + \sum_{n\in neighbor}k\cdot(T_{n}-T_{old})

    return 0;
}
