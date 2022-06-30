// 5.3.2 shared memory bitmap
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "common.h"

constexpr int DIM = 1024;

template <typename T = float>
__global__ void kernel(uint8_t *ptr, int x_dim, int y_dim) {
    // 2d share memory
    __shared__ float cache[16][16];
    constexpr T period = T(128.0);

    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < y_dim; y += gridDim.y * blockDim.y) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < x_dim; x += gridDim.x * blockDim.x) {
            int pixel_id = (x + y * x_dim) * 4;

            cache[threadIdx.x][threadIdx.y] = T(255.0 / 4.0) * (
                std::sin(x * T(2.0) * pi<T> / period) + T(1.0)
            ) * (
                std::sin(y * T(2.0) * pi<T> / period) + T(1.0)
            );

            // Synchronization before any thread communication.
            __syncthreads();

            ptr[pixel_id + 0] = 0;
            ptr[pixel_id + 1] = cache[15 - threadIdx.x][15 - threadIdx.y];
            ptr[pixel_id + 2] = 0;
            ptr[pixel_id + 3] = 255;
        }
    }
}

int main(int argc, char **argv) {
    PathChecker::checkFilePath(argc, argv, ".png");
    Bitmap bitmap(DIM, DIM);
    dim3 block_size(16, 16);
    dim3 grid_size(DIM / 16, DIM / 16);

    kernel<float><<<grid_size, block_size>>>(bitmap.dev_bitmap, DIM, DIM);
    bitmap.memcpyDeviceToHost();
    bitmap.toImage(argv[1]);

    return 0;
}

