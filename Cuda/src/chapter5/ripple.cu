// 5.2.2 GPU Ripple Using Threads
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "common.h"

constexpr int DIM = 1024;

template <typename T = float>
__global__ void kernel(uint8_t *ptr, int tick, int x_dim, int y_dim) {
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < y_dim; y += gridDim.y * blockDim.y) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < x_dim; x += gridDim.x * blockDim.x) {
            int pixel_id = (x + y * x_dim) * 4;
            int rx = x - x_dim / 2;
            int ry = y - y_dim / 2;
            T r = std::sqrt(T(rx * rx + ry * ry));
            uint8_t value = (
                T(128.0) + T(127.0) * std::cos(
                    (r / T(15.0) - tick / T(10.0)) * pi<T>
                ) / (r / T(15.0) + T(1.0))
            );

            ptr[pixel_id + 0] = value;
            ptr[pixel_id + 1] = value;
            ptr[pixel_id + 2] = value;
            ptr[pixel_id + 3] = 255;
        }
    }
}

int main(int argc, char **argv) {
    PathChecker::checkPath(argc, argv);
    Bitmap bitmap(DIM, DIM);
    dim3 block_size(16, 16);
    dim3 grid_size(32, 32);

    for (int tick{}; tick < 60; ++tick) {
        kernel<float><<<grid_size, block_size>>>(bitmap.dev_bitmap, tick, DIM, DIM);
        bitmap.memcpyDeviceToHost();
        bitmap.toImage(
            std::string(argv[1]) + '/' + std::to_string(tick) + ".png"
        );
    }

    return 0;
}
