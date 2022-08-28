// 5.2.2 GPU Ripple Using Threads
#include <cmath>

#include "cpu_anim_bitmap.h"

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

void renderFrame(uint8_t* device_ptr, int tick, int x_dim, int y_dim) {
    dim3 block_size(32, 16);
    dim3 grid_size(x_dim / 32, y_dim / 16);
    kernel<float><<<grid_size, block_size>>>(device_ptr, tick, x_dim, y_dim);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {
    CPUAnimBitmap bitmap(DIM, DIM, "Ripple", nullptr);

    bitmap.animate(renderFrame);

    return 0;
}
