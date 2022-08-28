// 8.3 GPU Ripple with Graphics Interoperability
#include <cmath>

#include "gpu_anim_bitmap.h"

constexpr int DIM = 1024;

template <typename T = float>
__global__ void kernel(uchar4 *ptr, int tick, int x_dim, int y_dim) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_id = x + y * x_dim;

    int rx = x - x_dim / 2;
    int ry = y - y_dim / 2;
    T r = std::sqrt(T(rx * rx + ry * ry));
    uint8_t value = (
        T(128.0) + T(127.0) * std::cos(
            (r / T(15.0) - tick / T(10.0)) * pi<T>
        ) / (r / T(15.0) + T(1.0))
    );

    ptr[pixel_id].x = value;
    ptr[pixel_id].y = value;
    ptr[pixel_id].z = value;
    ptr[pixel_id].w = 255;
}

void renderFrame(uchar4* device_ptr, int tick, int x_dim, int y_dim) {
    dim3 block_size(32, 16);
    dim3 grid_size(x_dim / 32, y_dim / 16);
    kernel<float><<<grid_size, block_size>>>(device_ptr, tick, x_dim, y_dim);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {
    GPUAnimBitmap bitmap(DIM, DIM, "Ripple CUDA-OpenGL interoperation", nullptr);

    bitmap.animate(renderFrame);

    return 0;
}
