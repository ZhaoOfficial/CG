// 5.2.2 GPU Ripple Using Threads
#include <vector>
#include <cmath>

#include <stdio.h>

#include "common.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

constexpr int DIM = 1024;
template <typename T>
constexpr T pi = T(3.1415926535897932);

template <typename T = float>
__global__ void kernel(uint8_t *ptr, int tick, int x_dim, int y_dim) {
    for (unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; y < y_dim; y += gridDim.y * blockDim.y) {
        for (unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; x < x_dim; x += gridDim.x * blockDim.x) {
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
    if (argc != 2) {
        printf("Output path invalid.\n");
        return EXIT_FAILURE;
    }
    printf("Output path: %s\n", argv[1]);

    std::vector<uint8_t> bitmap(DIM * DIM * 4);
    uint8_t *dev_bitmap;

    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.size() * sizeof(uint8_t)));

    dim3 block_size(16, 16);
    dim3 grid_size(32, 32);
    for (int tick{}; tick < 20; ++tick) {
        kernel<float><<<grid_size, block_size>>>(dev_bitmap, tick, DIM, DIM);

        HANDLE_ERROR(cudaMemcpy(bitmap.data(), dev_bitmap, bitmap.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        char path[256];
        sprintf(path, "%s/%d.png", argv[1], tick);
        stbi_flip_vertically_on_write(1);
        stbi_write_png(path, DIM, DIM, 4, bitmap.data(), 0);
        printf("%s output successfully!\n", path);
    }
    HANDLE_ERROR(cudaFree(dev_bitmap));
    return 0;
}
