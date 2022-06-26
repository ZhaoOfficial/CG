// 5.3.2 shared memory bitmap
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "common.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

constexpr int DIM = 1024;

template <typename T = float>
__global__ void kernel(uint8_t *ptr, unsigned int x_dim, unsigned int y_dim) {
    // 2d share memory
    __shared__ float cache[16][16];
    constexpr T period = T(128.0);

    for (unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; y < y_dim; y += gridDim.y * blockDim.y) {
        for (unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; x < x_dim; x += gridDim.x * blockDim.x) {
            unsigned int pixel_id = (x + y * x_dim) * 4;

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
    if (argc != 2) {
        std::printf("Output path invalid.\n");
        return EXIT_FAILURE;
    }
    std::filesystem::path file_path(argv[1]);
    if (file_path.extension() != ".png") {
        std::printf("Output path [%s] invalid.\n", file_path.string().c_str());
        return EXIT_FAILURE;
    }
    std::printf("Output path: %s\n", argv[1]);

    std::vector<uint8_t> bitmap(DIM * DIM * 4);
    uint8_t *dev_bitmap;

    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.size() * sizeof(uint8_t)));

    dim3 block_size(16, 16);
    dim3 grid_size(DIM / 16, DIM / 16);
    kernel<float><<<grid_size, block_size>>>(dev_bitmap, DIM, DIM);

    HANDLE_ERROR(cudaMemcpy(bitmap.data(), dev_bitmap, bitmap.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    stbi_flip_vertically_on_write(1);
    stbi_write_png(file_path.string().c_str(), DIM, DIM, 4, bitmap.data(), 0);
    std::printf("%s output successfully!\n", file_path.string().c_str());

    HANDLE_ERROR(cudaFree(dev_bitmap));
    return 0;
}

