// 4.2.2 A Fun Example
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "common.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

constexpr int DIM = 4096;

template <typename T = float>
struct CudaComplex {
    __device__ CudaComplex(T real, T imag) : real{real}, imag{imag} {}
    __device__ T squareNorm() { return real * real + imag * imag; }
    __device__ CudaComplex& operator+=(CudaComplex const& other) {
        this->real += other.real;
        this->imag += other.imag;
        return *this;
    }
    __device__ CudaComplex& operator-=(CudaComplex const& other) {
        this->real -= other.real;
        this->imag -= other.imag;
        return *this;
    }
    __device__ CudaComplex& operator*=(CudaComplex const& other) {
        T real_val = this->real * other.real - this->imag * other.imag;
        this->imag = this->imag * other.real + this->real * other.imag;
        this->real = real_val;
        return *this;
    }

    T real;
    T imag;
};

template <typename T = float>
__device__ int julia(int x_pos, int y_pos) {
    static constexpr T scale{1.5};
    float x = scale * (T)(DIM / 2 - x_pos) / (DIM / 2);
    float y = scale * (T)(DIM / 2 - y_pos) / (DIM / 2);

    CudaComplex<T> c(-0.8, 0.156);
    CudaComplex<T> z(x, y);

    int iters = 0;
    for (; iters < 255; ++iters) {
        z *= z;
        z += c;
        if (z.squareNorm() > T(1000.0)) {
            return iters;
        }
    }
    return 255;
};

template <typename T = float>
__global__ void kernel(uint8_t *ptr, int x_dim, int y_dim) {
    for (unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; y < y_dim; y += gridDim.y * blockDim.y) {
        for (unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; x < x_dim; x += gridDim.x * blockDim.x) {
            int pixel_id = (x + y * x_dim) * 4;
            uint8_t value = julia<T>(x, y);
            ptr[pixel_id + 0] = value;
            ptr[pixel_id + 1] = value;
            ptr[pixel_id + 2] = value;
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
    dim3 grid_size(32, 32);

    kernel<float><<<grid_size, block_size>>>(dev_bitmap, DIM, DIM);

    HANDLE_ERROR(cudaMemcpy(bitmap.data(), dev_bitmap, bitmap.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    stbi_flip_vertically_on_write(1);
    stbi_write_png(argv[1], DIM, DIM, 4, bitmap.data(), 0);
    std::printf("Image output successfully!\n");
    HANDLE_ERROR(cudaFree(dev_bitmap));
    return 0;
}
