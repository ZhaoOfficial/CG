// 4.2.2 A Fun Example
#include <cmath>

#include "common.h"

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
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < y_dim; y += gridDim.y * blockDim.y) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < x_dim; x += gridDim.x * blockDim.x) {
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
    PathChecker::checkFilePath(argc, argv, ".png");
    Bitmap bitmap(DIM, DIM);
    dim3 block_size(16, 16);
    dim3 grid_size(32, 32);

    kernel<float><<<grid_size, block_size>>>(bitmap.dev_bitmap, DIM, DIM);
    bitmap.memcpyDeviceToHost();
    bitmap.toImage(argv[1]);

    return 0;
}
