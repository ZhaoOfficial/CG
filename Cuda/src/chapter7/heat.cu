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

__global__ void copyConstantKernel(float *in_ptr, float const *const_ptr, int x_dim, int y_dim) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int coord = x + y * x_dim;

    if (const_ptr[coord] != 0.0f) {
        in_ptr[coord] = const_ptr[coord];
    }
}

__global__ void blendKernel(float *out_ptr, float const *in_ptr, int x_dim, int y_dim) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int coord = x + y * x_dim;

    int left   = (x == 0 ? coord : coord - 1);
    int right  = (x == x_dim - 1 ? coord : coord + 1);
    int top    = (y == 0 ? coord : coord - x_dim);
    int bottom = (y == y_dim - 1 ? coord : coord + x_dim);

    // Update equation: T_{new} = T_{old} + \sum_{n\in neighbor}k\cdot(T_{n}-T_{old})
    out_ptr[coord] = in_ptr[coord] + SPEED * (
        in_ptr[left] + in_ptr[right] + in_ptr[top] + in_ptr[bottom] - in_ptr[coord] * 4
    );
}

__global__ void floatToUint8(float *out_ptr, uint8_t *bitmap_ptr, int x_dim, int y_dim) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int coord = x + y * x_dim;
    int pixel_id = coord * 4;

    bitmap_ptr[pixel_id + 0] = out_ptr[coord];
    bitmap_ptr[pixel_id + 1] = out_ptr[coord];
    bitmap_ptr[pixel_id + 2] = out_ptr[coord];
    bitmap_ptr[pixel_id + 3] = 255;
}

class DataBlock {
public:
    DataBlock() : bitmap(DIM, DIM) {
        HANDLE_ERROR(cudaMalloc((void**)&dev_in, bitmap.numPixels()));
        HANDLE_ERROR(cudaMalloc((void**)&dev_out, bitmap.numPixels()));
        HANDLE_ERROR(cudaMalloc((void**)&dev_const, bitmap.numPixels()));
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

    void animate();

public:
    Bitmap bitmap;
    float * dev_in;
    float * dev_out;
    float * dev_const;
    cudaEvent_t start, stop;
    float total_time{};
    int frames{};
};

void DataBlock::animate() {
    dim3 block_size(16, 16);
    dim3 grid_size(DIM / 16, DIM / 16);

    HANDLE_ERROR(cudaEventRecord(this->start, 0));
    for (std::size_t i{}; i < 90; ++i) {
        copyConstantKernel<<<grid_size, block_size>>>(this->dev_in, this->dev_const, DIM, DIM);
        blendKernel<<<grid_size, block_size>>>(this->dev_out, this->dev_in, DIM, DIM);
        std::swap(this->dev_in, this->dev_out);
    }
    floatToUint8<<<grid_size, block_size>>>(this->dev_out, this->bitmap.dev_bitmap, DIM, DIM);

    HANDLE_ERROR(cudaEventRecord(this->stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(this->stop));

    float elapsed_time_ms;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time_ms, this->start, this->stop));
    this->total_time += elapsed_time_ms;
    ++this->frames;
    std::printf("Average time per frame: %.3f ms\n", this->total_time / this->frames);
}

int main(int argc, char **argv) {
    PathChecker::checkPath(argc, argv);
    DataBlock data;

    // Initialize the constant area.
    float * temp = new float[data.bitmap.numPixels()]{};
    for (std::size_t i{}; i < DIM * DIM; ++i) {
        int x = i % DIM;
        int y = i / DIM;
        if ((x > 300) && (x < 600) && (y > 310) && (y < 601)) {
            temp[i] = MAX_TEMP;
        }
    }
    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2.0f;
    temp[DIM * 700 + 100] = MIN_TEMP;
    temp[DIM * 300 + 300] = MIN_TEMP;
    temp[DIM * 200 + 700] = MIN_TEMP;
    for (std::size_t y{800}; y < 900; ++y) {
        for (std::size_t x{400}; x < 500; ++x) {
            temp[x + y * DIM] = MIN_TEMP;
        }
    }
    HANDLE_ERROR(cudaMemcpy(data.dev_const, temp, data.bitmap.numPixels(), cudaMemcpyHostToDevice));

    // Initialize the input area.
    for (std::size_t y{800}; y < DIM; ++y) {
        for (std::size_t x{0}; x < 200; ++x) {
            temp[x + y * DIM] = MAX_TEMP;
        }
    }
    HANDLE_ERROR(cudaMemcpy(data.dev_in, temp, data.bitmap.numPixels(), cudaMemcpyHostToDevice));
    delete[] temp;

    // Start animation
    for (int tick{}; tick < 40; ++tick) {
        data.animate();
        data.bitmap.memcpyDeviceToHost();
        data.bitmap.toImage(
            std::string(argv[1]) + '/' + std::to_string(tick) + ".png"
        );
    }

    return 0;
}
