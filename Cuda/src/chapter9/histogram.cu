// 9.4 Computing Histograms
#include "common.h"
#include <algorithm>
#include <numeric>
#include <random>

const std::size_t SIZE = 100 * 1024 * 1024;

// 55.15ms for RTX 2060
__global__ void kernel_global_mem(uint8_t const* d_buffer, std::size_t size, int* d_hist) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; idx < size; idx += stride) {
        atomicAdd(&d_hist[d_buffer[idx]], 1);
    }
}

// 1.66ms for RTX 2060
__global__ void kernel_share_mem(uint8_t const* d_buffer, std::size_t size, int* d_hist) {
    __shared__ int block_count[256];
    block_count[threadIdx.x] = 0;
    __syncthreads();

    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; idx < size; idx += stride) {
        atomicAdd(&block_count[d_buffer[idx]], 1);
    }
    __syncthreads();

    atomicAdd(&d_hist[threadIdx.x], block_count[threadIdx.x]);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    std::vector<uint8_t> buffer(SIZE);
    std::mt19937_64 rng;
    std::uniform_int_distribution<int> unif_u8(0, 255);
    std::generate(buffer.begin(), buffer.end(), [&rng, &unif_u8]() -> uint8_t {
        return static_cast<uint8_t>(unif_u8(rng));
    });
    uint8_t* d_buffer;
    HANDLE_ERROR(cudaMalloc((void**)&d_buffer, SIZE * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMemcpy(d_buffer, buffer.data(), SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice));

    Timer timer;
    int* d_hist;
    HANDLE_ERROR(cudaMalloc((void**)&d_hist, 256 * sizeof(int)));
    HANDLE_ERROR(cudaMemset(d_hist, 0, 256 * sizeof(int)));

    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    int multi_processor_count = prop.multiProcessorCount;
    std::printf("Multi processor count: %d\n", multi_processor_count);

    timer.startTimer();
    // kernel_global_mem<<<multi_processor_count * 4, 256>>>(d_buffer, SIZE, d_hist);
    kernel_share_mem<<<multi_processor_count * 4, 256>>>(d_buffer, SIZE, d_hist);
    timer.stopTimer();
    float time = timer.readTimer();
    std::printf("Time to generate: %3.2f ms\n", time);

    std::vector<int> hist(256);
    HANDLE_ERROR(cudaMemcpy(hist.data(), d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost));
    int count = std::reduce(hist.begin(), hist.end());
    std::printf("Sum of the histogram: %d\n", count);

    return 0;
}
