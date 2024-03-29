// 5.3.1 Dot Product
#include <algorithm>
#include <numeric>

#include "common.h"

constexpr int ThreadsPerBlock{256};

__global__ void dotProduct(int const N, float const* a, float const* b, float *c) {
    __shared__ float cache[ThreadsPerBlock];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cache_index = threadIdx.x;

    float result{};
    for (; tid < N; tid += blockDim.x * gridDim.x) {
        result += a[tid] * b[tid];
    }

    // Store the thread result in the cache.
    cache[cache_index] = result;
    __syncthreads();

    // Reduction process, reduce the result as one.
    // This requires blockDim.x to be power of 2.
    for (int i = blockDim.x / 2; i != 16; i /= 2) {
        if (cache_index < i) {
            cache[cache_index] += cache[cache_index + i];
        }
        //? Why `__syncthreads()` here?
        // It is because of SIMD. Detailedly, if there is a ` __syncthreads()` in the branch,
        // then all the thread in block will wait forever and core dump.
        //! Actually, I tried and nothing happened. Maybe the book is too old~
        __syncthreads();
    }
    // Warp reduce
    if (threadIdx.x < 32) {
        // Need volatile to implicit synchronization
        volatile float* vss = cache;
        float val = vss[threadIdx.x];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        // Store the block result in `c`.
        if (cache_index == 0) {
            c[blockIdx.x] = val;
        }
    }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {

    constexpr int ArraySize{33 * 1024};
    constexpr int BlocksPerGrid{std::min(32, (ArraySize + ThreadsPerBlock - 1) / ThreadsPerBlock)};

    float *a = new float[ArraySize];
    float *b = new float[ArraySize];
    float *c = new float[BlocksPerGrid];
    float *dev_a{}, *dev_b{}, *dev_c{};

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, ArraySize * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, ArraySize * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, BlocksPerGrid * sizeof(int)));

    for (int i{}; i < ArraySize; ++i) {
        a[i] = (float)i;
        b[i] = (float)i * 2;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, ArraySize * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, ArraySize * sizeof(int), cudaMemcpyHostToDevice));

    dotProduct<<<BlocksPerGrid, ThreadsPerBlock>>>(ArraySize, dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, BlocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));

    float result = std::accumulate(c, c + BlocksPerGrid, 0.0f);
    std::printf("Result = %f\n", result);
    auto sumOfSquare = [](float x) -> float { return x * (x + 1) * (2 * x + 1) / 6; };
    std::printf("Expected result = %f\n", sumOfSquare(ArraySize - 1) * 2.0f);
    std::printf("The difference comes from the floating point error.\n");

    delete[] a;
    delete[] b;
    delete[] c;
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));
    return 0;
}
