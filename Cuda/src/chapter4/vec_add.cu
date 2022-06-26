// 4.2.1 summing Vectors
#include <cstdio>

#include "common.h"

constexpr int N = 10;

__global__ void addKernel(int const *a, int const *b, int *c) {
    int tid = threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;
    
    // Allocate memory on the GPU.
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, sizeof(int) * N));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, sizeof(int) * N));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int) * N));

    // Fill the arrays `a` and `b` on the CPU.
    for (int i{}; i < N; ++i) {
        a[i] = -i;
        b[i] = i * i;
    }

    // Copy the arrays `a` and `b` to the GPU.
    HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice));

    addKernel<<<1, N>>>(dev_a, dev_b, dev_c);

    // Copy the array `c` back from the GPU to the CPU.
    HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost));

    // Display the results.
    for (int i{}; i < N; ++i) {
        std::printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));
    return 0;
}
