// 10.2 Page-Locked Host Memory
#include <cstdlib>

#include "common.h"

void cudaMallocTest(std::size_t size, bool host_to_device) {
    Timer timer;

    int* a, *dev_a;
    a = (int*)malloc(size * sizeof(int));
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(int)));

    timer.startTimer();
    for (int i = 0; i < 100; ++i) {
        if (host_to_device) {
            HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
        }
        else {
            HANDLE_ERROR(cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost));
        }
    }
    timer.stopTimer();
    float time = timer.readTimer();
    std::printf("Elasped time: %f ms\n", time);
    std::printf("%3.1f GB/s during copy\n", size * 100 * sizeof(int) * 1000.0f / (1024 * 1024 * 1024 * time));

    free(a);
    HANDLE_ERROR(cudaFree(dev_a));
}

void cudaHostAllocTest(std::size_t size, bool host_to_device) {
    Timer timer;

    int* a, *dev_a;
    HANDLE_ERROR(cudaHostAlloc((void**)&a, size * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(int)));

    timer.startTimer();
    for (int i = 0; i < 100; ++i) {
        if (host_to_device) {
            HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
        }
        else {
            HANDLE_ERROR(cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost));
        }
    }
    timer.stopTimer();
    float time = timer.readTimer();
    std::printf("Elasped time: %f ms\n", time);
    std::printf("%3.1f GB/s during copy\n", size * 100 * sizeof(int) * 1000.0f / (1024 * 1024 * 1024 * time));

    HANDLE_ERROR(cudaFreeHost(a));
    HANDLE_ERROR(cudaFree(dev_a));
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {
    
    constexpr int size = 10 * 1024 * 1024;

    std::printf("Test malloc() from host to device\n");
    cudaMallocTest(size, true);
    std::printf("Test malloc() from device to host\n");
    cudaMallocTest(size, false);
    std::printf("Test HostAlloc() from host to device\n");
    cudaHostAllocTest(size, true);
    std::printf("Test HostAlloc() from device to host\n");
    cudaHostAllocTest(size, false);

    return 0;
}
