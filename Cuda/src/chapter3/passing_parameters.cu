// 3.2.3 Passing Parameters
#include "common.h"

__global__ void addKernel(int a, int b, int *c) {
    *c = a + b;
}

int main() {
    int c;
    int *dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));
    addKernel<<<1, 1>>>(2, 7, dev_c);
    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("2 + 7 = %d\n", c);

    HANDLE_ERROR(cudaFree(dev_c));
    return 0;
}
