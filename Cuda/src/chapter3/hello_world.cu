// 3.2.2 A Kernel Call
#include <cstdio>

__global__ void kernel() {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    std::printf("Hello World from the kernel, thread = %d!\n", tid);
}

int main() {
    kernel<<<1, 1>>>();
    std::printf("Hello World!\n");
    return 0;
}
