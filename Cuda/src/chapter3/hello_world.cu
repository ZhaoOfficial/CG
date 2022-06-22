// 3.2.2 A Kernel Call
#include <cstdio>

__global__ void kernel() {}

int main() {
    kernel<<<1, 1>>>();
    std::printf("Hello World!\n");
    return 0;
}
