// 3.2.2 A Kernel Call
#include <stdio.h>

__global__ void kernel() {}

int main() {
    kernel<<<1, 1>>>();
    printf("Hello World!\n");
    return 0;
}
