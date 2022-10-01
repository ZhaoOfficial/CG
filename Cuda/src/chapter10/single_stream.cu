// 10.4 Using a Single CUDA Stream
#include "common.h"

__global__ void kernel(uint32_t num_elem, float const* a, float const* b, float* c) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_elem) {
        uint32_t idx1 = (idx + 1) % 256;
        uint32_t idx2 = (idx + 2) % 256;
        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2.0f;
    }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {

    cudaDeviceProp prop;
    int which_device;
    HANDLE_ERROR(cudaGetDevice(&which_device));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, which_device));
    if (!prop.deviceOverlap) {
        throw std::runtime_error{"Device will not handle overlaps, so no speed up from streams\n"};
    }

    return 0;
}
