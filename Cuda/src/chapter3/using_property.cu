// 3.4 Using Device Properties
#include <cstdio>

#include "common.h"

int main() {
    int device;
    HANDLE_ERROR(cudaGetDevice(&device));
    std::printf("ID of current CUDA device: %d\n", device);

    // Fill a `cudaDeviceProp` struct with
    // the properties we need our device have.
    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 7;
    prop.minor = 5;

    // Then we can choose the device.
    HANDLE_ERROR(cudaChooseDevice(&device, &prop));
    std::printf("ID of CUDA device closest to 7.5: %d\n", device);
    HANDLE_ERROR(cudaSetDevice(device));

    return 0;
}
