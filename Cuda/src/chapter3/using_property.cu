// 3.4 Using Device Properties
#include <stdio.h>

#include "common.h"

int main() {
    int device;
    HANDLE_ERROR(cudaGetDevice(&device));
    printf("ID of current CUDA device: %d\n", device);

    // fill a `cudaDeviceProp` struct with
    // the properties we need our device have
    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 7;
    prop.minor = 5;

    // then we can choose the device
    HANDLE_ERROR(cudaChooseDevice(&device, &prop));
    printf("ID of CUDA device closest to 7.5: %d\n", device);
    HANDLE_ERROR(cudaSetDevice(device));

    return 0;
}
