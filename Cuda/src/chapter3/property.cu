// 3.3 Querying Devices
#include <cstdio>

#include "common.h"

using namespace std;

// Parameters of devices, see more in "CUDA Programming Guide".
int main(int argc, char **argv) {

    printf("%s starting...\n", argv[0]);

    // Find cuda device.
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n->%s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA.\n");
    }
    else {
        printf("Detectd %d CUDA Capable device(s).\n", deviceCount);
    }

    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d: \"%s\".\n", dev, deviceProp.name);

    // Find driver.
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf(
        "CUDA Driver Version / Runtime Version %d.%d / %d.%d\n",
        driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10
    );
    printf("CUDA Capability Major / Minor version number: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Total amount of global memory: %0.2lf GBytes (%zu bytes)\n", (double)deviceProp.totalGlobalMem / (1073741824.0), deviceProp.totalGlobalMem);
    printf("GPU Clock rate: %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
    printf("Memory Clock rate: %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
    printf("Memory Bus Width: %d-bit\n", deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize) {
        printf("L2 Cache Size: %d bytes\n", deviceProp.l2CacheSize);
    }

    printf(
        "Max Texture Dimension Size (x, y, z) 1D = (%d), 2D = (%d, %d), 3D = (%d, %d, %d)\n",
        deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
        deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]
    );
    printf(
        "Max Layered Texture Size (dim) x layers 1D = (%d) x %d, 2D = (%d, %d) x %d\n",
        deviceProp.maxTexture1DLayered[0], deviceProp.maxSurface1DLayered[1],
        deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]
    );

    printf("Total amount of constant memory: %zu bytes\n", deviceProp.totalConstMem);
    printf("Total amount of shared memory per block: %zu bytes\n", deviceProp.sharedMemPerBlock);
    printf("Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    printf("Total number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
    printf("Warp size: %d\n", deviceProp.warpSize);
    printf("Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Maximum sizes of each dimension of a block: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("Maximum sizes of each dimension of a grid: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("Maximum memory pitch: %zu bytes\n", deviceProp.memPitch);

    return 0;
}
