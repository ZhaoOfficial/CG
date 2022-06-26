// 6.3 Constant Memory
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <vector>

#include "common.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

constexpr int DIM = 2048;
constexpr int NUM_SPHERE = 30;

struct Sphere {
    float x, y, z;
    float radius;
    float r, g, b;
    // Quite simple hit function.
    // Since we only hit along z axis.
    __device__ float hit(float ox, float oy, float* n) const {
        float dx = ox - this->x;
        float dy = oy - this->y;
        if (dx * dx + dy * dy < this->radius * this->radius) {
            float dz = std::sqrt(this->radius * this->radius - dx * dx - dy * dy);
            *n = dz / this->radius;
            return dz + this->z;
        }
        return -inf<float>;
    }
};

__constant__ Sphere dev_s[NUM_SPHERE];

__global__ void kernel(uint8_t* ptr, unsigned int x_dim, unsigned int y_dim) {
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int pixel_id = (x + y * x_dim) * 4;

    float ox = ((float)x - (float)x_dim / 2);
    float oy = ((float)y - (float)y_dim / 2);
    float r{}, g{}, b{};
    float max_z = -inf<float>;

    for (int i{}; i < NUM_SPHERE; ++i) {
        float n;
        float t = dev_s[i].hit(ox, oy, &n);
        if (t > max_z) {
            float f_scale = n;
            r = dev_s[i].r * f_scale;
            g = dev_s[i].g * f_scale;
            b = dev_s[i].b * f_scale;
            max_z = t;
        }
    }

    ptr[pixel_id + 0] = r * 255.0f;
    ptr[pixel_id + 1] = g * 255.0f;
    ptr[pixel_id + 2] = b * 255.0f;
    ptr[pixel_id + 3] = 255;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::printf("Output path invalid.\n");
        return EXIT_FAILURE;
    }
    std::filesystem::path file_path(argv[1]);
    if (file_path.extension() != ".png") {
        std::printf("Output path [%s] invalid.\n", file_path.string().c_str());
        return EXIT_FAILURE;
    }
    std::printf("Output path: %s\n", argv[1]);

    // Startup
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    std::vector<uint8_t> bitmap(DIM * DIM * 4);
    uint8_t *dev_bitmap;
    Sphere *s = new Sphere[NUM_SPHERE];

    std::random_device rd;
    // std::mt19937_64 rng(rd{});
    std::mt19937_64 rng(0);
    std::uniform_real_distribution<float> color(0.0f, 1.0f);
    std::uniform_real_distribution<float> position(-800.0f, 800.0f);
    std::uniform_real_distribution<float> radius(50.0f, 150.0f);

    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.size()));
    HANDLE_ERROR(cudaMalloc((void**)&dev_s, sizeof(Sphere) * NUM_SPHERE));

    for (std::size_t i{}; i < NUM_SPHERE; ++i) {
        s[i].r = color(rng);
        s[i].g = color(rng);
        s[i].b = color(rng);
        s[i].x = position(rng);
        s[i].y = position(rng);
        s[i].z = position(rng);
        s[i].radius = radius(rng);
    }

    HANDLE_ERROR(cudaMemcpyToSymbol(dev_s, s, sizeof(Sphere) * NUM_SPHERE));

    // Rendering
    dim3 block_size(16, 16);
    dim3 grid_size(DIM / 16, DIM / 16);
    kernel<<<grid_size, block_size>>>(dev_bitmap, DIM, DIM);

    HANDLE_ERROR(cudaMemcpy(bitmap.data(), dev_bitmap, bitmap.size(), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    float elapsed_time_ms;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time_ms, start, stop));
    std::printf("Time to generate figure: %.3f ms\n", elapsed_time_ms);

    stbi_flip_vertically_on_write(1);
    stbi_write_png(file_path.string().c_str(), DIM, DIM, 4, bitmap.data(), 0);
    std::printf("%s output successfully!\n", file_path.string().c_str());

    delete[] s;
    HANDLE_ERROR(cudaFree(dev_bitmap));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    return 0;
}
