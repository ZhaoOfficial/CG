// 6.3 Constant Memory
#include <random>

#include "cpu_anim_bitmap.h"

constexpr int DIM = 4096;
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

__global__ void kernel(uint8_t* ptr, int x_dim, int y_dim) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int pixel_id = (x + y * x_dim) * 4;

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
    PathChecker::checkFilePath(argc, argv, ".png");
    CPUAnimBitmap bitmap(DIM, DIM, "Ray Tracing", nullptr);
    Timer timer;

    // Startup
    std::random_device rd;
    // std::mt19937_64 rng(rd());
    std::mt19937_64 rng(0);
    std::uniform_real_distribution<float> color(0.0f, 1.0f);
    std::uniform_real_distribution<float> position(-1600.0f, 1600.0f);
    std::uniform_real_distribution<float> radius(100.0f, 300.0f);

    timer.startTimer();
    Sphere *s = new Sphere[NUM_SPHERE];

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
    kernel<<<grid_size, block_size>>>(bitmap.gpu_bitmap, DIM, DIM);

    timer.stopTimer();

    // 5~6 ms
    std::printf("Time to generate figure: %.3f ms\n", timer.readTimer());
    bitmap.toImage(argv[1]);

    delete[] s;
    return 0;
}
