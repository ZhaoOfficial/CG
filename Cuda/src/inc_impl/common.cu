#include "common.h"

__device__ uint8_t value(float n1, float n2, int hue) {
    if (hue > 360)    hue -= 360;
    else if (hue < 0) hue += 360;

    if (hue < 60) {
        return (uint8_t)(255 * (n1 + (n2 - n1) * hue / 60));
    }
    else if (hue < 180) {
        return (uint8_t)(255 * n2);
    }
    else if (hue < 240) {
        return (uint8_t)(255 * (n1 + (n2 - n1) * (240 - hue) / 60));
    }
    else {
        return (uint8_t)(255 * n1);
    }
}

__global__ void floatToUint8(
    uint8_t *bitmap, float const *src, int x_dim, int y_dim
) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int coord = x + y * x_dim;

    float c = src[coord];
    int hue = (180 + (int)(360.0f * c)) % 360;

    float m2{1};
    if (c <= 0.5f) {
        m2 = 2 * c;
    }
    float m1 = 2 * c - m2;

    int pixel_id = coord * 4;
    bitmap[pixel_id + 0] = value(m1, m2, hue + 120);
    bitmap[pixel_id + 1] = value(m1, m2, hue);
    bitmap[pixel_id + 2] = value(m1, m2, hue - 120);
    bitmap[pixel_id + 3] = 255;
}

__global__ void floatToUint8(
    uchar4 *bitmap, float const *src, int x_dim, int y_dim
) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int coord = x + y * x_dim;

    float c = src[coord];
    int hue = (180 + (int)(360.0f * c)) % 360;

    float m2{1};
    if (c <= 0.5f) {
        m2 = 2 * c;
    }
    float m1 = 2 * c - m2;

    int pixel_id = coord;
    bitmap[pixel_id].x = value(m1, m2, hue + 120);
    bitmap[pixel_id].y = value(m1, m2, hue);
    bitmap[pixel_id].z = value(m1, m2, hue - 120);
    bitmap[pixel_id].w = 255;
}

void PathChecker::checkNumArgs(int argc) {
    if (argc != 2) {
        std::printf("Number of arguments must be 2.\n");
        std::exit(EXIT_FAILURE);
    }
}

void PathChecker::checkPath(int argc, char **argv) {
    checkNumArgs(argc);
    std::filesystem::path file_path(argv[1]);
    if (file_path.has_extension()) {
        std::printf("Output path [%s] invalid.\n", file_path.string().c_str());
        std::exit(EXIT_FAILURE);
    }
    std::printf("Output path: %s\n", argv[1]);
}

void PathChecker::checkFilePath(int argc, char **argv, std::string_view ext) {
    checkNumArgs(argc);
    std::filesystem::path file_path(argv[1]);
    if (file_path.extension() != ext) {
        std::printf("Output path [%s] invalid.\n", file_path.string().c_str());
        std::exit(EXIT_FAILURE);
    }
    std::printf("Output path: %s\n", argv[1]);
}

Timer::Timer() {
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
}

Timer::~Timer() {
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
}

void Timer::startTimer() {
    HANDLE_ERROR(cudaEventRecord(start, 0));
}

void Timer::stopTimer() {
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
}

float Timer::readTimer() {
    float elapsed_time_ms;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time_ms, start, stop));
    return elapsed_time_ms;
}
