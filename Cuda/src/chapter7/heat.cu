// 7.3 Simulating Heat Transfer
//! Deprecated!!!
#include <string>

#include "common.h"

constexpr int DIM = 1024;
constexpr float SPEED = 0.25f;
constexpr float MAX_TEMP = 1.0f;
constexpr float MIN_TEMP = 1e-4f;

texture<float> tex_in;
texture<float> tex_out;
texture<float> tex_const;

__global__ void copyConstantKernel(float *in_ptr, int x_dim, int y_dim) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int coord = x + y * x_dim;

    float c = tex1Dfetch(tex_const, coord);
    if (c != 0.0f) {
        in_ptr[coord] = c;
    }
}

__global__ void blend1DKernel(float *dst, int x_dim, int y_dim, bool flag) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int coord = x + y * x_dim;

    int left   = (x == 0 ? coord : coord - 1);
    int right  = (x == x_dim - 1 ? coord : coord + 1);
    int top    = (y == 0 ? coord : coord - x_dim);
    int bottom = (y == y_dim - 1 ? coord : coord + x_dim);

    // Update equation: T_{new} = T_{old} + \sum_{n\in neighbor}k\cdot(T_{n}-T_{old})
    float c, l, r, t, b;
    if (flag) {
        c = tex1Dfetch(tex_in, coord);
        l = tex1Dfetch(tex_in, left);
        r = tex1Dfetch(tex_in, right);
        t = tex1Dfetch(tex_in, top);
        b = tex1Dfetch(tex_in, bottom);
    }
    else {
        c = tex1Dfetch(tex_out, coord);
        l = tex1Dfetch(tex_out, left);
        r = tex1Dfetch(tex_out, right);
        t = tex1Dfetch(tex_out, top);
        b = tex1Dfetch(tex_out, bottom);
    }

    dst[coord] = c + SPEED * (l + r + t + b - 4.0f * c);
}

class DataBlock {
public:
    DataBlock() : bitmap(DIM, DIM) {
        HANDLE_ERROR(cudaMalloc((void**)&dev_in, bitmap.numPixels()));
        HANDLE_ERROR(cudaMalloc((void**)&dev_out, bitmap.numPixels()));
        HANDLE_ERROR(cudaMalloc((void**)&dev_const, bitmap.numPixels()));

        HANDLE_ERROR(cudaBindTexture(0, tex_in, dev_in, bitmap.numPixels()));
        HANDLE_ERROR(cudaBindTexture(0, tex_out, dev_out, bitmap.numPixels()));
        HANDLE_ERROR(cudaBindTexture(0, tex_const, dev_const, bitmap.numPixels()));
    }

    ~DataBlock() {
        HANDLE_ERROR(cudaUnbindTexture(tex_in));
        HANDLE_ERROR(cudaUnbindTexture(tex_out));
        HANDLE_ERROR(cudaUnbindTexture(tex_const));

        HANDLE_ERROR(cudaFree(dev_in));
        HANDLE_ERROR(cudaFree(dev_out));
        HANDLE_ERROR(cudaFree(dev_const));
    }

    void animate();

public:
    Bitmap bitmap;
    float * dev_in;
    float * dev_out;
    float * dev_const;
    Timer timer;
    float total_time{};
    int frames{};
};

void DataBlock::animate() {
    dim3 block_size(16, 16);
    dim3 grid_size(DIM / 16, DIM / 16);

    this->timer.startTimer();
    bool flag = true;
    for (std::size_t i{}; i < 90; ++i) {
        copyConstantKernel<<<grid_size, block_size>>>(this->dev_in, DIM, DIM);
        blend1DKernel<<<grid_size, block_size>>>(this->dev_out, flag, DIM, DIM);
        std::swap(this->dev_in, this->dev_out);
        flag = !flag;
    }
    floatToUint8<<<grid_size, block_size>>>(this->bitmap.dev_bitmap, this->dev_out, DIM, DIM);
    this->timer.stopTimer();

    this->total_time += this->timer.readTimer();
    ++this->frames;
    std::printf("Average time per frame: %.3f ms\n", this->total_time / this->frames);
}

int main(int argc, char **argv) {
    PathChecker::checkPath(argc, argv);
    DataBlock data;

    {
        // Initialize the constant area.
        float * temp = new float[data.bitmap.numPixels()]{};
        for (std::size_t y{300}; y < 600; ++y) {
            for (std::size_t x{300}; x < 600; ++x) {
                temp[x + y * DIM] = MAX_TEMP;
            }
        }
        temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2.0f;
        temp[DIM * 700 + 100] = MIN_TEMP;
        temp[DIM * 300 + 300] = MIN_TEMP;
        temp[DIM * 200 + 700] = MIN_TEMP;
        for (std::size_t y{800}; y < 900; ++y) {
            for (std::size_t x{400}; x < 500; ++x) {
                temp[x + y * DIM] = MIN_TEMP;
            }
        }
        HANDLE_ERROR(cudaMemcpy(data.dev_const, temp, data.bitmap.numPixels(), cudaMemcpyHostToDevice));

        // Initialize the input area.
        for (std::size_t y{800}; y < DIM; ++y) {
            for (std::size_t x{0}; x < 200; ++x) {
                temp[x + y * DIM] = MAX_TEMP;
            }
        }
        HANDLE_ERROR(cudaMemcpy(data.dev_in, temp, data.bitmap.numPixels(), cudaMemcpyHostToDevice));
        delete[] temp;
    }

    // Start animation
    for (int tick{}; tick < 40; ++tick) {
        data.animate();
        data.bitmap.memcpyDeviceToHost();
        data.bitmap.toImage(std::string(argv[1]) + '/' + std::to_string(tick) + ".png");
    }

    return 0;
}
