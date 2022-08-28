// 7.3 Simulating Heat Transfer
//! Deprecated!!!
#include <functional>
#include <string>

#include "cpu_anim_bitmap.h"

constexpr int DIM = 1024;
constexpr float SPEED = 0.25f;
constexpr float MAX_TEMP = 1.0f;
constexpr float MIN_TEMP = 1e-4f;

texture<float> tex_in;
texture<float> tex_out;
texture<float> tex_const;

texture<float, 2> tex2d_in;
texture<float, 2> tex2d_out;
texture<float, 2> tex2d_const;

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
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
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

__global__ void blend2DKernel(float *dst, int x_dim, int y_dim, bool flag) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int coord = x + y * x_dim;

    // Update equation: T_{new} = T_{old} + \sum_{n\in neighbor}k\cdot(T_{n}-T_{old})
    float c, l, r, t, b;
    if (flag) {
        c = tex2D(tex2d_in, x, y);
        l = tex2D(tex2d_in, x - 1, y);
        r = tex2D(tex2d_in, x + 1, y);
        t = tex2D(tex2d_in, x, y - 1);
        b = tex2D(tex2d_in, x, y + 1);
    }
    else {
        c = tex2D(tex2d_out, x, y);
        l = tex2D(tex2d_out, x - 1, y);
        r = tex2D(tex2d_out, x + 1, y);
        t = tex2D(tex2d_out, x, y - 1);
        b = tex2D(tex2d_out, x, y + 1);
    }

    dst[coord] = c + SPEED * (l + r + t + b - 4.0f * c);
}

class DataBlock {
public:
    DataBlock(int x_dim, int y_dim) {
        HANDLE_ERROR(cudaMalloc((void**)&dev_in, x_dim * y_dim * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_out, x_dim * y_dim * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_const, x_dim * y_dim * sizeof(float)));

        HANDLE_ERROR(cudaBindTexture(0, tex_in, dev_in, x_dim * y_dim * sizeof(float)));
        HANDLE_ERROR(cudaBindTexture(0, tex_out, dev_out, x_dim * y_dim * sizeof(float)));
        HANDLE_ERROR(cudaBindTexture(0, tex_const, dev_const, x_dim * y_dim * sizeof(float)));

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        HANDLE_ERROR(cudaBindTexture2D(0, tex2d_in, dev_in, desc, x_dim, y_dim, x_dim * sizeof(float)));
        HANDLE_ERROR(cudaBindTexture2D(0, tex2d_out, dev_out, desc, x_dim, y_dim, x_dim * sizeof(float)));
        HANDLE_ERROR(cudaBindTexture2D(0, tex2d_const, dev_const, desc, x_dim, y_dim, x_dim * sizeof(float)));
    }

    ~DataBlock() {
        HANDLE_ERROR(cudaUnbindTexture(tex_in));
        HANDLE_ERROR(cudaUnbindTexture(tex_out));
        HANDLE_ERROR(cudaUnbindTexture(tex_const));

        HANDLE_ERROR(cudaUnbindTexture(tex2d_in));
        HANDLE_ERROR(cudaUnbindTexture(tex2d_out));
        HANDLE_ERROR(cudaUnbindTexture(tex2d_const));

        HANDLE_ERROR(cudaFree(dev_in));
        HANDLE_ERROR(cudaFree(dev_out));
        HANDLE_ERROR(cudaFree(dev_const));
    }

    void renderFrame(uint8_t* device_ptr, int tick, int x_dim, int y_dim);

public:
    float * dev_in;
    float * dev_out;
    float * dev_const;
    Timer timer;
    float total_time{};
    int frames{};
};

void DataBlock::renderFrame(uint8_t* device_ptr, [[maybe_unused]] int tick, int x_dim, int y_dim) {
    dim3 block_size(32, 8);
    dim3 grid_size(x_dim / 32, y_dim / 8);

    this->timer.startTimer();
    bool flag = true;
    for (std::size_t i{}; i < 90; ++i) {
        copyConstantKernel<<<grid_size, block_size>>>(this->dev_in, x_dim, y_dim);
        // blend1DKernel<<<grid_size, block_size>>>(this->dev_out, DIM, DIM, flag);
        blend2DKernel<<<grid_size, block_size>>>(this->dev_out, x_dim, y_dim, flag);
        std::swap(this->dev_in, this->dev_out);
        flag = !flag;
    }
    floatToUint8<<<grid_size, block_size>>>(device_ptr, this->dev_out, x_dim, y_dim);
    this->timer.stopTimer();

    this->total_time += this->timer.readTimer();
    ++this->frames;
    std::printf("\rAverage time per frame: %.3f ms", this->total_time / this->frames);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {
    DataBlock data(DIM, DIM);
    CPUAnimBitmap bitmap(DIM, DIM, "", nullptr);

    {
        // Initialize the constant area.
        float * temp = new float[DIM * DIM]{};
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
        HANDLE_ERROR(cudaMemcpy(data.dev_const, temp, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice));

        // Initialize the input area.
        for (std::size_t y{800}; y < DIM; ++y) {
            for (std::size_t x{0}; x < 200; ++x) {
                temp[x + y * DIM] = MAX_TEMP;
            }
        }
        HANDLE_ERROR(cudaMemcpy(data.dev_in, temp, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice));
        delete[] temp;
    }

    std::function<void (uint8_t *, int, int, int)> renderFrame = std::bind(&DataBlock::renderFrame, &data, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
    bitmap.animate(renderFrame);

    return 0;
}
