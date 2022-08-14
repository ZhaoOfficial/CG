// 8.2 Graphics Interoperation

#include <cstdlib>
#include <cmath>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "cuda_gl_interop.h" 

#include "common.h"

void process_keyboard(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

__global__ void kernel(uchar4* device_ptr, int x_dim, int y_dim) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * x_dim;

    float fx = x / (float)x_dim - 0.5f;
    float fy = y / (float)y_dim - 0.5f;
    uint8_t green = 128 + 127 * std::sin(std::abs(fx * 100.0f) - std::abs(fy * 100.0f));

    device_ptr[offset].x = 0;
    device_ptr[offset].y = green;
    device_ptr[offset].z = 0;
    device_ptr[offset].w = 255;
}

constexpr int WINDOW_WIDTH = 1024;
constexpr int WINDOW_HEIGHT = 1024;

int main(int argc, char **argv) {

    // Choose a CUDA device.
    int device_id;
    cudaDeviceProp prop;

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 7;
    prop.minor = 5;
    HANDLE_ERROR(cudaChooseDevice(&device_id, &prop));
    std::printf("ID of CUDA device closest to 7.5: %d.\n", device_id);
    HANDLE_ERROR(cudaGLSetGLDevice(device_id));

    // Initialize a OpenGL window.
    GLFWwindow *window;
    if (!glfwInit())
        std::cerr << "Failed to init glfw.\n";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    if (!(window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "CUDA-OpenGL interoperation", nullptr, nullptr)))
        std::cerr << "Failed to create glfw window.\n";
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    // Initialize glew
    if (glewInit() != GLEW_OK)
        std::cerr << "Failed to init glew";
    std::printf("An OpenGL window has been created.\n");

    // OpenGL operations.
    // Shared data buffers that can be used for both OpenGL and CUDA.
    unsigned int VBO;
    cudaGraphicsResource *resource;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, VBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, WINDOW_WIDTH * WINDOW_HEIGHT * 4, NULL, GL_DYNAMIC_DRAW_ARB);
    // Notify the CUDA that the share of buffer.
    // In OpenGL call, we'll refer to this buffer with the handle `VBO`.
    // In CUDA runtime calls, we'll refer to this buffer with the pointer `resource`.
    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, VBO, cudaGraphicsMapFlagsNone));

    // An actual address in device memory that can be passed to our kernel.
    std::size_t size;
    uchar4* device_ptr;
    HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, 0));
    HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&device_ptr, &size, resource));

    // CUDA parts.
    dim3 block_size(32, 32);
    dim3 grid_size(WINDOW_WIDTH / 32, WINDOW_HEIGHT / 32);
    kernel<<<grid_size, block_size>>>(device_ptr, WINDOW_WIDTH, WINDOW_HEIGHT);
    // TODO
    HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));

    // Shown on window.
    std::printf("Start rendering.\n");
    while (!glfwWindowShouldClose(window)) {
        process_keyboard(window);
        glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glfwPollEvents();
        glfwSwapBuffers(window);
    }

    glDeleteBuffers(1, &VBO);
    return 0;
}
