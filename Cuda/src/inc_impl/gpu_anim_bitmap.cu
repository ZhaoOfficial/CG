#include "gpu_anim_bitmap.h"

GPUAnimBitmap::GPUAnimBitmap(int width, int height, std::string const& title, void *data)
    : width(width), height(height), data(data)
{
    // Initialize glfw
    if (!glfwInit())
        std::cerr << "Failed to init glfw.\n";

    // Create the window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    this->window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (this->window == nullptr)
        std::cerr << "Failed to create glfw window.\n";
    glfwMakeContextCurrent(this->window);
    glfwSwapInterval(1);

    // Initialize glew
    if (glewInit() != GLEW_OK)
        std::cerr << "Failed to init glew";

    std::printf("An OpenGL window has been created.\n");

    // OpenGL operations.
    // Create a shared data buffers that can be used for both OpenGL and CUDA.
    glGenBuffers(1, &this->buffer_obj);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, this->buffer_obj);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, this->width * this->height * 4, NULL, GL_DYNAMIC_DRAW_ARB);

    // Notify the CUDA that the share of buffer.
    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&this->cuda_resource, this->buffer_obj, cudaGraphicsMapFlagsNone));
}

GPUAnimBitmap::~GPUAnimBitmap() {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glDeleteBuffers(1, &this->buffer_obj);
    glfwTerminate();
}

std::size_t GPUAnimBitmap::numPixels() const { return this->width * this->height; }
std::size_t GPUAnimBitmap::numBytes() const { return this->width * this->height * 4; }

void GPUAnimBitmap::updateRenderFunc(std::function<void(uchar4*, int, int, int)> render_func) {
    this->render_func = render_func;
}

void GPUAnimBitmap::animate() {
    std::size_t size;
    uchar4* device_ptr;
    int tick{};

    while (!glfwWindowShouldClose(this->window)) {
        HANDLE_ERROR(cudaGraphicsMapResources(1, &this->cuda_resource, 0));
        HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&device_ptr, &size, this->cuda_resource));
        this->render_func(device_ptr, ++tick, this->width, this->height);
        HANDLE_ERROR(cudaGraphicsUnmapResources(1, &this->cuda_resource, NULL));

        process_keyboard(this->window);
        glDrawPixels(this->width, this->height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glfwPollEvents();
        glfwSwapBuffers(this->window);
    }
}

void GPUAnimBitmap::process_keyboard(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}
