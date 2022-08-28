#include "gpu_anim_bitmap.h"

GPUAnimBitmap::GPUAnimBitmap(int width, int height, std::string const& title, void *data)
    : AnimBitmap(width, height, title, data)
{
    // Notify the CUDA that the share of buffer.
    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&this->cuda_resource, this->buffer_obj, cudaGraphicsMapFlagsNone));
}

void GPUAnimBitmap::animate(std::function<void(uchar4*, int, int, int)> render_func) {
    std::size_t size;
    uchar4* device_ptr;
    int tick{};

    while (!glfwWindowShouldClose(this->window)) {
        HANDLE_ERROR(cudaGraphicsMapResources(1, &this->cuda_resource, 0));
        HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&device_ptr, &size, this->cuda_resource));
        render_func(device_ptr, tick++, this->width, this->height);
        HANDLE_ERROR(cudaGraphicsUnmapResources(1, &this->cuda_resource, NULL));

        process_keyboard(this->window);
        glDrawPixels(this->width, this->height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glfwPollEvents();
        glfwSwapBuffers(this->window);
    }
}
