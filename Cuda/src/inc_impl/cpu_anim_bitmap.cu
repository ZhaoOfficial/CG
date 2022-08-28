#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "cpu_anim_bitmap.h"

CPUAnimBitmap::CPUAnimBitmap(int width, int height, std::string const& title, void *data)
    : AnimBitmap(width, height, title, data), cpu_bitmap(this->numBytes())
{
    HANDLE_ERROR(cudaMalloc((void**)&this->gpu_bitmap, this->numBytes()));
}

CPUAnimBitmap::~CPUAnimBitmap() {
    HANDLE_ERROR(cudaFree(this->gpu_bitmap));
}

void CPUAnimBitmap::animate(std::function<void(uint8_t*, int, int, int)> render_func) {
    int tick{};

    while (!glfwWindowShouldClose(this->window)) {
        render_func(this->gpu_bitmap, tick++, this->width, this->height);
        HANDLE_ERROR(cudaMemcpy(this->cpu_bitmap.data(), this->gpu_bitmap, this->numBytes(), cudaMemcpyDeviceToHost));
        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, this->numBytes(), this->cpu_bitmap.data(), GL_DYNAMIC_DRAW_ARB);

        process_keyboard(this->window);
        glDrawPixels(this->width, this->height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glfwPollEvents();
        glfwSwapBuffers(this->window);
    }
}

void CPUAnimBitmap::toImage(std::string const& path) {
    HANDLE_ERROR(cudaMemcpy(this->cpu_bitmap.data(), this->gpu_bitmap, this->numBytes(), cudaMemcpyDeviceToHost));
    stbi_flip_vertically_on_write(1);
    stbi_write_png(path.c_str(), this->width, this->height, 4, this->cpu_bitmap.data(), 0);
    std::printf("Image [%s] output successfully!\n", path.c_str());
}
