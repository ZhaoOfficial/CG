#ifndef _GPU_ANIM_BITMAP_
#define _GPU_ANIM_BITMAP_

#include <functional>
#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "cuda_gl_interop.h"

#include "common.h"

class GPUAnimBitmap {
public:
    // Member variables

    int width, height;
    void *data;
    GLFWwindow* window{};
    unsigned int buffer_obj;
    cudaGraphicsResource *cuda_resource{};
    std::function<void(uchar4*, int, int, int)> render_func;

public:
    GPUAnimBitmap(int width, int height, std::string const& title, void *data);
    ~GPUAnimBitmap();

    std::size_t numPixels() const;
    std::size_t numBytes() const;

    void updateRenderFunc(std::function<void(uchar4*, int, int, int)> render_func);
    void animate();

public:
    // Callback functions.

    static void process_keyboard(GLFWwindow *window);
};

#endif // !_GPU_ANIM_BITMAP_
