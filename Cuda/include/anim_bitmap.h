#ifndef _ANIM_BITMAP_
#define _ANIM_BITMAP_

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "cuda_gl_interop.h"

#include "common.h"

class AnimBitmap {
public:
    int width, height;
    void *data;
    GLFWwindow* window{};
    unsigned int buffer_obj;

public:
    AnimBitmap(int width, int height, std::string const& title, void *data);
    virtual ~AnimBitmap();

    std::size_t numPixels() const;
    std::size_t numBytes() const;

public:
    // Callback functions.

    static void process_keyboard(GLFWwindow *window);
};

#endif // !_CPU_ANIM_BITMAP_
