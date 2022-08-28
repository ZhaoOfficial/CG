#include "anim_bitmap.h"

AnimBitmap::AnimBitmap(int width, int height, std::string const& title, void *data)
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
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, this->numBytes(), NULL, GL_DYNAMIC_DRAW_ARB);
}

AnimBitmap::~AnimBitmap() {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glDeleteBuffers(1, &this->buffer_obj);
    glfwTerminate();
}

std::size_t AnimBitmap::numPixels() const { return this->width * this->height; }

std::size_t AnimBitmap::numBytes() const { return this->width * this->height * 4; }

void AnimBitmap::process_keyboard(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}
