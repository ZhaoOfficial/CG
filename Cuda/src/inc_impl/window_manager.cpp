#include "window_manager.h"

WindowManager::WindowManager(GLFWwindow*& window, int width, int height, std::string const& title) {
    
    // Initialize glfw.
    if (!glfwInit())
        std::cerr << "Failed to init glfw.\n";

    // Create the window.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (window == nullptr)
        std::cerr << "Failed to create glfw window.\n";
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Initialize glew.
    if (glewInit() != GLEW_OK)
        std::cerr << "Failed to init glew";

    std::printf("An OpenGL window has been created.\n");
}

WindowManager::~WindowManager() { glfwTerminate(); }
