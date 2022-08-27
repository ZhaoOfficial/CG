#ifndef _WINDOW_MANAGER_H_
#define _WINDOW_MANAGER_H_

#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

class WindowManager {
public:
    WindowManager(GLFWwindow*& window, int width, int height, std::string const& title);
    ~WindowManager();
};

#endif // !_WINDOW_MANAGER_H_
