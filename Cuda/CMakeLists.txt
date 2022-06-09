cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME CudaTutorial LANGUAGES CXX CUDA)
project(PROJECT_NAME)

enable_language(CUDA)
# 75 for GeForce RTX2060
# 86 for GeForce RTX3090
set(CMAKE_CUDA_ARCHITECTURES "75;86")

# warnings
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")
endif()

# standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# build type
set(CMAKE_BUILD_TYPE "Release")

# folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(src)

