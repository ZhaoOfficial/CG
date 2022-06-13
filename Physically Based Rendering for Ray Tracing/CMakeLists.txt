cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME PBRT)
project(${PROJECT_NAME})

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    message("[+] OpenMP found.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Release version
SET(CMAKE_BUILD_TYPE "Release")

# Test parts
option(BUILD_TEST "Build test for debug." ON)
if(${BUILD_TEST})
    message("[+] Build test for debug.")
    enable_testing()
    add_subdirectory("test")
endif()

# Folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Make src/ as a lib
add_subdirectory(src)
