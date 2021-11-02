cmake_minimum_required(VERSION 3.10)

project(Raytracing)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# openmp
find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    message("OpenMP found.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# optimization
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

include_directories(include)

aux_source_directory(src SOURCE_DIR)

add_executable(main ${SOURCE_DIR})
