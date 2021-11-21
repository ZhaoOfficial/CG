cmake_minimum_required(VERSION 3.10)

project(PBRT)

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
include(CheckIPOSupported)
check_ipo_supported(RESULT _IsIPOSupported)
  if(_IsIPOSupported)
  message(STATUS "Turn on INTERPROCEDURAL_OPTIMIZATION")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# release version
SET(CMAKE_BUILD_TYPE "Release")

include_directories(include)

aux_source_directory(src SOURCE_DIR)

add_executable(main ${SOURCE_DIR})
