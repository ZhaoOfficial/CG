cmake_minimum_required(VERSION 3.16)

add_library(libcommon OBJECT common.cu)
target_include_directories(libcommon PUBLIC ${CMAKE_SOURCE_DIR}/include)

foreach(num RANGE 3 8 1)
    add_subdirectory(chapter${num})
endforeach()
