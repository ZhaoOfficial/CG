cmake_minimum_required(VERSION 3.16)

foreach(num RANGE 3 7 1)
    add_subdirectory(chapter${num})
endforeach()
