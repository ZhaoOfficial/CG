cmake_minimum_required(VERSION 3.16)

add_library(libcommon OBJECT common.cu)
target_include_directories(libcommon PUBLIC ${TUTORIAL_INCLUDES})
target_link_libraries(libcommon PUBLIC ${TUTORIAL_LIBRARIES})

foreach(num RANGE 3 8 1)
    add_subdirectory(chapter${num})
endforeach()
