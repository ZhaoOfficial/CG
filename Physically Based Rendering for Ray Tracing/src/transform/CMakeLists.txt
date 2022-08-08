# Get the name of the folder where the CMakeLists.txt currently locates
get_filename_component(folder_name ${CMAKE_CURRENT_SOURCE_DIR} NAME_WE)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SOURCE_FILES)
add_library(${folder_name} STATIC ${SOURCE_FILES})
target_include_directories(${folder_name} PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/dependency/fmt"
)
target_link_libraries(${folder_name} PUBLIC fmt)
