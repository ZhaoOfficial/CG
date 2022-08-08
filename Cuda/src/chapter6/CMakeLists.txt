cmake_minimum_required(VERSION 3.16)

# find all the .cu file
file(GLOB file_name_list RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cu")
# get the name of the folder that the CMakeLists.txt currently locates
get_filename_component(folder_name ${CMAKE_CURRENT_SOURCE_DIR} NAME_WE)

foreach(file_name IN LISTS file_name_list)
    get_filename_component(exe_name ${file_name} NAME_WE)
    add_executable(${exe_name} ${file_name})
    target_include_directories(libcommon PUBLIC ${TUTORIAL_LIBRARIES})
    target_link_libraries(${exe_name} PRIVATE libcommon)
    set_property(TARGET ${exe_name} PROPERTY FOLDER ${folder_name})
    message("[+] Find `${CMAKE_CURRENT_SOURCE_DIR}/${file_name}`, and `${exe_name}` is configured")
endforeach()
