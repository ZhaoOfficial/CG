# get the name of the folder that the CMakeLists.txt currently locates
get_filename_component(folder_name ${CMAKE_CURRENT_SOURCE_DIR} NAME_WE)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SOURCE_FILES)
add_library(${folder_name} OBJECT ${SOURCE_FILES})
target_include_directories(${folder_name} PUBLIC ${TUTORIAL_INCLUDES})
target_link_libraries(${folder_name} PUBLIC ${TUTORIAL_LIBRARIES})
