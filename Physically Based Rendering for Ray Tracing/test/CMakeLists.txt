# Terms for testing.
add_executable(test_vector geometry/test_vector.cpp)
target_include_directories(test_vector PUBLIC "${CMAKE_SOURCE_DIR}/include")
set_property(TARGET test_vector PROPERTY FOLDER test/geometry)
