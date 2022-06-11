# Terms for testing.
add_executable(test_vector geometry/test_vector.cpp)
target_include_directories(test_vector PUBLIC "${CMAKE_SOURCE_DIR}/include")
set_property(TARGET test_vector PROPERTY FOLDER test/geometry)

add_executable(test_ray geometry/test_ray.cpp)
target_include_directories(test_ray PUBLIC "${CMAKE_SOURCE_DIR}/include")
set_property(TARGET test_ray PROPERTY FOLDER test/geometry)

add_executable(test_aabb geometry/test_aabb.cpp)
target_include_directories(test_aabb PUBLIC "${CMAKE_SOURCE_DIR}/include")
set_property(TARGET test_aabb PROPERTY FOLDER test/geometry)
