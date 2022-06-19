# Terms for testing.
add_executable(test_vector geometry/test_vector.cpp)
target_include_directories(test_vector PUBLIC "${CMAKE_SOURCE_DIR}/include")

add_executable(test_ray geometry/test_ray.cpp)
target_include_directories(test_ray PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(test_ray PUBLIC geometry)

add_executable(test_aabb geometry/test_aabb.cpp)
target_include_directories(test_aabb PUBLIC "${CMAKE_SOURCE_DIR}/include")

add_executable(test_square_matrix math/test_square_matrix.cpp)
target_include_directories(test_square_matrix PUBLIC "${CMAKE_SOURCE_DIR}/include")

add_executable(test_transform transform/test_transform.cpp)
target_include_directories(test_transform PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(test_transform PUBLIC geometry transform)
