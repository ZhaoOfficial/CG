list(APPEND PBRT_INCLUDES
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/dependency/include"
)

list(APPEND PBRT_LIBARAIES
    geometry
    transform
    fmt
)
    
# Terms for testing.
add_executable(test_aabb geometry/test_aabb.cpp)
target_include_directories(test_aabb PUBLIC ${PBRT_INCLUDES})
target_link_libraries(test_aabb PUBLIC ${PBRT_LIBARAIES})

add_executable(test_ray geometry/test_ray.cpp)
target_include_directories(test_ray PUBLIC ${PBRT_INCLUDES})
target_link_libraries(test_ray PUBLIC ${PBRT_LIBARAIES})

add_executable(test_quaternion geometry/test_quaternion.cpp)
target_include_directories(test_quaternion PUBLIC ${PBRT_INCLUDES})
target_link_libraries(test_quaternion PUBLIC ${PBRT_LIBARAIES})

add_executable(test_vector geometry/test_vector.cpp)
target_include_directories(test_vector PUBLIC ${PBRT_INCLUDES})
target_link_libraries(test_vector PUBLIC ${PBRT_LIBARAIES})

add_executable(test_arithmetic math/test_arithmetic.cpp)
target_include_directories(test_arithmetic PUBLIC ${PBRT_INCLUDES})
target_link_libraries(test_arithmetic PUBLIC ${PBRT_LIBARAIES})

add_executable(test_square_matrix math/test_square_matrix.cpp)
target_include_directories(test_square_matrix PUBLIC ${PBRT_INCLUDES})
target_link_libraries(test_square_matrix PUBLIC ${PBRT_LIBARAIES})

add_executable(test_transform transform/test_transform.cpp)
target_include_directories(test_transform PUBLIC ${PBRT_INCLUDES})
target_link_libraries(test_transform PUBLIC ${PBRT_LIBARAIES})
