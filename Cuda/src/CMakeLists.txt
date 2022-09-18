add_subdirectory(inc_impl)
foreach(num RANGE 3 10 1)
    add_subdirectory(chapter${num})
endforeach()
