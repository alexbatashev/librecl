find_package(TBB REQUIRED)

set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)

# add_subdirectory(runtime)
add_subdirectory(compiler)
