function(add_librecl_example name)
  add_library(${name}_obj OBJECT ${ARGN})
  target_include_directories(${name}_obj PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers)

  find_package(OpenCL)
  if (${OpenCL_FOUND} AND NOT APPLE)
    add_executable(${name} ${ARGN})
    set_target_properties(${name}
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/examples)
    target_include_directories(${name} PRIVATE
      ${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers
      )
    target_link_libraries(${name} PRIVATE ${OpenCL_LIBRARIES})
  endif()
endfunction()

add_librecl_example(platform_info platform_info.cpp)
add_librecl_example(vector_add vector_add.cpp)
