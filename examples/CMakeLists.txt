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

  get_property(examples_list GLOBAL PROPERTY LIBRECL_EXAMPLES_LIST)
  set_property(GLOBAL PROPERTY LIBRECL_EXAMPLES_LIST "${examples_list};${name}")
endfunction()

function(setup_runtime_examples postfix lib_name)
  get_property(examples_list GLOBAL PROPERTY LIBRECL_EXAMPLES_LIST)
  file(WRITE ${PROJECT_BINARY_DIR}/empty.cpp)
  foreach(example IN LISTS examples_list)
    if (NOT "${example}" STREQUAL "")
      add_executable(${example}_${postfix}
        ${PROJECT_BINARY_DIR}/empty.cpp
        $<TARGET_OBJECTS:${example}_obj>
      )
      set_target_properties(${example}_${postfix}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/examples)
      target_include_directories(${example}_${postfix} PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party/OpenCL)
      target_link_libraries(${example}_${postfix} PRIVATE ${lib_name})
    endif()
  endforeach()
endfunction()

add_librecl_example(platform_info platform_info.cpp)
add_librecl_example(vector_add vector_add.cpp)
