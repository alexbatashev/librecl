macro(add_librecl_runtime name type)
  add_library(${name} ${type} ${ARGN})

  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/compiler
  )
  target_include_directories(${name} SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers
  )
  target_compile_definitions(${name} PRIVATE
    -DLCL_BUILD_DLL
    -DCL_TARGET_OPENCL_VERSION=300
  )
endmacro()
