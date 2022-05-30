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

macro(add_librecl_tool name)
  add_executable(${name} ${ARGN})
  if (MSVC)
    target_compile_options(${name} PRIVATE /EHs-c- /GR-)
  else()
    target_compile_options(${name} PRIVATE -fno-exceptions -fno-rtti)
  endif()

  set_target_properties(${name}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  target_include_directories(${name} PRIVATE
    ${PROJECT_SOURCE_DIR}/src/compiler
    ${LLVM_INCLUDE_DIRS}
    ${CLANG_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
  )
endmacro()