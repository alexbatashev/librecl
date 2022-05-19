cmake_minimum_required (VERSION 3.17)

project(librecl VERSION 0.1 LANGUAGES C CXX)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

  option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
  option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

  find_package(MLIR REQUIRED CONFIG)
  find_package(Clang REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  message(STATUS "Using ClangConfig.cmake in: ${CLANG_CMAKE_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(AddClang)
  include(HandleLLVMOptions)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(FileEmbed)
FileEmbedSetup()

add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(third_party)
