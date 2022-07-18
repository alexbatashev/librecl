add_library(LCLPasses STATIC
  AIRKernelABIPass.cpp
  SPIRToGPUConversion.cpp
  ExpandOpenCLFunctionsPass.cpp
  GPUToSPIRVPass.cpp
  InferPointerTypesPass.cpp
  ExpandGPUBuiltins.cpp
  GPUToCppPass.cpp
  StructureCFGPass.cpp
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

list(APPEND CMAKE_MODULE_PATH
  "${PROJECT_SOURCE_DIR}/third_party/llvm-project/llvm/cmake/modules")

include(AddLLVM)

target_include_directories(LCLPasses SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  )
target_include_directories(LCLPasses PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../dialects/
  ${CMAKE_CURRENT_BINARY_DIR}/../../dialects
  )

target_link_libraries(LCLPasses PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  )
