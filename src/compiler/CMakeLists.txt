add_library(lcl_compiler SHARED
  ClangFrontend.cpp
  MetalBackend.cpp
)

list(APPEND CMAKE_MODULE_PATH
  "${PROJECT_SOURCE_DIR}/third_party/llvm-project/llvm/cmake/modules")

include(AddLLVM)

llvm_map_components_to_libnames(llvm_libs all native target orcjit orcshared
  orctargetprocess executionengine support instcombine support transformutils
  runtimedyld scalaropts)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_include_directories(lcl_compiler PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CLANG_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  )
target_link_libraries(lcl_compiler PUBLIC
  ${dialect_libs}
  ${conversion_libs}
  ${llvm_libs}
  clangBasic
  clangCodeGen
  clangFrontend
  clangFrontendTool
  MLIRDLTI
  MLIRIR
  MLIRLLVMIR
  MLIRTargetLLVMIRImport
  MLIRTargetLLVMIRExport
  MLIRLLVMIRTransforms
  MLIRTranslateLib
  MLIRLLVMToLLVMIRTranslation
  MLIRToLLVMIRTranslationRegistration
  file_embed
)
