add_subdirectory(dialects/RawMemory)

add_library(lcl_compiler SHARED
  ClangFrontend.cpp
  MetalBackend.cpp
  passes/mlir/AIRKernelABIPass.cpp
  passes/mlir/SPIRToGPUConversion.cpp
  passes/mlir/ExpandOpenCLFunctionsPass.cpp
  passes/mlir/GPUToSPIRVPass.cpp
)

list(APPEND CMAKE_MODULE_PATH
  "${PROJECT_SOURCE_DIR}/third_party/llvm-project/llvm/cmake/modules")

include(AddLLVM)

llvm_map_components_to_libnames(llvm_libs
  aggressiveinstcombine
  #amdgpu amdgpuasmparser amdgpucodegen amdgpudesc amdgpudisassembler amdgpuinfo amdgputargetmca amdgpuutils 
  analysis
  asmparser asmprinter
  binaryformat bitreader bitstreamreader bitwriter
  cfguard
  codegen
  core
  coroutines
  coverage
  debuginfocodeview debuginfodwarf debuginfogsym debuginfomsf debuginfopdb
  # demangle
  dlltooldriver dwarflinker
  dwp
  #engine executionengine
  extensions
  #filecheck
  #frontendopenacc
  frontendopenmp
  #fuzzercli
  #fuzzmutate
  globalisel
  instcombine
  #instrumentation
  #interfacestub
  #interpreter
  ipo
  irreader
  #jitlink
  #libdriver lineeditor
  linker
  lto
  mc mca mcdisassembler
  #mcjit
  mcparser
  mirparser
  native nativecodegen
  #nvptx nvptxcodegen nvptxdesc nvptxinfo
  #objcarcopts objcopy object objectyaml
  option
  orcjit orcshared orctargetprocess
  passes
  #profiledata
  #remarks
  #runtimedyld
  scalaropts
  #selectiondag
  support
  #symbolize
  #tablegen
  target
  #textapi
  transformutils
  vectorize
  #windowsdriver windowsmanifest
  #x86 x86asmparser x86codegen x86desc x86disassembler x86info x86targetmca
  #xray
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_include_directories(lcl_compiler PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CLANG_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/dialects
  ${CMAKE_CURRENT_BINARY_DIR}/dialects
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
)
