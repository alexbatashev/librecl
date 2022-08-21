include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_mlir_dialect(LibreCLOps lcl)
add_mlir_doc(LibreCLDialect LibreCLDialect LibreCL/ -gen-dialect-doc)
add_mlir_doc(LibreCLOps LibreCLOps LibreCL/ -gen-op-doc)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/../..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_mlir_dialect_library(MLIRLibreCL
        LibreCLDialect.cpp
        LibreCLOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/../..

        DEPENDS
        MLIRLibreCLOpsIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
)

