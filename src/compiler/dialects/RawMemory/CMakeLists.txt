include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_mlir_dialect(RawMemoryOps rawmem)
add_mlir_doc(RawMemoryDialect RawMemoryDialect RawMemory/ -gen-dialect-doc)
add_mlir_doc(RawMemoryOps RawMemoryOps RawMemory/ -gen-op-doc)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_mlir_dialect_library(MLIRRawMemory
        RawMemoryDialect.cpp
        RawMemoryOps.cpp
        RawMemoryTypes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/RawMemory
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/..

        DEPENDS
        MLIRRawMemoryOpsIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
)
