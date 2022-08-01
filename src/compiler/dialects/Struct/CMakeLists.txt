include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_mlir_dialect(StructOps struct)
add_mlir_doc(StructDialect StructDialect Struct/ -gen-dialect-doc)
add_mlir_doc(StructOps StructOps Struct/ -gen-op-doc)

add_mlir_dialect_library(MLIRStruct
        StructDialect.cpp
        StructOps.cpp
        StructTypes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Struct
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/..

        DEPENDS
        MLIRStructOpsIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
)

