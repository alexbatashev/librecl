add_executable(lcloc main.cpp)

target_link_libraries(lcloc PRIVATE lcl_compiler)
target_include_directories(lcloc PRIVATE
  ${PROJECT_SOURCE_DIR}/src/compiler
  ${LLVM_INCLUDE_DIRS}
  ${CLANG_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
