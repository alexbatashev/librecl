add_librecl_tool(lcl-opt main.cpp)

target_link_libraries(lcl-opt PRIVATE lcl_compiler)

target_include_directories(lcl-opt PRIVATE
  ${PROJECT_SOURCE_DIR}/src/compiler/dialects
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(lcl-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRRawMemory
  LCLPasses
  )

target_include_directories(lcl-opt PRIVATE
  ${PROJECT_BINARY_DIR}/src/compiler/dialects
  )

#llvm_map_components_to_libnames(llvm_libs support option)

#target_link_libraries(lcloc PRIVATE ${llvm_libs})

