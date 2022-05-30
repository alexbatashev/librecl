add_librecl_tool(lcloc main.cpp)

target_link_libraries(lcloc PRIVATE lcl_compiler)

llvm_map_components_to_libnames(llvm_libs support option)

target_link_libraries(lcloc PRIVATE ${llvm_libs})
