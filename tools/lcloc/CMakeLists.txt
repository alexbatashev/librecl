add_executable(lcloc main.cpp)

target_link_libraries(lcloc PRIVATE lcl_compiler)
target_include_directories(lcloc PRIVATE ${PROJECT_SOURCE_DIR}/src/compiler)
