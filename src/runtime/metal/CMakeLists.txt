add_library(lcl_icd_metal SHARED
  metal.cpp
  program.cpp
  kernel.cpp
  memory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../framework/platform.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../framework/device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../framework/context.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../framework/queue.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../framework/program.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../framework/kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../framework/memory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../framework/log.cpp
)

target_include_directories(lcl_icd_metal PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src/compiler
)

target_include_directories(lcl_icd_metal SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers
  ${PROJECT_SOURCE_DIR}/third_party/metal-cpp
)

target_link_libraries(lcl_icd_metal PRIVATE
  "-framework Metal"
  "-framework MetalKit"
  "-framework AppKit"
  "-framework Foundation"
  "-framework QuartzCore"
  TBB::tbb
  range-v3
  lcl_compiler
)

target_compile_definitions(lcl_icd_metal PRIVATE -DCL_TARGET_OPENCL_VERSION=300)

file(WRITE ${PROJECT_BINARY_DIR}/empty.cpp)
add_executable(platform_info_metal
  ${PROJECT_BINARY_DIR}/empty.cpp
  $<TARGET_OBJECTS:platform_info_obj>
)

set_target_properties(platform_info_metal
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/examples)
target_include_directories(platform_info_metal PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers)
target_link_libraries(platform_info_metal PRIVATE lcl_icd_metal)

add_executable(vector_add_metal
  ${PROJECT_BINARY_DIR}/empty.cpp
  $<TARGET_OBJECTS:vector_add_obj>
)

set_target_properties(vector_add_metal
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/examples)
target_include_directories(vector_add_metal PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers)
target_link_libraries(vector_add_metal PRIVATE lcl_icd_metal)
