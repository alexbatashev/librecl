add_library(lcl_icd_metal SHARED
  metal.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../framework/platform.cpp
)

target_include_directories(lcl_icd_metal PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src/compiler
)

target_include_directories(lcl_icd_metal PRIVATE SYSTEM
  ${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers
  ${PROJECT_SOURCE_DIR}/third_party/metal-cpp
)

target_link_libraries(lcl_icd_metal PRIVATE
  "-framework Metal"
  "-framework MetalKit"
  "-framework AppKit"
  "-framework Foundation"
  "-framework QuartzCore"
)

target_compile_definitions(lcl_icd_metal PRIVATE -DCL_TARGET_OPENCL_VERSION=300)
