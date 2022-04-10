add_library(librecl SHARED
  dispatch.cpp
)

target_include_directories(librecl PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers
)

target_compile_definitions(librecl PRIVATE -DCL_TARGET_OPENCL_VERSION=300)
