add_library(lcl_icd_cuda SHARED
  runtime/platform.cpp
  runtime/device.cpp
  runtime/context.cpp
  runtime/queue.cpp
  runtime/program.cpp
)

find_package(CUDAToolkit REQUIRED)

target_include_directories(lcl_icd_cuda PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers
  ${PROJECT_SOURCE_DIR}/src/compiler
)

target_link_libraries(lcl_icd_cuda PRIVATE
  CUDA::cuda_driver
  TBB::tbb
  range-v3
  lcl_compiler
)

# TODO migrate to libc++/libstdc++ ranges
if (NOT WIN32)
  target_compile_options(lcl_icd_cuda PRIVATE -fexceptions)
endif()

target_compile_definitions(lcl_icd_cuda PRIVATE -DCL_TARGET_OPENCL_VERSION=300)
