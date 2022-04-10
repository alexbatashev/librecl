add_library(lcl_icd_cuda SHARED
  runtime/platform.cpp
)

find_package(CUDAToolkit REQUIRED)

target_include_directories(lcl_icd_cuda PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers
)

target_link_libraries(lcl_icd_cuda PRIVATE CUDA::cuda_driver)

target_compile_definitions(lcl_icd_cuda PRIVATE -DCL_TARGET_OPENCL_VERSION=300)
