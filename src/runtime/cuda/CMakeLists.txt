add_librecl_runtime(lcl_icd_cuda SHARED
  ${FRAMEWORK_SOURCES}
)

find_package(CUDAToolkit REQUIRED)

target_include_directories(lcl_icd_cuda PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../
)

target_link_libraries(lcl_icd_cuda PRIVATE
  CUDA::cuda_driver
  TBB::tbb
  range-v3
  lcl_compiler
)

setup_runtime_examples(cuda lcl_icd_cuda)

