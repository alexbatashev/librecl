add_custom_target(cudaclc ALL
  clang --cuda-gpu-arch=sm_35 -S -Xclang -emit-llvm-bc -x cuda
    --cuda-device-only
    ${CMAKE_CURRENT_SOURCE_DIR}/index.cu
    -o ${PROJECT_BINARY_DIR}/cudaclc.bc
  )
