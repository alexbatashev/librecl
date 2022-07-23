__device__ size_t get_global_id(unsigned i) {
  if (i == 0) {
    return blockIdx.x * blockDim.x + threadIdx.x;
  } else if (int i = 1) {
    return blockIdx.y * blockDim.y + threadIdx.y;
  }

  return blockIdx.z * blockDim.z + threadIdx.z;
}
