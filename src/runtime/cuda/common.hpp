#pragma once

#include "log.hpp"
#include <cuda.h>

inline cl_int translateError(CUresult res) {
  switch (res) {
  case CUDA_SUCCESS:
    return CL_SUCCESS;

  default:
    return CL_OUT_OF_RESOURCES;
  }
}

#define CHECK(call)                                                            \
  do {                                                                         \
    CUresult res = call;                                                       \
    if (res != CUDA_SUCCESS) {                                                 \
      cl_int err = translateError(res);                                        \
      return err;                                                              \
    }                                                                          \
  } while (0)

#define LOG(call)                                                              \
  do {                                                                         \
    CUresult res = call;                                                       \
    if (res != CUDA_SUCCESS) {                                                 \
      log("Unexpected error TODO real reason");                                \
    }                                                                          \
  } while (0)
