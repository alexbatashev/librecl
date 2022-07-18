//===- kernel.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel.hpp"
#include "error.hpp"
#include "log.hpp"
#include "ocl_api.hpp"

#include <CL/cl.h>

extern "C" {
cl_kernel LCL_API clCreateKernel(cl_program program, const char *kernel_name,
                                 cl_int *errcode_ret) {
  if (!program) {
    log("program is NULL");
    *errcode_ret = CL_INVALID_PROGRAM;
    return nullptr;
  }

  if (!program->isExecutable()) {
    program->getContext()->notifyError("program is not in executable state");
    *errcode_ret = CL_INVALID_PROGRAM_EXECUTABLE;
    return nullptr;
  }

  if (!kernel_name) {
    program->getContext()->notifyError("kernel_name is NULL");
    *errcode_ret = CL_INVALID_VALUE;
    return nullptr;
  }

  cl_kernel kernel = nullptr;
  try {
    kernel = new _cl_kernel(program, kernel_name);
  } catch (const UnsupportedFeature &err) {
    program->getContext()->notifyError(std::string(err.what()) +
                                       std::string("\nSupported features:\n") +
                                       std::string(err.getSupportedFeatures()));
    *errcode_ret = CL_INVALID_PROGRAM_EXECUTABLE;
  }

  return kernel;
}

cl_int LCL_API clSetKernelArg(cl_kernel kernel, cl_uint arg_index,
                              size_t arg_size, const void *arg_value) {
  if (!kernel) {
    log("kernel is NULL");
    return CL_INVALID_KERNEL;
  }

  return kernel->setArg(arg_index, arg_size, arg_value);
}
}
