//===- memory.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory.hpp"
#include "context.hpp"
#include "ocl_api.hpp"

#include <CL/cl.h>

extern "C" {
cl_mem LCL_API clCreateBuffer(cl_context context, cl_mem_flags flags,
                              size_t size, void *host_ptr,
                              cl_int *errcode_ret) {
  if (!context) {
    log(LogLevel::Error, "context is NULL");
    *errcode_ret = CL_INVALID_CONTEXT;
    return nullptr;
  }

  if ((flags & CL_MEM_USE_HOST_PTR) && (flags & CL_MEM_ALLOC_HOST_PTR)) {
    context->notifyError(
        "CL_MEM_USE_HOST_PTR and CL_MEM_ALLOC_HOST_PTR are mutually exclusive");
    *errcode_ret = CL_INVALID_VALUE;
    return nullptr;
  }

  if ((flags & CL_MEM_USE_HOST_PTR) && (flags & CL_MEM_COPY_HOST_PTR)) {
    context->notifyError(
        "CL_MEM_USE_HOST_PTR and CL_MEM_COPY_HOST_PTR are mutually exclusive");
    *errcode_ret = CL_INVALID_VALUE;
    return nullptr;
  }

  if (size == 0) {
    context->notifyError("Buffer size is 0");
    *errcode_ret = CL_INVALID_BUFFER_SIZE;
    return nullptr;
  }

  // TODO better error handling
  // TODO flags handling

  return new _cl_mem(context, flags, size);
}
}
