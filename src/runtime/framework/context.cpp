//===- context.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "context.hpp"
#include "ocl_api.hpp"

extern "C" {
cl_context LCL_API clCreateContext(
    const cl_context_properties *properties, cl_uint num_devices,
    const cl_device_id *devices,
    void(CL_CALLBACK *pfn_notify)(const char *errinfo, const void *private_info,
                                  size_t cb, void *user_data),
    void *user_data, cl_int *errcode_ret) {
  if (properties != nullptr) {
    log(LogLevel::Error, "properties are not supported yet");
    *errcode_ret = CL_INVALID_VALUE;
    return nullptr;
  }

  if (num_devices == 0 || devices == nullptr) {
    log(LogLevel::Error, "no devices in context");
    *errcode_ret = CL_INVALID_VALUE;
    return nullptr;
  }

  _cl_context *ctx = new _cl_context{std::span(devices, num_devices)};

  if (pfn_notify) {
    ctx->setNotifier(pfn_notify, user_data);
  }

  *errcode_ret = CL_SUCCESS;

  return ctx;
}
}
