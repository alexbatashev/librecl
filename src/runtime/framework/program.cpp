//===- program.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "program.hpp"
#include "ocl_api.hpp"

#include <CL/cl.h>

#include <optional>

extern "C" {
cl_program LCL_API clCreateProgramWithSource(cl_context context, cl_uint count,
                                             const char **strings,
                                             const size_t *lengths,
                                             cl_int *errcode_ret) {
  if (!context) {
    *errcode_ret = CL_INVALID_CONTEXT;
    return nullptr;
  }

  if (count == 0) {
    context->notifyError("count is zero");
    *errcode_ret = CL_INVALID_VALUE;
    return nullptr;
  }

  if (strings == nullptr) {
    context->notifyError("strings is NULL");
    *errcode_ret = CL_INVALID_VALUE;
    return nullptr;
  }

  std::string program;
  for (size_t i = 0; i < count; i++) {
    if (strings[i] == nullptr) {
      context->notifyError("strings[" + std::to_string(i) + "] is NULL");
      *errcode_ret = CL_INVALID_VALUE;
      return nullptr;
    }

    if (lengths && lengths[i] > 0) {
      program += std::string{strings[i], lengths[i]};
    } else {
      program += std::string{strings[i]};
    }
  }

  return new _cl_program(context, program);
}

cl_int LCL_API clBuildProgram(cl_program program, cl_uint num_devices,
                              const cl_device_id *device_list,
                              const char *options,
                              void(CL_CALLBACK *pfn_notify)(cl_program program,
                                                            void *user_data),
                              void *user_data) {
  if (!program) {
    // log: program is nullptr
    return CL_INVALID_PROGRAM;
  }

  // TODO check program and devices are in the same context
  // TODO check all devices are associated with the program

  // TODO check build options

  if (num_devices > 0 && device_list == nullptr) {
    program->getContext()->notifyError(
        "num_devices is > 0 and device_list is NULL");
    return CL_INVALID_VALUE;
  }
  if (pfn_notify == nullptr && user_data != nullptr) {
    program->getContext()->notifyError(
        "pfn_notify is NULL and user_data is not");
    return CL_INVALID_VALUE;
  }

  std::optional<_cl_program::callback_t> callback = std::nullopt;

  if (pfn_notify) {
    callback = _cl_program::callback_t{pfn_notify, user_data};
  }

  program->build(std::span<const cl_device_id>{device_list, num_devices}, {},
                 callback);

  return CL_SUCCESS;
}
}
