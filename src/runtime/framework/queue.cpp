//===- queue.cpp ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <range/v3/algorithm/find.hpp>

#include "queue.hpp"
#include "ocl_api.hpp"

#include <algorithm>
#include <iostream>

extern "C" {
cl_command_queue LCL_API clCreateCommandQueueWithProperties(
    cl_context context, cl_device_id device,
    const cl_queue_properties *properties, cl_int *errcode_ret) {
  if (context == nullptr) {
    context->notifyError("context is nullptr");
    // TODO check context has devices
    *errcode_ret = CL_INVALID_CONTEXT;
    return nullptr;
  }
  if (device == nullptr) {
    context->notifyError("device is nullptr");
    *errcode_ret = CL_INVALID_DEVICE;
    return nullptr;
  }

  auto it = ranges::find(context->getDevices(), device);
  if (it == ranges::end(context->getDevices())) {
    context->notifyError("device is not associated with context");
    *errcode_ret = CL_INVALID_DEVICE;
    return nullptr;
  }

  cl_command_queue queue = new InOrderQueue(device, context);

  // TODO handle properties

  *errcode_ret = CL_SUCCESS;

  return queue;
}
}
