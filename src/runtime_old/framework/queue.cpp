//===- queue.cpp ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/cl.h>
#include <range/v3/algorithm/fill.hpp>
#include <range/v3/algorithm/find.hpp>

#include "command.hpp"
#include "ocl_api.hpp"
#include "queue.hpp"

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

cl_int LCL_API clEnqueueWriteBuffer(cl_command_queue command_queue,
                                    cl_mem buffer, cl_bool blocking_write,
                                    size_t offset, size_t size, const void *ptr,
                                    cl_uint num_events_in_wait_list,
                                    const cl_event *event_wait_list,
                                    cl_event *event) {
  // TODO all checks from the spec
  // https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#clEnqueueWriteBuffer

  auto blocking = blocking_write ? Command::EnqueueType::Blocking
                                 : Command::EnqueueType::NonBlocking;

  // TODO pass wait list
  MemWriteBufferCommand cmd{buffer, blocking, offset, size, ptr, {}};

  command_queue->submit(cmd);

  return CL_SUCCESS;
}

cl_int LCL_API clEnqueueReadBuffer(cl_command_queue command_queue,
                                   cl_mem buffer, cl_bool blocking_read,
                                   size_t offset, size_t cb, void *ptr,
                                   cl_uint num_events_in_wait_list,
                                   const cl_event *event_wait_list,
                                   cl_event *event) {
  // TODO all checks from the spec

  auto blocking = blocking_read ? Command::EnqueueType::Blocking
                                : Command::EnqueueType::NonBlocking;

  // TODO pass wait list
  MemReadBufferCommand cmd{buffer, blocking, offset, cb, ptr, {}};

  // TODO this can be an error?
  command_queue->submit(cmd);

  return CL_SUCCESS;
}

cl_int LCL_API clEnqueueNDRangeKernel(
    cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  // TODO all checks from the spec
  // https://registry.khronos.org/OpenCL/sdk/2.2/docs/man/html/clEnqueueNDRangeKernel.html

  const auto fillArray = [](std::array<size_t, 3> &array, const size_t *cArray,
                            unsigned size, size_t defaultValue) {
    if (cArray) {
      for (unsigned i = 0; i < size; i++) {
        array[i] = cArray[i];
      }
      for (unsigned i = size; i < 3; i++) {
        array[i] = defaultValue;
      }
    } else {
      ranges::fill(array, defaultValue);
    }
  };

  ExecKernelCommand::NDRange range;
  fillArray(range.globalOffset, global_work_offset, work_dim, 0);
  fillArray(range.globalSize, global_work_size, work_dim, 1);
  fillArray(range.localSize, local_work_size, work_dim, 1);

  ExecKernelCommand cmd(
      kernel, range,
      std::span<const cl_event>{event_wait_list, num_events_in_wait_list});

  command_queue->submit(cmd);

  return CL_SUCCESS;
}

cl_int LCL_API clFinish(cl_command_queue command_queue) {
  if (!command_queue) {
    log(LogLevel::Error, "command_queue is NULL");
    return CL_INVALID_COMMAND_QUEUE;
  }

  return command_queue->finish();
}
}
