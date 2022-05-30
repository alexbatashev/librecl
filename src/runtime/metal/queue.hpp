//===- queue.hpp ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "context.hpp"
#include "device.hpp"

#include <CL/cl.h>
#include <vulkan/vulkan.hpp>

#include <span>

struct _cl_command_queue {
  _cl_command_queue(cl_device_id device, cl_context context)
      : mDevice(device), mContext(context) {}
  virtual cl_event submit() = 0;
  virtual bool isInOrder() const = 0;
  virtual ~_cl_command_queue() = default;

protected:
  cl_device_id mDevice;
  cl_context mContext;
};

class InOrderQueue : public _cl_command_queue {
public:
  InOrderQueue(cl_device_id device, cl_context context)
      : _cl_command_queue(device, context) {}
  bool isInOrder() const final { return false; }

  cl_event submit() final{};

private:
};
