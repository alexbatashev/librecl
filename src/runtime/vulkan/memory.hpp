//===- memory.hpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/cl.h>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include <unordered_map>

struct _cl_mem {
  struct AllocationInfo {
    VmaAllocation allocation;
    vk::Buffer buffer;
  };

  _cl_mem(cl_context ctx, cl_mem_flags flags, size_t size);

  AllocationInfo getAllocInfoForDevice(cl_device_id device) const {
    return mBuffers.at(device);
  }

  cl_context getContext() { return mContext; }

private:
  cl_context mContext;
  std::unordered_map<cl_device_id, AllocationInfo> mBuffers;
};