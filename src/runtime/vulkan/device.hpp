//===- device.hpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/cl.h>
#include <vulkan/vulkan.hpp>

struct _cl_device_id {
  _cl_device_id(cl_platform_id plt, vk::PhysicalDevice device,
                cl_device_type type);

  cl_platform_id getPlatform() const { return mPlatform; }
  cl_platform_id getPlatform() { return mPlatform; }

  cl_device_type getDeviceType() const { return mType; }

  std::string getName() const { return mDeviceName; }

private:
  cl_platform_id mPlatform;
  vk::PhysicalDevice mDevice;

  std::string mDeviceName;
  cl_device_type mType;
};
