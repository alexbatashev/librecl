//===- device.hpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/cl.h>
#include <cuda.h>

#include <string>

struct _cl_device_id {
  _cl_device_id(cl_platform_id platform, CUdevice device, cl_device_type type)
      : mPlatform(platform), mDevice(device), mType(type) {}

  cl_platform_id getPlatform() const { return mPlatform; }
  cl_platform_id getPlatform() { return mPlatform; }

  cl_device_type getDeviceType() const { return mType; }

  std::string getName() const { return mDeviceName; }

  CUdevice &getNativeDevice() { return mDevice; }
  const CUdevice &getNativeDevice() const { return mDevice; }

private:
  cl_platform_id mPlatform;
  CUdevice mDevice;

  std::string mDeviceName;
  cl_device_type mType;
};
