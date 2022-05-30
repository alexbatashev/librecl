//===- device.hpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/cl.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <string>

struct _cl_device_id {
  _cl_device_id(cl_platform_id plt, MTL::Device *device)
      : mPlatform(plt), mDevice(device) {
    mName = std::string{
        mDevice->name()->cString(NS::StringEncoding::ASCIIStringEncoding)};
  }

  cl_device_type getDeviceType() const { return CL_DEVICE_TYPE_GPU; }

  std::string getName() const { return mName; }

  MTL::Device *getNativeDevice() { return mDevice; }
  const MTL::Device *getNativeDevice() const { return mDevice; }

private:
  cl_platform_id mPlatform;
  MTL::Device *mDevice;

  std::string mName;
};
