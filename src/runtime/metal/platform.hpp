//===- platform.hpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "device.hpp"

#include "Foundation/NSArray.hpp"
#include "Metal/MTLDevice.hpp"
#include <CL/cl.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <string>
#include <vector>

struct _cl_platform_id {
  static void initialize(_cl_platform_id **platforms, unsigned &numPlatforms) {
    numPlatforms = 1;
    *platforms = new _cl_platform_id();
  }

  _cl_platform_id() {
    NS::Array *allDevices = MTL::CopyAllDevices();
    mDevices.reserve(allDevices->count());
    for (unsigned i = 0; i < allDevices->count(); i++) {
      mDevices.emplace_back(this,
                            static_cast<MTL::Device *>(allDevices->object(i)));
    }
  }

  std::string getName() const { return "LibreCL over Apple Metal"; }
  std::string getVendorName() const { return "Apple"; }

  bool isFullProfile() const { return true; }

  cl_device_type getDefaultDeviceType() const { return CL_DEVICE_TYPE_GPU; }

  std::vector<_cl_device_id> &getDevices() { return mDevices; }
  const std::vector<_cl_device_id> &getDevices() const { return mDevices; }

private:
  std::vector<_cl_device_id> mDevices;
};
