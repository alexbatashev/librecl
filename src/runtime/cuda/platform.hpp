//===- platform.hpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "device.hpp"

#include <CL/cl.h>
#include <cuda.h>

#include <vector>

struct _cl_platform_id {
  static void initialize(_cl_platform_id **platforms, unsigned &numPlatforms);

  std::string getName() const { return mName; }
  std::string getVendorName() const { return mVendorName; }

  bool isFullProfile() const { return true; }

  cl_device_type getDefaultDeviceType() const { return CL_DEVICE_TYPE_GPU; }

  std::vector<_cl_device_id> &getDevices() { return mDevices; }
  const std::vector<_cl_device_id> &getDevices() const { return mDevices; }

private:
  std::vector<_cl_device_id> mDevices;
  std::string mName = "LibreCL over NVIDIA CUDA";
  std::string mVendorName = "NVIDIA";
};
