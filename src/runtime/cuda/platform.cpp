//===- platform.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/cl.h>
#include <cuda.h>

#include "common.hpp"
#include "platform.hpp"

#include <string_view>
#include <vector>

void _cl_platform_id::initialize(_cl_platform_id **platforms,
                                 unsigned &numPlatforms) {

  LOG(cuInit(0));

  int numDevices = 0;
  LOG(cuDeviceGetCount(&numDevices));

  *platforms = new _cl_platform_id{};

  *platforms->mDevices.reserve(numDevices);

  for (int i = 0; i < numDevices; i++) {
    CUdevice device;
    CUresult res = cuDeviceGet(&device, i);
    if (res == CUDA_SUCCESS) {
      // All CUDA devices are GPUs for now
      *platforms->mDevices.emplace_back(device, CL_DEVICE_TYPE_GPU);
    }
    // TODO else log: error
  }
}
