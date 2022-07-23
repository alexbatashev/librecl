//===- context.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <fmt/format.h>

#include "context.hpp"
#include "platform.hpp"

_cl_context::_cl_context(std::span<_cl_device_id *const> devices)
    : lcl::debuggable_object<_cl_context>(devices.front()->getDebugMode(),
                                          "clCreateContext"),
      mDevices(devices.begin(), devices.end()) {
  for (auto dev : mDevices) {
    VmaAllocatorCreateInfo allocatorCreateInfo = {};
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_1;
    allocatorCreateInfo.physicalDevice = dev->getNativeDevice();
    allocatorCreateInfo.device = dev->getLogicalDevice();
    allocatorCreateInfo.instance = dev->getPlatform()->getInstance();

    VmaAllocator allocator;
    vmaCreateAllocator(&allocatorCreateInfo, &allocator);
    mAllocators[dev] = allocator;
  }
}

void _cl_context::notifyError(const std::string &errMessage) {
  constexpr auto fmtString = "For context {}:\n{}";
  log(LogLevel::Error,
      fmt::format(fmtString, getDebugName().value_or("unknown context"),
                  errMessage));
  mErrCallback(errMessage.data(), nullptr, 0, mUserData);
}
