//===- device.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device.hpp"

#include <iostream>
#include <range/v3/algorithm/find_if.hpp>
#include <range/v3/iterator/operations.hpp>

_cl_device_id::_cl_device_id(cl_platform_id plt, vk::PhysicalDevice device,
                             cl_device_type type)
    : mPlatform(plt), mDevice(std::move(device)), mType(type) {
  vk::PhysicalDeviceProperties props = mDevice.getProperties();

  mDeviceName = std::string(static_cast<const char *>(props.deviceName),
                            std::strlen(props.deviceName));

  std::vector<vk::QueueFamilyProperties> queueFamilyProps =
      mDevice.getQueueFamilyProperties();
  auto it = ranges::find_if(
      queueFamilyProps, [](const vk::QueueFamilyProperties &prop) -> bool {
        return static_cast<bool>(prop.queueFlags & vk::QueueFlagBits::eCompute);
      });

  mQueueFamilyIndex = ranges::distance(ranges::begin(queueFamilyProps), it);

  vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(),
                                                  mQueueFamilyIndex, 1);
  vk::DeviceCreateInfo deviceCreateInfo(vk::DeviceCreateFlags(),
                                        deviceQueueCreateInfo);
  mLogicalDevice = mDevice.createDevice(deviceCreateInfo);
}
