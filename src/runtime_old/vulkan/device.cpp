//===- device.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device.hpp"
#include "framework/debug_modes.hpp"
#include "framework/log.hpp"

#include <fmt/core.h>
#include <iostream>
#include <range/v3/algorithm/any_of.hpp>
#include <range/v3/algorithm/find_if.hpp>
#include <range/v3/iterator/operations.hpp>
#include <vulkan/vulkan.hpp>

_cl_device_id::_cl_device_id(cl_platform_id plt, vk::PhysicalDevice device,
                             cl_device_type type, lcl::DebugMode mode)
    : lcl::debuggable_object<_cl_device_id>(mode, "clGetDeviceIDs"),
      mPlatform(plt), mDevice(std::move(device)), mType(type) {
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

  const float queuePriority = 1.0f;
  vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(),
                                                  mQueueFamilyIndex, 1);
  deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

  auto supportedFeatures = device.getFeatures();

  mDeviceOptions = Options(supportedFeatures);

  vk::DeviceCreateInfo deviceCreateInfo(vk::DeviceCreateFlags(),
                                        deviceQueueCreateInfo);
  auto [enabledFeatures] = mDeviceOptions.to_vulkan();
  deviceCreateInfo.pEnabledFeatures = &enabledFeatures;

  std::vector<const char *> deviceExtensions;

  deviceCreateInfo.enabledExtensionCount =
      static_cast<uint32_t>(deviceExtensions.size());
  deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
  mLogicalDevice = mDevice.createDevice(deviceCreateInfo);
}

bool _cl_device_id::hasVulkanExtension(vk::PhysicalDevice device,
                                       std::string_view extName) {
  auto extensions = device.enumerateDeviceExtensionProperties();
  return ranges::any_of(
      extensions, [extName](vk::ExtensionProperties candidate) {
        return std::string_view{candidate.extensionName} == extName;
      });
}
