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

  vk::PhysicalDevice &getNativeDevice() { return mDevice; }
  const vk::PhysicalDevice &getNativeDevice() const { return mDevice; }

  vk::Device &getLogicalDevice() { return mLogicalDevice; }
  const vk::Device &getLogicalDevice() const { return mLogicalDevice; }

  uint32_t getQueueFamilyIndex() const { return mQueueFamilyIndex; }

  std::string getSupportedOptions() const { return mDeviceOptions.to_string(); }

private:
  struct Options {
    Options() = default;
    Options(vk::PhysicalDeviceFeatures features) {
      hasInt64Support = features.shaderInt64;
      hasDoubleSupport = features.shaderFloat64;
      hasInt16Support = features.shaderInt16;
    }

    std::tuple<vk::PhysicalDeviceFeatures> to_vulkan() const noexcept {
      vk::PhysicalDeviceFeatures features;
      features.shaderInt64 = hasInt64Support;
      features.shaderInt16 = hasInt16Support;
      features.shaderFloat64 = hasDoubleSupport;

      return features;
    }

    std::string to_string() const {
      std::string supportedFeautres;
      if (hasInt64Support) {
        supportedFeautres += "Int64\n";
      }
      if (hasInt16Support) {
        supportedFeautres += "Int16\n";
      }
      if (hasDoubleSupport) {
        supportedFeautres += "Float64";
      }

      return supportedFeautres;
    }

    bool hasInt64Support = false;
    bool hasDoubleSupport = false;
    bool hasInt16Support = false;
  };

  cl_platform_id mPlatform;
  vk::PhysicalDevice mDevice;
  vk::Device mLogicalDevice;

  std::string mDeviceName;
  cl_device_type mType;

  uint32_t mQueueFamilyIndex;
  Options mDeviceOptions;
};
