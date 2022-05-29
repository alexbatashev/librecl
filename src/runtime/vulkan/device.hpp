#pragma once

// #include "compiler.hpp"

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
