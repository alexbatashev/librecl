#include "device.hpp"

#include <iostream>

_cl_device_id::_cl_device_id(cl_platform_id plt, vk::PhysicalDevice device,
                             cl_device_type type)
    : mPlatform(plt), mDevice(device), mType(type) {

  vk::PhysicalDeviceProperties props = mDevice.getProperties();

  mDeviceName = std::string(static_cast<const char *>(props.deviceName),
                            std::strlen(props.deviceName));
}
