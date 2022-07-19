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
