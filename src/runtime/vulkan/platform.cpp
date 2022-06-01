#include "platform.hpp"
#include "device.hpp"

#include <vulkan/vulkan.hpp>

#include <unordered_map>

void _cl_platform_id::initialize(_cl_platform_id **platforms,
                                 unsigned &numPlatforms) {
  vk::ApplicationInfo appInfo{
      "LibreCL Vulkan Backend", // Application Name
      3,                        // Application Version
      nullptr,                  // Engine Name or nullptr
      0,                        // Engine Version
      VK_API_VERSION_1_1        // Vulkan API version
  };

  const std::vector<const char *> layers = {"VK_LAYER_KHRONOS_validation"};

  vk::InstanceCreateInfo instanceCreateInfo(vk::InstanceCreateFlags(), // Flags
                                            &appInfo,       // Application Info
                                            layers.size(),  // Layers count
                                            layers.data()); // Layers
  vk::Instance instance = vk::createInstance(instanceCreateInfo);

  std::unordered_map<uint32_t, _cl_platform_id *> tmpPlatforms;

  auto convertDeviceType = [](vk::PhysicalDeviceType type) {
    switch (type) {
    case vk::PhysicalDeviceType::eCpu:
      return CL_DEVICE_TYPE_CPU;
    case vk::PhysicalDeviceType::eIntegratedGpu:
    case vk::PhysicalDeviceType::eDiscreteGpu:
    case vk::PhysicalDeviceType::eVirtualGpu:
      return CL_DEVICE_TYPE_GPU;
    default:
      return CL_DEVICE_TYPE_CUSTOM;
    }
  };

  for (auto &dev : instance.enumeratePhysicalDevices()) {
    vk::PhysicalDeviceProperties props = dev.getProperties();
    _cl_platform_id *platform;
    if (tmpPlatforms.count(props.vendorID)) {
      platform = tmpPlatforms[props.vendorID];
    } else {
      platform = new _cl_platform_id(props.vendorID, instance);
      tmpPlatforms[props.vendorID] = platform;
    }

    platform->mDevices.emplace_back(platform, dev,
                                    convertDeviceType(props.deviceType));
  }

  size_t i = 0;
  for (auto &pair : tmpPlatforms) {
    platforms[i++] = pair.second;
  }

  numPlatforms = i;
}

_cl_platform_id::_cl_platform_id(uint32_t vid, vk::Instance instance)
    : mInstance(instance) {
  switch (vid) {
  case 4318:
    mVendorName = "NVIDIA";
    break;
  case 5045:
    mVendorName = "ARM";
    break;
  case 4098:
    mVendorName = "AMD";
    break;
  case 32902:
    mVendorName = "Intel";
    break;
  case 4203:
    mVendorName = "Apple";
    break;
  case 20803:
    mVendorName = "Qualcomm";
    break;
  default:
    mVendorName = "Unknown vendor";
  }

  mName = "LibreCL " + mVendorName;
}
