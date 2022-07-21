//===- platform.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "platform.hpp"
#include "device.hpp"
#include "framework/debug_modes.hpp"
#include "framework/log.hpp"

#include <vulkan/vulkan.hpp>

#include <range/v3/algorithm/any_of.hpp>
#include <string_view>
#include <unordered_map>

static PFN_vkCreateDebugReportCallbackEXT fpCreateDebugReportCallback;
static PFN_vkCreateDebugUtilsMessengerEXT fpCreateDebugUtilsMessenger;
static PFN_vkSetDebugUtilsObjectNameEXT fpSetDebugUtilsObjectName;

// Yes, it's a hack. Deal with it.
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugReportCallbackEXT(
    VkInstance instance, const VkDebugReportCallbackCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugReportCallbackEXT *pCallback) {
  return fpCreateDebugReportCallback(instance, pCreateInfo, pAllocator,
                                     pCallback);
}
VKAPI_ATTR VkResult VKAPI_CALL vkSetDebugUtilsObjectNameEXT(
    VkDevice device, const VkDebugUtilsObjectNameInfoEXT *pNameInfo) {
  return fpSetDebugUtilsObjectName(device, pNameInfo);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pMessenger) {
  return fpCreateDebugUtilsMessenger(instance, pCreateInfo, pAllocator,
                                     pMessenger);
}

static bool
hasInstanceExtension(const std::vector<vk::ExtensionProperties> &extensions,
                     std::string_view extName) {
  return ranges::any_of(
      extensions, [extName](vk::ExtensionProperties candidate) {
        return std::string_view{candidate.extensionName} == extName;
      });
}

static vk::Bool32 debugCallback(VkDebugReportFlagsEXT rawFlags,
                                VkDebugReportObjectTypeEXT objectType,
                                uint64_t object, size_t location,
                                int32_t messageCode, const char *layerPrefix,
                                const char *message,
                                void *userData [[maybe_unused]]) {
  vk::DebugReportFlagsEXT flags{rawFlags};
  const auto chooseLevel = [](vk::DebugReportFlagsEXT flags) {
    if (flags & vk::DebugReportFlagBitsEXT::eDebug)
      return LogLevel::Debug;
    if (flags & vk::DebugReportFlagBitsEXT::eError)
      return LogLevel::Error;
    if (flags & vk::DebugReportFlagBitsEXT::ePerformanceWarning)
      return LogLevel::Performance;
    if (flags & vk::DebugReportFlagBitsEXT::eWarning)
      return LogLevel::Warning;

    return LogLevel::Information;
  };

  LogLevel level = chooseLevel(flags);

  // TODO improve message style
  log(level, message);

  return VK_FALSE;
}

static VkBool32
debugUtilsCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                   VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                   const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                   void *pUserData) {
  const auto chooseLevel = [](vk::DebugUtilsMessageSeverityFlagsEXT flags) {
    if (flags & vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose)
      return LogLevel::Debug;
    if (flags & vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo)
      return LogLevel::Information;
    if (flags & vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
      return LogLevel::Warning;
    if (flags & vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
      return LogLevel::Error;

    return LogLevel::Information;
  };

  LogLevel level =
      chooseLevel(vk::DebugUtilsMessageSeverityFlagsEXT(messageSeverity));

  // TODO improve message style
  log(level, pCallbackData->pMessage);

  return VK_FALSE;
}

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
  std::vector<const char *> instanceExtensions;

  auto extensions = vk::enumerateInstanceExtensionProperties();

  const bool hasDebug = std::getenv("LIBRECL_DEBUG") != nullptr;
  const bool hasDebugReport =
      hasDebug &&
      hasInstanceExtension(extensions, VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
  const bool hasDebugUtils =
      hasDebug &&
      hasInstanceExtension(extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  if (hasDebugUtils) {
    instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  } else if (hasDebugReport) {
    instanceExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
  }

  vk::InstanceCreateInfo instanceCreateInfo(vk::InstanceCreateFlags(), // Flags
                                            &appInfo,       // Application Info
                                            layers.size(),  // Layers count
                                            layers.data()); // Layers
  instanceCreateInfo.enabledExtensionCount =
      static_cast<uint32_t>(instanceExtensions.size());
  instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
  vk::Instance instance = vk::createInstance(instanceCreateInfo);

  lcl::DebugMode debugMode;

  if (hasDebug)
    debugMode.set(lcl::DebugModeFlags::LogReport);

  if (hasDebugUtils) {
    debugMode.set(lcl::DebugModeFlags::CaptureObjectLocations);
    debugMode.set(lcl::DebugModeFlags::CaptureObjectNames);

    fpSetDebugUtilsObjectName =
        (PFN_vkSetDebugUtilsObjectNameEXT)instance.getProcAddr(
            "vkSetDebugUtilsObjectNameEXT");
    fpCreateDebugUtilsMessenger =
        (PFN_vkCreateDebugUtilsMessengerEXT)instance.getProcAddr(
            "vkCreateDebugUtilsMessengerEXT");

    vk::DebugUtilsMessengerCreateInfoEXT createInfo;
    createInfo.messageSeverity =
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                             vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                             vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
    createInfo.pfnUserCallback = &debugUtilsCallback;

    instance.createDebugUtilsMessengerEXT(createInfo);
  } else if (hasDebugReport) {
    debugMode.set(lcl::DebugModeFlags::CaptureObjectLocations);

    fpCreateDebugReportCallback =
        (PFN_vkCreateDebugReportCallbackEXT)instance.getProcAddr(
            "vkCreateDebugReportCallbackEXT");
    vk::DebugReportCallbackCreateInfoEXT reportInfo;
    reportInfo.flags = vk::DebugReportFlagBitsEXT::eDebug |
                       vk::DebugReportFlagBitsEXT::eWarning |
                       vk::DebugReportFlagBitsEXT::eInformation |
                       vk::DebugReportFlagBitsEXT::eError;
    reportInfo.pfnCallback = &debugCallback;

    vk::DebugReportCallbackEXT cb;
    instance.createDebugReportCallbackEXT(&reportInfo, nullptr, &cb);
  }

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
      platform = new _cl_platform_id(props.vendorID, instance, debugMode);
      tmpPlatforms[props.vendorID] = platform;
    }

    platform->mDevices.emplace_back(
        platform, dev, convertDeviceType(props.deviceType), debugMode);
  }

  size_t i = 0;
  for (auto &pair : tmpPlatforms) {
    platforms[i++] = pair.second;
  }

  numPlatforms = i;
}

_cl_platform_id::_cl_platform_id(uint32_t vid, vk::Instance instance,
                                 lcl::DebugMode debugMode)
    : lcl::debuggable_object<_cl_platform_id>(debugMode, "clGetPlatformIDs"),
      mInstance(instance) {
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
