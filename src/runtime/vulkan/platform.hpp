#pragma once

#include "device.hpp"

#include <vector>

struct _cl_platform_id {
  _cl_platform_id(uint32_t vid); //  : mVendorID(vid) {}

  static void initialize(_cl_platform_id **platforms, unsigned &numPlatforms);

  std::string getName() const { return mName; }
  std::string getVendorName() const { return mVendorName; }

  bool isFullProfile() const { return true; }

  cl_device_type getDefaultDeviceType() const { return CL_DEVICE_TYPE_GPU; }

  std::vector<_cl_device_id> mDevices;
  std::string mName;
  std::string mVendorName;
};
