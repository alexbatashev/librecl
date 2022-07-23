#pragma once

#include "device.hpp"

#include <cuda.h>

#include <vector>

struct _cl_platform_id {
  std::vector<_cl_device_id> mDevices;
};
