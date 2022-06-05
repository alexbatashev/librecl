#include "device.hpp"
#include "common.hpp"
#include "platform.hpp"

#include <CL/cl.h>
#include <cuda.h>
#include <range/v3/algorithm/copy.hpp>
#include <range/v3/algorithm/count.hpp>
#include <range/v3/algorithm/count_if.hpp>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/view/counted.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>

extern "C" {
cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type,
                      cl_uint num_entries, cl_device_id *devices,
                      cl_uint *num_devices) {
  using namespace ranges;

  if (platform == nullptr) {
    // log: Not a CUDA platform
    // TODO more sophisticated platform check
    return CL_INVALID_PLATFORM;
  }
  if (!(device_type == CL_DEVICE_TYPE_CPU ||
        device_type == CL_DEVICE_TYPE_GPU ||
        device_type == CL_DEVICE_TYPE_ACCELERATOR ||
        device_type == CL_DEVICE_TYPE_CUSTOM ||
        device_type == CL_DEVICE_TYPE_DEFAULT ||
        device_type == CL_DEVICE_TYPE_ALL)) {
    // log: Unsupported device_type
    return CL_INVALID_DEVICE_TYPE;
  }

  auto isType = [device_type](const _cl_device_id &device) {
    return device.mType == device_type;
  };

  if (devices == nullptr && num_devices == nullptr) {
    // log: Both devices and num_devices are NULL
    return CL_INVALID_VALUE;
  }

  if (num_entries == 0 && devices != nullptr) {
    // log: num_entries can't be 0 when devices != NULL
    return CL_INVALID_VALUE;
  }

  if (num_devices) {
    *num_devices = ranges::count_if(platform->mDevices, isType);
  }

  if (devices) {
    unsigned deviceCount = ranges::count_if(platform->mDevices, isType);
    auto allowedDevices =
        platform->mDevices | views::filter(isType) | views::take(deviceCount) |
        views::transform([](_cl_device_id &device) { return &device; });
    ranges::copy(allowedDevices, devices);
  }

  return CL_SUCCESS;
}
}
