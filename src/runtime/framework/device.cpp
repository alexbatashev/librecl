#include "device.hpp"
#include "info.hpp"
#include "platform.hpp"

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
    // log: Not a LibreCL platform
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

  cl_device_type selectedType = device_type == CL_DEVICE_TYPE_DEFAULT
                                    ? platform->getDefaultDeviceType()
                                    : device_type;

  using filter_t = std::function<bool(const _cl_device_id &)>;

  const filter_t isType = [selectedType](const _cl_device_id &device) {
    return device.getDeviceType() == selectedType;
  };

  const filter_t anyType = [](const _cl_device_id &) { return true; };

  if (devices == nullptr && num_devices == nullptr) {
    // log: Both devices and num_devices are NULL
    return CL_INVALID_VALUE;
  }

  if (num_entries == 0 && devices != nullptr) {
    // log: num_entries can't be 0 when devices != NULL
    return CL_INVALID_VALUE;
  }

  filter_t filter = device_type == CL_DEVICE_TYPE_ALL ? anyType : isType;

  if (num_devices) {
    *num_devices = ranges::count_if(platform->mDevices, filter);
  }

  if (devices) {
    unsigned deviceCount = ranges::count_if(platform->mDevices, filter);
    auto allowedDevices =
        platform->mDevices | views::filter(filter) | views::take(deviceCount) |
        views::transform([](_cl_device_id &device) { return &device; });
    ranges::copy(allowedDevices, devices);
  }

  return CL_SUCCESS;
}

cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name,
                       size_t param_value_size, void *param_value,
                       size_t *param_value_size_ret) {
  if (device == nullptr) {
    // log: device is nullptr
    // TODO check platform is recognized by this plugin
    return CL_INVALID_DEVICE;
  }

  switch (param_name) {
  case CL_DEVICE_TYPE:
    return setParamValue(device->getDeviceType(), param_value_size, param_value,
                         param_value_size_ret);
  case CL_DEVICE_NAME: {
    return setParamValueStr(device->getName(), param_value_size, param_value,
                            param_value_size_ret);
  }
  }

  // log: unsupported cl_device_info
  return CL_INVALID_VALUE;
}
}
