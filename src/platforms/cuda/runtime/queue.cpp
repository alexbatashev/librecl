#include "queue.hpp"

#include <algorithm>

extern "C" {
cl_command_queue
clCreateCommandQueueWithProperties(cl_context context, cl_device_id device,
                                   const cl_queue_properties *properties,
                                   cl_int *errcode_ret) {
  if (context == nullptr) {
    // log: context is nullptr
    // TODO check context has devices
    *errcode_ret = CL_INVALID_CONTEXT;
    return nullptr;
  }
  if (device == nullptr) {
    // log: device is nullptr
    *errcode_ret = CL_INVALID_DEVICE;
    return nullptr;
  }

  auto it =
      std::find(context->mDevices.begin(), context->mDevices.end(), device);
  if (it == context->mDevices.end()) {
    // log: device is not associated with context
    *errcode_ret = CL_INVALID_DEVICE;
    return nullptr;
  }

  cl_command_queue queue = new InOrderQueue(device, context);

  // TODO handle properties

  *errcode_ret = CL_SUCCESS;

  return queue;
}
}
