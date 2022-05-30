#include "queue.hpp"
#include "ocl_api.hpp"

#include <algorithm>

extern "C" {
cl_command_queue LCL_API clCreateCommandQueueWithProperties(
    cl_context context, cl_device_id device,
    const cl_queue_properties *properties, cl_int *errcode_ret) {
  if (context == nullptr) {
    // log: context is nullptr
    context->notifyError("context is nullptr");
    // TODO check context has devices
    *errcode_ret = CL_INVALID_CONTEXT;
    return nullptr;
  }
  if (device == nullptr) {
    // log: device is nullptr
    context->notifyError("device is nullptr");
    *errcode_ret = CL_INVALID_DEVICE;
    return nullptr;
  }

  auto it = std::find(context->getDevices().begin(),
                      context->getDevices().end(), device);
  if (it == context->getDevices().end()) {
    // log: device is not associated with context
    context->notifyError("device is not associated with context");
    *errcode_ret = CL_INVALID_DEVICE;
    return nullptr;
  }

  cl_command_queue queue = new InOrderQueue(device, context);

  // TODO handle properties

  *errcode_ret = CL_SUCCESS;

  return queue;
}
}
