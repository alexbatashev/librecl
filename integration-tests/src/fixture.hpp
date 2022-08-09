#pragma once

#include <CL/cl.h>

#include <vector>
#include <functional>

class DeviceTest {
public:
  DeviceTest() {
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);

    std::vector<cl_platform_id> platforms;
    platforms.resize(numPlatforms);

    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    for (unsigned i = 0; i < numPlatforms; i++) {
      size_t last = devices.size();

      cl_uint numDevices = 0;
      clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
      devices.resize(last + numDevices);

      clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices,
                     devices.data() + last, nullptr);
    }

    for (auto device : devices) {
      cl_int err = 0;
      contexts.push_back(
          clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err));
    }

    for (unsigned i = 0; i < devices.size(); i++) {
      cl_int err = 0;
      queues.push_back(clCreateCommandQueueWithProperties(
          contexts[i], devices[i], nullptr, &err));
    }
  }

protected:
  void with_all(const std::function<void(cl_device_id, cl_context,
                                         cl_command_queue)> &func) {
    for (unsigned i = 0; i < devices.size(); i++) {
      func(devices[i], contexts[i], queues[i]);
    }
  }

private:
  std::vector<cl_command_queue> queues;
  std::vector<cl_context> contexts;
  std::vector<cl_device_id> devices;
};
