#pragma once

#include "device.hpp"

#include <CL/cl.h>
#include <cuda.h>

#include <functional>
#include <span>
#include <string>
#include <vector>

struct _cl_context {
  _cl_context() = default;
  _cl_context(std::span<_cl_device_id *const> devices)
      : mDevices(devices.begin(), devices.end()) {}

  std::vector<_cl_device_id *> mDevices;
  std::function<void(const char *, const void *, size_t, void *)>
      mErrCallback{};
  void *mUserData;

  void notifyError(const std::string &errMessage) {
    mErrCallback(errMessage.data(), nullptr, 0, mUserData);
  }
};
