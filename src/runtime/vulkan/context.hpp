#pragma once

#include "ClangFrontend.hpp"
#include "VulkanBackend.hpp"

#include "device.hpp"

#include <CL/cl.h>
#include <oneapi/tbb.h>

#include <functional>
#include <span>
#include <string>
#include <vector>

struct _cl_context {
  using callback_t =
      std::function<void(const char *, const void *, size_t, void *)>;
  _cl_context() = default;
  _cl_context(std::span<_cl_device_id *const> devices)
      : mDevices(devices.begin(), devices.end()) {}

  void notifyError(const std::string &errMessage) {
    mErrCallback(errMessage.data(), nullptr, 0, mUserData);
  }

  void setNotifier(callback_t cb, void *userData) {
    mErrCallback = cb;
    mUserData = userData;
  }

  lcl::ClangFrontend &getClangFE() { return mClangFE; }

  std::vector<_cl_device_id *> getDevices() { return mDevices; }

private:
  std::vector<_cl_device_id *> mDevices;
  callback_t mErrCallback{};
  void *mUserData;

  tbb::task_arena mServiceThreadPool;

  lcl::ClangFrontend mClangFE;
  lcl::VulkanBackend mBE;
};
