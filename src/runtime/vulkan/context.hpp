//===- context.hpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "ClangFrontend.hpp"
#include "VulkanBackend.hpp"

#include "device.hpp"
#include "framework/debug_modes.hpp"
#include "framework/log.hpp"

#include <CL/cl.h>
#include <oneapi/tbb.h>
#include <vk_mem_alloc.h>

#include <functional>
#include <span>
#include <string>
#include <vector>

struct _cl_context : public lcl::debuggable_object<_cl_context> {
  using callback_t =
      std::function<void(const char *, const void *, size_t, void *)>;

  _cl_context(std::span<_cl_device_id *const> devices);

  void notifyError(const std::string &errMessage);

  void setNotifier(callback_t cb, void *userData) {
    mErrCallback = cb;
    mUserData = userData;
  }

  lcl::ClangFrontend &getClangFE() { return mClangFE; }
  lcl::VulkanBackend &getVulkanBE() { return mVulkanBE; }

  std::vector<_cl_device_id *> &getDevices() { return mDevices; }
  const std::vector<_cl_device_id *> &getDevices() const { return mDevices; }

  const std::unordered_map<cl_device_id, VmaAllocator> &getAllocators() const {
    return mAllocators;
  }

private:
  std::vector<_cl_device_id *> mDevices;
  callback_t mErrCallback = [](const char *, const void *, size_t, void *) {};
  void *mUserData = nullptr;

  tbb::task_arena mServiceThreadPool;

  lcl::ClangFrontend mClangFE;
  lcl::VulkanBackend mVulkanBE;

  std::unordered_map<cl_device_id, VmaAllocator> mAllocators;
};
