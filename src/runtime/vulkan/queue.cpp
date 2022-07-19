//===- queue.cpp ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "queue.hpp"

#include <CL/cl.h>

_cl_command_queue::_cl_command_queue(cl_device_id device, cl_context context)
    : mDevice(device), mContext(context) {
  vk::CommandPoolCreateInfo commandPoolCreateInfo(
      vk::CommandPoolCreateFlags(), mDevice->getQueueFamilyIndex());
  mCommandPool =
      mDevice->getLogicalDevice().createCommandPool(commandPoolCreateInfo);
}

cl_int _cl_command_queue::finish() noexcept {
  try {
    mDevice->getLogicalDevice().waitIdle();
  } catch (...) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  return CL_SUCCESS;
}

vk::CommandBuffer _cl_command_queue::getCommandBufferForThread() {
  // TODO multithreading
  if (!mBuffer.has_value()) {
    vk::CommandBufferAllocateInfo commandBufferAllocInfo(
        mCommandPool, vk::CommandBufferLevel::ePrimary, 1);
    const std::vector<vk::CommandBuffer> cmdBuffers =
        mDevice->getLogicalDevice().allocateCommandBuffers(
            commandBufferAllocInfo);
    mBuffer = cmdBuffers.front();

    vk::CommandBufferBeginInfo beginInfo(
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    mBuffer->begin(beginInfo);
  }

  return *mBuffer;
}

cl_event InOrderQueue::submit(Command &cmd) {
  vk::CommandBuffer buf = getCommandBufferForThread();

  // TODO ensure order
  cl_event evt = cmd.recordCommand(this, buf);

  // TODO should we submit here?

  return evt;
}
