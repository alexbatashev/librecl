//===- queue.cpp ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "queue.hpp"

_cl_command_queue::_cl_command_queue(cl_device_id device, cl_context context)
    : mDevice(device), mContext(context) {
  vk::CommandPoolCreateInfo commandPoolCreateInfo(
      vk::CommandPoolCreateFlags(), mDevice->getQueueFamilyIndex());
  mCommandPool =
      mDevice->getLogicalDevice().createCommandPool(commandPoolCreateInfo);
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
  }

  return *mBuffer;
}

cl_event InOrderQueue::submit(Command &cmd) {
  vk::CommandBuffer buf = getCommandBufferForThread();

  // TODO ensure order
  return cmd.recordCommand(this, buf);
}
