//===- command.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "command.hpp"
#include "context.hpp"
#include "memory.hpp"
#include "queue.hpp"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

MemWriteBufferCommand::MemWriteBufferCommand(cl_mem buffer, EnqueueType type,
                                             size_t offset, size_t size,
                                             const void *ptr,
                                             std::span<cl_event> waitList)
    : mDst(buffer), mOffset(offset), mSize(size), mSrc(ptr),
      mWaitList(waitList.begin(), waitList.end()), Command(type) {}

cl_event MemWriteBufferCommand::recordCommand(cl_command_queue queue,
                                              vk::CommandBuffer commandBuffer) {
  auto command = [=, this]() {
    // TODO wait for all events
    auto info = mDst->getAllocInfoForDevice(queue->getDevice());
    auto allocator = mDst->getContext()->getAllocators().at(queue->getDevice());

    void *mappedData;

    VkResult err = vmaMapMemory(allocator, info.allocation, &mappedData);
    if (err != VK_SUCCESS) {
      // TODO add info about VK error
      queue->getContext()->notifyError("Failed to map Vulkan buffer");
      return;
    }

    std::memcpy(reinterpret_cast<char *>(mappedData) + mOffset, mSrc, mSize);

    vmaUnmapMemory(allocator, info.allocation);
  };

  if (mEnqueueType == EnqueueType::Blocking) {
    command();
  }

  return nullptr;
}

cl_event ExecKernelCommand::recordCommand(cl_command_queue queue,
                                          vk::CommandBuffer buffer) {}
