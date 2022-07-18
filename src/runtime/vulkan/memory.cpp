//===- memory.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory.hpp"
#include "context.hpp"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

_cl_mem::_cl_mem(cl_context ctx, cl_mem_flags, size_t size)
    : mContext(ctx), mSize(size) {
  for (auto &dev : mContext->getDevices()) {
    uint32_t index = dev->getQueueFamilyIndex();
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                  nullptr,
                                  0,
                                  size,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  VK_SHARING_MODE_EXCLUSIVE,
                                  1,
                                  &index};
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocator allocator = ctx->getAllocators().at(dev);
    vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &allocation,
                    nullptr);
    mBuffers[dev] = AllocationInfo{allocation, buffer};
  }
}
