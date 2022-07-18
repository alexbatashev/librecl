//===- kernel.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel.hpp"
#include "framework/error.hpp"
#include "framework/utils.hpp"
#include "memory.hpp"

#include <cstring>
#include <iostream>
#include <variant>
#include <vulkan/vulkan.hpp>

_cl_kernel::_cl_kernel(cl_program program, const std::string &kernelName)
    : mProgram(program), mKernelName(kernelName) {

  const _cl_program::KernelArgInfo &info =
      mProgram->getKernelArgInfo(mKernelName);
  mKernelArgs.resize(info.info.size());

  vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(
      vk::DescriptorSetLayoutCreateFlags(), info.bindings);
  for (auto &shader : mProgram->getShaders()) {
    vk::DescriptorSetLayout descriptorSetLayout =
        shader.first->getLogicalDevice().createDescriptorSetLayout(
            descriptorSetLayoutCreateInfo);
    mDescriptorSetLayouts[shader.first] = descriptorSetLayout;
    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(
        vk::PipelineLayoutCreateFlags(), descriptorSetLayout);
    vk::PipelineLayout pipelineLayout =
        shader.first->getLogicalDevice().createPipelineLayout(
            pipelineLayoutCreateInfo);
    mComputePipelineLayout[shader.first] = pipelineLayout;
    vk::PipelineCache pipelineCache =
        shader.first->getLogicalDevice().createPipelineCache(
            vk::PipelineCacheCreateInfo());

    vk::PipelineShaderStageCreateInfo pipelineShaderCreateInfo(
        vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute,
        shader.second, mKernelName.c_str());
    vk::ComputePipelineCreateInfo computePipelineCreateInfo(
        vk::PipelineCreateFlags(), pipelineShaderCreateInfo, pipelineLayout);
    try {
      vk::ResultValue<vk::Pipeline> computePipeline =
          shader.first->getLogicalDevice().createComputePipeline(
              pipelineCache, computePipelineCreateInfo);
      mComputePipeline[shader.first] = computePipeline.value;
    } catch (vk::FeatureNotPresentError err) {
      throw UnsupportedFeature{err.what(), shader.first->getSupportedOptions()};
    }
  }
}

cl_int _cl_kernel::setArg(size_t index, size_t size, const void *value) {
  if (index > mKernelArgs.size()) {
    // TODO improve log error
    mProgram->getContext()->notifyError(
        "Index exceeds number of arguments for kernel {}");
    return CL_INVALID_ARG_INDEX;
  }

  const _cl_program::KernelArgInfo &info =
      mProgram->getKernelArgInfo(mKernelName);

  if (info.info[index].isBuffer) {
    std::cerr << "SETTING UP A BUFFER idx=" << index << "\n";
    if (size != sizeof(cl_mem)) {
      // TODO improve log error
      mProgram->getContext()->notifyError(
          "size is not equal to sizeof(cl_mem)");
      return CL_INVALID_ARG_SIZE;
    }
    // C-style cast is required here
    cl_mem *buffer = (cl_mem *)value;
    mKernelArgs[index].data = *buffer;
  } else {
    // FIXME this check is malfunction
    /*
    if (info.info[index].size != size) {
      mProgram->getContext()->notifyError(
          "size is not equal to the expected size of kernel argument");
      return CL_INVALID_ARG_SIZE;
    }
    */

    ImplicitMemoryBuffer implicitBuffer;
    implicitBuffer.bufferSize = size;

    // TODO only require devices, that were used to build the program
    // TODO try to unify with cl_mem in memory.cpp
    for (auto &dev : mProgram->getContext()->getDevices()) {
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
      allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
      allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
      VkBuffer buffer;
      VmaAllocation allocation;
      VmaAllocator allocator = mProgram->getContext()->getAllocators().at(dev);
      vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &allocation,
                      nullptr);
      implicitBuffer.buffers[dev] =
          ImplicitMemoryBuffer::Buffer{allocation, buffer};

      void *gpuData;
      vmaMapMemory(allocator, allocation, &gpuData);

      std::memcpy(gpuData, value, size);
      vmaUnmapMemory(allocator, allocation);
    }

    mKernelArgs[index].data = implicitBuffer;
  }

  return CL_SUCCESS;
}

vk::DescriptorSet _cl_kernel::prepareKernelArgs(cl_device_id device) {
  vk::DescriptorPoolSize poolSize{vk::DescriptorType::eStorageBuffer,
                                  static_cast<uint32_t>(mKernelArgs.size())};
  vk::DescriptorPoolCreateInfo poolCreateInfo(vk::DescriptorPoolCreateFlags(),
                                              1, poolSize);
  vk::DescriptorPool pool =
      device->getLogicalDevice().createDescriptorPool(poolCreateInfo);

  vk::DescriptorSetAllocateInfo allocateInfo{pool, 1,
                                             &mDescriptorSetLayouts.at(device)};
  const std::vector<vk::DescriptorSet> descriptorSets =
      device->getLogicalDevice().allocateDescriptorSets(allocateInfo);
  vk::DescriptorSet descriptorSet = descriptorSets.front();

  std::vector<vk::DescriptorBufferInfo> bufferInfos;
  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;

  const auto addBuffer = [&](const vk::Buffer &buffer, size_t index) {
    bufferInfos.emplace_back(buffer, 0, VK_WHOLE_SIZE);
    writeDescriptorSets.emplace_back(descriptorSet, index, 0, 1,
                                     vk::DescriptorType::eStorageBuffer,
                                     nullptr, &bufferInfos.back());
  };

  for (size_t idx = 0; idx < mKernelArgs.size(); idx++) {
    std::visit(overloaded{[&](cl_mem mem) {
                            const auto &info =
                                mem->getAllocInfoForDevice(device);
                            addBuffer(info.buffer, idx);
                          },
                          [&](const ImplicitMemoryBuffer &buffer) {
                            addBuffer(buffer.buffers.at(device).buffer, idx);
                          },
                          // TODO emit meaningful error
                          [](std::monostate) { std::terminate(); }},
               mKernelArgs[idx].data);
  }

  device->getLogicalDevice().updateDescriptorSets(writeDescriptorSets, {});

  return descriptorSet;
}
