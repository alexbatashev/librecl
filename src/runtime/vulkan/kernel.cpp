//===- kernel.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel.hpp"

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
    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(
        vk::PipelineLayoutCreateFlags(), descriptorSetLayout);
    vk::PipelineLayout pipelineLayout =
        shader.first->getLogicalDevice().createPipelineLayout(
            pipelineLayoutCreateInfo);
    vk::PipelineCache pipelineCache =
        shader.first->getLogicalDevice().createPipelineCache(
            vk::PipelineCacheCreateInfo());

    vk::PipelineShaderStageCreateInfo pipelineShaderCreateInfo(
        vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute,
        shader.second, mKernelName.c_str());
    vk::ComputePipelineCreateInfo computePipelineCreateInfo(
        vk::PipelineCreateFlags(), pipelineShaderCreateInfo, pipelineLayout);
    vk::ResultValue<vk::Pipeline> computePipeline =
        shader.first->getLogicalDevice().createComputePipeline(
            pipelineCache, computePipelineCreateInfo);
    mComputePipeline[shader.first] = computePipeline.value;
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
    if (size != sizeof(cl_mem)) {
      // TODO improve log error
      mProgram->getContext()->notifyError(
          "size is not equal to sizeof(cl_mem)");
      return CL_INVALID_ARG_SIZE;
    }
    // C-style cast is required here
    cl_mem buffer = (cl_mem)value;
    mKernelArgs[index].data = buffer;
  } else {
    if (info.info[index].size != size) {
      mProgram->getContext()->notifyError(
          "size is not equal to the expected size of kernel argument");
      return CL_INVALID_ARG_SIZE;
    }
    std::vector<unsigned char> data{
        reinterpret_cast<const unsigned char *>(value),
        reinterpret_cast<const unsigned char *>(value) + size};

    mKernelArgs[index].data = data;
  }

  return CL_SUCCESS;
}
