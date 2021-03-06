//===- program.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "program.hpp"

_cl_program::_cl_program(cl_context ctx, std::string_view program)
    : mContext(ctx) {
  mProgramSource = SourceProgram{std::string{program}};
}

void _cl_program::build(std::span<const cl_device_id> devices,
                        std::span<std::string_view> options,
                        std::optional<callback_t> callback) {
  const auto build = [this, devices]() -> cl_int {
    lcl::FrontendResult res = std::visit(
        overloaded{[this](SourceProgram prog) {
                     return mContext->getClangFE().process(prog.source, {});
                   },
                   [](auto prog) { return lcl::FrontendResult(""); }},
        mProgramSource);

    if (!res.success()) {
      mContext->notifyError(res.error());
      return CL_BUILD_PROGRAM_FAILURE;
    }

    lcl::BinaryProgram spv = mContext->getVulkanBE().compile(res);

    vk::ShaderModuleCreateInfo shaderInfo(
        vk::ShaderModuleCreateFlags(), spv.binary.size(),
        reinterpret_cast<const uint32_t *>(spv.binary.data()));

    for (auto dev : devices) {
      mShaders.emplace(
          dev,
          std::move(dev->getLogicalDevice().createShaderModule(shaderInfo)));
    }

    for (auto kernel : spv.kernels) {
      std::vector<vk::DescriptorSetLayoutBinding> bindings;
      bindings.reserve(kernel.arguments.size());
      std::vector<KernelArgInfo::Info> info;
      info.reserve(kernel.arguments.size());
      for (auto &arg : kernel.arguments) {
        vk::DescriptorSetLayoutBinding binding{
            bindings.size(), vk::DescriptorType::eStorageBuffer, 1,
            vk::ShaderStageFlagBits::eCompute};
        bindings.push_back(binding);
        info.emplace_back(arg.type == lcl::ArgumentInfo::ArgType::GlobalBuffer,
                          arg.size);
      }

      mKernelInfo[kernel.kernelName] = KernelArgInfo{info, bindings};
    }

    return CL_SUCCESS;
  };

  if (!callback.has_value()) {
    build();
  }
}
