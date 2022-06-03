//===- program.hpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "context.hpp"

#include <utility>

#include <CL/cl.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <variant>

struct SourceProgram {
  std::string source;
};

struct OclSPIRVProgram {};

using program_source_t =
    std::variant<std::monostate, SourceProgram, OclSPIRVProgram>;

struct _cl_program {
  struct callback_t {
    std::function<void(cl_program, void *)> callback;
    void *data;
  };

  _cl_program(cl_context ctx, std::string_view program);

  cl_context getContext() const { return mContext; }

  void build(std::span<const cl_device_id> devices,
             std::span<std::string_view> options,
             std::optional<callback_t> callback);

  bool isExecutable() const { return !mShaders.empty(); }

private:
  cl_context mContext;
  program_source_t mProgramSource;
  std::unordered_map<cl_device_id, vk::ShaderModule> mShaders;
};
