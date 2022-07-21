//===- program.hpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "context.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <CL/cl.h>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <variant>

struct SourceProgram {
  std::string source;
};

struct MSLProgram {};

using program_source_t =
    std::variant<std::monostate, SourceProgram, MSLProgram>;

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

  std::unordered_map<cl_device_id, MTL::Library *> &getBuiltProgram() {
    return mCompiledProgram;
  }

  bool isExecutable() const { return !mCompiledProgram.empty(); }

private:
  cl_context mContext;
  program_source_t mProgramSource;
  std::unordered_map<cl_device_id, MTL::Library *> mCompiledProgram;
};
