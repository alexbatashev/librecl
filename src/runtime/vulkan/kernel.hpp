//===- kernel.hpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "program.hpp"

#include <CL/cl.h>

#include <string>
#include <variant>
#include <vector>

struct _cl_kernel {
  _cl_kernel(cl_program program, const std::string &kernelName);

  cl_int setArg(size_t index, size_t size, const void *value);

private:
  struct KernelArg {
    std::variant<std::monostate, cl_mem, std::vector<unsigned char>> data;
  };

  cl_program mProgram;
  std::string mKernelName;

  std::vector<KernelArg> mKernelArgs;
  std::unordered_map<cl_device_id, vk::Pipeline> mComputePipeline;
};
