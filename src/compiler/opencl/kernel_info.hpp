//===- kernel_info.hpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "visibility.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace lcl {
struct LCL_COMP_EXPORT ArgumentInfo {
  enum class ArgType : uint32_t { GlobalBuffer, USMPointer, Image, POD };

  ArgType type;
  uint64_t index;
  size_t size;
};

struct LCL_COMP_EXPORT KernelInfo {
  KernelInfo(std::string kernelName, std::vector<ArgumentInfo> args)
      : kernelName(std::move(kernelName)), arguments(std::move(args)) {}
  std::string kernelName;
  std::vector<ArgumentInfo> arguments;
};

struct LCL_COMP_EXPORT BinaryProgram {
  std::vector<unsigned char> binary;
  std::vector<KernelInfo> kernels;
};
} // namespace lcl
