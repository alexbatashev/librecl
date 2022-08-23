//===- Options.h --------------------------------------------------*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stddef.h>

struct Options {
  bool target_vulkan_spv;
  bool target_opencl_spv;
  bool target_metal_macos;
  bool target_metal_ios;
  bool target_nvptx;
  bool target_amdgpu;
  bool print_before_mlir;
  bool print_after_mlir;
  bool print_before_llvm;
  bool print_after_llvm;
  int opt_level;
  bool mad_enable;
  bool kernel_arg_info;
  const char **other_options;
  size_t num_other_options;
};
