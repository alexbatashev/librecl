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
#include <unordered_map>

struct _cl_kernel {
  _cl_kernel(cl_program program, const std::string &kernelName);

private:
  cl_program mProgram;
  std::string mKernelName;
};
