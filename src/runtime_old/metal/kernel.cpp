//===- kernel.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel.hpp"

_cl_kernel::_cl_kernel(cl_program program, const std::string &kernelName)
    : mProgram(program) {
  NS::String *name = NS::String::string(
      kernelName.c_str(), NS::StringEncoding::ASCIIStringEncoding);
  for (auto &devLib : program->getBuiltProgram()) {
    mFunctions[devLib.first] = devLib.second->newFunction(name);
  }
  name->release();
}
