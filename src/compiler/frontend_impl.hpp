//===- frontend_impl.hpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "frontend.hpp"

#include "llvm/IR/Module.h"

#include <memory>

namespace lcl::detail {
struct Module {
  Module(std::unique_ptr<llvm::Module> module) : module(std::move(module)) {}

  std::unique_ptr<llvm::Module> module;
};
} // namespace lcl::detail
