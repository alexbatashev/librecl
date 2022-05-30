//===- frontend.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "frontend.hpp"
#include "frontend_impl.hpp"

#include "llvm/IR/Module.h"

namespace lcl {
FrontendResult::FrontendResult(std::shared_ptr<detail::Module> module)
    : mModule(std::move(module)) {}

std::unique_ptr<llvm::Module> FrontendResult::takeModule() {
  return std::move(mModule->module);
}

bool FrontendResult::empty() const { return mModule->module == nullptr; }

FrontendResult::~FrontendResult() = default;
FrontendResult::FrontendResult(FrontendResult &&) = default;
} // namespace lcl
