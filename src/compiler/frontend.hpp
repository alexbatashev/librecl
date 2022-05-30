//===- frontend.hpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "visibility.hpp"

#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace llvm {
class Module;
}

namespace lcl {
namespace detail {
struct Module;
};
class LCL_COMP_EXPORT FrontendResult {
public:
  FrontendResult(std::shared_ptr<detail::Module> module);

  FrontendResult(const FrontendResult &) = delete;
  FrontendResult(FrontendResult &&);

  FrontendResult(std::string errorString) : mError(errorString) {}

  bool success() const { return mError.empty(); }

  bool empty() const;

  std::string error() const { return mError; }

  std::unique_ptr<llvm::Module> takeModule();

  ~FrontendResult();

private:
  std::string mError;
  std::shared_ptr<detail::Module> mModule;
};

class LCL_COMP_EXPORT Frontend {
public:
  virtual FrontendResult process(std::string_view input,
                                 std::span<std::string_view> options) = 0;
  virtual ~Frontend() = default;
};
} // namespace lcl
