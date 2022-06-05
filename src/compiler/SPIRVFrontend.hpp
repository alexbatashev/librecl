//===- SPIRVFrontend.hpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "frontend.hpp"
#include "visibility.hpp"

#include <memory>
#include <span>
#include <string_view>

namespace lcl {
namespace detail {
class SPIRVFrontendImpl;
}
class LCL_COMP_EXPORT SPIRVFrontend : public Frontend {
public:
  SPIRVFrontend();

  FrontendResult process(std::string_view input,
                         std::span<std::string_view> options) final;

  ~SPIRVFrontend() = default;

private:
  std::shared_ptr<detail::SPIRVFrontendImpl> mImpl;
};
} // namespace lcl
