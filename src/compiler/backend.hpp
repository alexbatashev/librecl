//===- backend.hpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "frontend.hpp"
#include "visibility.hpp"

#include <vector>

namespace lcl {
class LCL_COMP_EXPORT Backend {
public:
  virtual std::vector<unsigned char> compile(FrontendResult &module) = 0;

  virtual ~Backend() = default;
};
} // namespace lcl
