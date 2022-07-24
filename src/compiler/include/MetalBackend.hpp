//===- MetalBackend.hpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "backend.hpp"
#include "kernel_info.hpp"
#include "visibility.hpp"

#include <functional>
#include <memory>

namespace lcl {
namespace detail {
class MetalBackendImpl;
}
class LCL_COMP_EXPORT MetalBackend : public Backend {
public:
  MetalBackend();

  BinaryProgram compile(FrontendResult &module) final;

  ~MetalBackend() = default;

  void setLLVMIRPrinter(std::function<void(std::span<char>)> printer);
  void setMLIRPrinter(std::function<void(std::string_view)> printer);
  void setLLVMTextPrinter(std::function<void(std::string_view)> printer);
  void setSPVPrinter(std::function<void(std::span<unsigned char>)> printer);
  void setMSLPrinter(std::function<void(std::string_view)> printer);

private:
  std::shared_ptr<detail::MetalBackendImpl> mImpl;
};
}
