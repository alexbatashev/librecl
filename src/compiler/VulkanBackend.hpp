//===- VulkanBackend.hpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "backend.hpp"
#include "visibility.hpp"

#include <functional>
#include <memory>

namespace lcl {
namespace detail {
class VulkanSPVBackendImpl;
}
class LCL_COMP_EXPORT VulkanBackend : public Backend {
public:
  VulkanBackend();

  std::vector<unsigned char> compile(FrontendResult &module) final;

  ~VulkanBackend() = default;

  void setLLVMIRPrinter(std::function<void(std::span<char>)> printer);
  void setLLVMTextPrinter(std::function<void(std::string_view)> printer);
  void setMLIRPrinter(std::function<void(std::string_view)> printer);
  void setSPVPrinter(std::function<void(std::span<unsigned char>)> printer);

private:
  std::shared_ptr<detail::VulkanSPVBackendImpl> mImpl;
};
} // namespace lcl
