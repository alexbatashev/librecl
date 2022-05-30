//===- VulkanBackend.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VulkanBackend.hpp"
#include "VulkanSPVBackendImpl.hpp"
#include "passes/mlir/passes.hpp"

#include <memory>
#include <vector>

namespace lcl {
VulkanBackend::VulkanBackend()
    : mImpl(std::make_shared<detail::VulkanSPVBackendImpl>()) {}

std::vector<unsigned char> VulkanBackend::compile(FrontendResult &module) {
  return mImpl->compile(module.takeModule());
}
void VulkanBackend::setLLVMIRPrinter(
    std::function<void(std::span<char>)> printer) {
  mImpl->setLLVMIRPrinter(printer);
}
void VulkanBackend::setMLIRPrinter(
    std::function<void(std::string_view)> printer) {
  mImpl->setMLIRPrinter(printer);
}
void VulkanBackend::setLLVMTextPrinter(
    std::function<void(std::string_view)> printer) {
  mImpl->setLLVMTextPrinter(printer);
}
void VulkanBackend::setSPVPrinter(
    std::function<void(std::span<unsigned char>)> printer) {
  mImpl->setSPVPrinter(printer);
}
} // namespace lcl
