//===- VulkanSPVBackendImpl.hpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <functional>
#include <span>
#include <string_view>

namespace lcl {
namespace detail {
class VulkanSPVBackendImpl {
public:
  VulkanSPVBackendImpl();

  virtual std::vector<unsigned char>
  compile(std::unique_ptr<llvm::Module> module);

  virtual ~VulkanSPVBackendImpl() = default;

  void setLLVMIRPrinter(std::function<void(std::span<char>)> printer) {
    mLLVMIRPrinter = printer;
  }
  void setMLIRPrinter(std::function<void(std::string_view)> printer) {
    mMLIRPrinter = printer;
  }
  void setSPVPrinter(std::function<void(std::span<unsigned char>)> printer) {
    mSPVPrinter = printer;
  }
  void setLLVMTextPrinter(std::function<void(std::string_view)> printer) {
    mLLVMTextPrinter = printer;
  }

private:
  mlir::MLIRContext mContext;
  mlir::PassManager mPM;

  std::function<void(std::span<char>)> mLLVMIRPrinter = [](std::span<char>) {};
  std::function<void(std::string_view)> mLLVMTextPrinter =
      [](std::string_view) {};
  std::function<void(std::string_view)> mMLIRPrinter = [](std::string_view) {};
  std::function<void(std::span<unsigned char>)> mSPVPrinter =
      [](std::span<unsigned char>) {};
};

} // namespace detail
} // namespace lcl
