//===- Compiler.hpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "kernel_info.hpp"
#include "Options.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <memory>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace lcl {

class CompileResult {
public:
  explicit CompileResult(std::string error) : mResult(error) {}

  explicit CompileResult(std::unique_ptr<llvm::Module> m)
      : mResult(std::move(m)) {}

  explicit CompileResult(mlir::OwningOpRef<mlir::ModuleOp> m)
      : mResult(std::move(m)) {}

  explicit CompileResult(BinaryProgram b) : mResult(std::move(b)) {}

  bool isError() const { return std::holds_alternative<std::string>(mResult); }

  const std::string &getError() const { return std::get<std::string>(mResult); }

  bool hasLLVMIR() const {
    return std::holds_alternative<std::unique_ptr<llvm::Module>>(mResult);
  }

  const std::unique_ptr<llvm::Module> &getLLVMIR() const {
    return std::get<std::unique_ptr<llvm::Module>>(mResult);
  }

  bool hasMLIR() const {
    return std::holds_alternative<mlir::OwningOpRef<mlir::ModuleOp>>(mResult);
  }

  const mlir::OwningOpRef<mlir::ModuleOp> &getMLIR() const {
    return std::get<mlir::OwningOpRef<mlir::ModuleOp>>(mResult);
  }

  bool hasBinary() const {
    return std::holds_alternative<BinaryProgram>(mResult);
  }

  const BinaryProgram &getBinary() const {
    return std::get<BinaryProgram>(mResult);
  }

private:
  std::variant<std::string, std::unique_ptr<llvm::Module>,
               mlir::OwningOpRef<mlir::ModuleOp>, BinaryProgram>
      mResult;
};

class CompilerJob;

class Compiler {
public:
  CompileResult compile_from_mlir(std::span<const char>, const Options&);
  CompileResult compile(std::span<const char> input, const Options &options);
  CompileResult compile(std::span<CompileResult *> inputs,
                        const Options &options);

private:
  llvm::LLVMContext mLLVMContext;
  mlir::MLIRContext mMLIRContext;

  // Frontend jobs
  std::unique_ptr<CompilerJob>
  createStandardClangJob(const Options &options);
  std::unique_ptr<CompilerJob>
  createSPIRVTranslatorJob(const Options &options);

  std::unique_ptr<CompilerJob>
  createMergeMLIRJob(const Options &options);

  // Optimization jobs
  std::unique_ptr<CompilerJob>
  createOptimizeLLVMIRJob(const Options &options);
  std::unique_ptr<CompilerJob>
  createOptimizeMLIRJob(const Options &options); // Placeholder

  // Conversion jobs
  std::unique_ptr<CompilerJob>
  createConvertToMLIRJob(const Options &options);
  std::unique_ptr<CompilerJob>
  createConvertMLIRToVulkanSPIRVJob(const Options &options);
  std::unique_ptr<CompilerJob>
  createConvertMLIRToMSLJob(const Options &options);
  std::unique_ptr<CompilerJob> createConvertMLIRToPTXJob();    // Placeholder
  std::unique_ptr<CompilerJob> createConvertMLIRToAMDGPUJob(); // Placeholder
};

} // namespace lcl
