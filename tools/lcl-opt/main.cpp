//===- main.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "RawMemory/RawMemoryDialect.h"
#include "passes/mlir/passes.hpp"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  // TODO: Register LibreCL passes here.
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return lcl::createAIRKernelABIPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return lcl::createExpandGPUBuiltinsPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return lcl::createGPUToCppPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return lcl::createStructureCFGPass();
  });

  mlir::DialectRegistry registry;
  registry.insert<mlir::rawmem::RawMemoryDialect>();
  registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "LibreCL optimizer driver\n", registry));
}
