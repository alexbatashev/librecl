//===- MetalBackend.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalBackend.hpp"
#include "CppEmitter.hpp"
#include "VulkanSPVBackendImpl.hpp"
#include "passes/mlir/passes.hpp"
#include "visibility.hpp"

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

#include "RawMemory/RawMemoryDialect.h"

#include <iostream>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <spirv_msl.hpp>
#include <vector>

namespace lcl {
namespace detail {
class LCL_COMP_EXPORT MetalBackendImpl : public VulkanSPVBackendImpl {
public:
  MetalBackendImpl() : VulkanSPVBackendImpl(/*initializeSPV*/ false) {
    mPM.addNestedPass<mlir::gpu::GPUModuleOp>(createAIRKernelABIPass());
    mPM.addPass(createExpandGPUBuiltinsPass());
    mPM.addPass(mlir::createCanonicalizerPass());
    mPM.addPass(createGPUToCppPass());
  }

  BinaryProgram compile(std::unique_ptr<llvm::Module> module) override {
    VulkanSPVBackendImpl::prepareLLVMModule(module);
    auto mlirModule = VulkanSPVBackendImpl::convertLLVMIRToMLIR(module);

    std::string source;
    llvm::raw_string_ostream mslStream{source};
    // TODO check for errors
    lcl::translateToCpp(mlirModule.get(), mslStream, false);

    mslStream.flush();

    mMSLPrinter(source);

    std::vector<unsigned char> binary{
        reinterpret_cast<unsigned char *>(source.data()),
        reinterpret_cast<unsigned char *>(source.data() + source.size())};
    // TODO kernels
    return BinaryProgram{binary, {}};
  }

  void setMSLPrinter(std::function<void(std::string_view)> printer) {
    mMSLPrinter = printer;
  }

private:
  std::function<void(std::string_view)> mMSLPrinter = [](std::string_view) {};
};
} // namespace detail

MetalBackend::MetalBackend()
    : mImpl(std::make_shared<detail::MetalBackendImpl>()) {}

BinaryProgram MetalBackend::compile(FrontendResult &module) {
  return mImpl->compile(std::move(module.takeModule()));
}

void MetalBackend::setLLVMIRPrinter(
    std::function<void(std::span<char>)> printer) {
  mImpl->setLLVMIRPrinter(printer);
}
void MetalBackend::setLLVMTextPrinter(
    std::function<void(std::string_view)> printer) {
  mImpl->setLLVMTextPrinter(printer);
}
void MetalBackend::setMLIRPrinter(
    std::function<void(std::string_view)> printer) {
  mImpl->setMLIRPrinter(printer);
}
void MetalBackend::setSPVPrinter(
    std::function<void(std::span<unsigned char>)> printer) {
  mImpl->setSPVPrinter(printer);
}
void MetalBackend::setMSLPrinter(
    std::function<void(std::string_view)> printer) {
  mImpl->setMSLPrinter(printer);
}
} // namespace lcl
