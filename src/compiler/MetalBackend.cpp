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
#include <vector>

namespace lcl {
namespace detail {
class LCL_COMP_EXPORT MetalBackendImpl : public VulkanSPVBackendImpl {
public:
  MetalBackendImpl() : VulkanSPVBackendImpl(/*initializeSPV*/ false), mHelperPM(&(VulkanSPVBackendImpl::mContext)) {
    mHelperPM.addNestedPass<mlir::gpu::GPUModuleOp>(createAIRKernelABIPass());
    mHelperPM.addPass(createExpandGPUBuiltinsPass());
    mHelperPM.addPass(mlir::createCanonicalizerPass());
    mHelperPM.addPass(createGPUToCppPass());
  }

  BinaryProgram compile(std::unique_ptr<llvm::Module> module) override {
    VulkanSPVBackendImpl::prepareLLVMModule(module);
    auto mlirModule = VulkanSPVBackendImpl::convertLLVMIRToMLIR(module);
    // auto spv = VulkanSPVBackendImpl::convertMLIRToSPIRV(mlirModule);

    std::vector<KernelInfo> kernels;

    // TODO account for data layout types
    const auto getTypeSize = [](mlir::Type type) -> size_t {
      if (type.isa<mlir::rawmem::PointerType>()) {
        return 8;
      }
      if (type.isIntOrFloat()) {
        return type.getIntOrFloatBitWidth() / 8;
      }

      return -1;
    };

    mlirModule->walk([&kernels, &getTypeSize](mlir::gpu::GPUFuncOp func) {
      if (!func.isKernel())
        return;
      std::string name = func.getName().str();
      std::vector<ArgumentInfo> args;

      for (auto arg : func.getArgumentTypes()) {
        size_t size = getTypeSize(arg);
        if (arg.isa<mlir::rawmem::PointerType>()) {
          args.push_back(ArgumentInfo{.type = ArgumentInfo::ArgType::GlobalBuffer,
                                      .index = args.size(),
                                      .size = size});
        } else {
          args.push_back(ArgumentInfo{.type = ArgumentInfo::ArgType::POD,
                                      .index = args.size(),
                                      .size = size});
        }
      }

      kernels.emplace_back(name, args);
    });

    mHelperPM.run(mlirModule.get());

    std::string source;
    {
      llvm::raw_string_ostream mslStream{source};
      // TODO check for errors
      lcl::translateToCpp(mlirModule.get(), mslStream, false);

      mslStream.flush();
    }

    mMSLPrinter(source);

    std::vector<unsigned char> binary{
        reinterpret_cast<unsigned char *>(source.data()),
        reinterpret_cast<unsigned char *>(source.data() + source.size())};
    // TODO kernels
    return BinaryProgram{binary, kernels};
  }

  void setMSLPrinter(std::function<void(std::string_view)> printer) {
    mMSLPrinter = printer;
  }

private:
  mlir::PassManager mHelperPM;
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
