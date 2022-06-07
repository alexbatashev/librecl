//===- VulkanSPVBackendImpl.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VulkanSPVBackendImpl.hpp"
#include "frontend.hpp"
#include "passes/llvm/FixupStructuredCFGPass.h"
#include "passes/mlir/passes.hpp"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "RawMemory/RawMemoryDialect.h"

#include <cstring>
#include <llvm/ADT/SetVector.h>
#include <memory>
#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
#include <mlir/Target/SPIRV/Serialization.h>
#include <vector>

namespace lcl {
namespace detail {
VulkanSPVBackendImpl::VulkanSPVBackendImpl(bool initializeSPV)
    : mPM(&mContext) {
  mlir::registerAllDialects(mContext);
  mlir::DialectRegistry registry;
  registry.insert<mlir::rawmem::RawMemoryDialect>();
  mContext.appendDialectRegistry(registry);
  mContext.loadAllAvailableDialects();
  mlir::registerAllPasses();

  mContext.disableMultithreading();
  mPM.enableIRPrinting();

  mPM.addPass(mlir::createCanonicalizerPass());
  mPM.addPass(createSPIRToGPUPass());
  mPM.addPass(mlir::createCanonicalizerPass());
  mPM.addNestedPass<mlir::gpu::GPUModuleOp>(createExpandOpenCLFunctionsPass());
  mPM.addPass(mlir::createInlinerPass());
  mPM.addPass(lcl::createInferPointerTypesPass());
  // This is supposed to cleanup extra reinterpret_casts
  mPM.addPass(mlir::createCanonicalizerPass());

  if (initializeSPV) {
    mPM.addPass(lcl::createGPUToSPIRVPass());
    mPM.addNestedPass<mlir::spirv::ModuleOp>(createStructureCFGPass());
    mPM.addNestedPass<mlir::spirv::ModuleOp>(
        mlir::spirv::createLowerABIAttributesPass());
    mPM.addNestedPass<mlir::spirv::ModuleOp>(
        mlir::spirv::createUpdateVersionCapabilityExtensionPass());
  }
}

void VulkanSPVBackendImpl::prepareLLVMModule(
    std::unique_ptr<llvm::Module> &module) {
  // TODO this is an unnecessary hack to avoid support of memrefs of memrefs
  // in MLIR.
  using namespace llvm;

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PassBuilder PB;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM =
      PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2);
  MPM.addPass(
      createModuleToFunctionPassAdaptor(clspv::FixupStructuredCFGPass()));

  MPM.run(*module, MAM);

  if (mHasLLVMPrinter) {
    llvm::SmallVector<char, 10000> buffer;
    llvm::BitcodeWriter writer(buffer);
    writer.writeModule(*module);
    writer.writeStrtab();
    std::span<char> res{buffer.begin(), buffer.end()};
    mLLVMIRPrinter(res);
  }
  if (mHasLLVMTextPrinter) {
    std::string res;
    llvm::raw_string_ostream os{res};
    module->print(os, nullptr);
    os.flush();
    mLLVMTextPrinter(res);
  }
}

mlir::OwningOpRef<mlir::ModuleOp> VulkanSPVBackendImpl::convertLLVMIRToMLIR(
    std::unique_ptr<llvm::Module> &module) {
  auto clone = llvm::CloneModule(*module);
  auto mlirModule = mlir::translateLLVMIRToModule(std::move(clone), &mContext);

  // TODO check result
  mPM.run(mlirModule.get());

  // TODO make conditional
  {
    std::string res;
    llvm::raw_string_ostream os{res};
    mlirModule->print(os);
    os.flush();
    mMLIRPrinter(res);
  }

  return mlirModule;
}

std::vector<unsigned char> VulkanSPVBackendImpl::convertMLIRToSPIRV(
    mlir::OwningOpRef<mlir::ModuleOp> &mlirModule) {
  llvm::SmallVector<uint32_t, 10000> binary;

  auto spvModule =
      mlirModule->lookupSymbol<mlir::spirv::ModuleOp>("__spv__ocl_program");
<<<<<<< HEAD

  // TODO check result
  mlir::spirv::serialize(spvModule, binary);
=======
  if (spvModule) {
    mlir::spirv::serialize(spvModule, binary);
  }
  // TODO return error here
>>>>>>> 3918006 (experimental pass to outline structured control flow)

  std::vector<unsigned char> resBinary;
  resBinary.resize(sizeof(uint32_t) * binary.size());
  std::memcpy(resBinary.data(), binary.data(), resBinary.size());

  { mSPVPrinter(resBinary); }
}

BinaryProgram
VulkanSPVBackendImpl::compile(std::unique_ptr<llvm::Module> module) {
  if (!module) {
    llvm::errs() << "INVALID MODULE!!!\n";
    // TODO return error instead.
    std::terminate();
  }

  prepareLLVMModule(module);
  auto mlirModule = convertLLVMIRToMLIR(module);
  auto resBinary = convertMLIRToSPIRV(mlirModule);

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
        args.emplace_back(ArgumentInfo::ArgType::GlobalBuffer, args.size(),
                          size);
      } else {
        args.emplace_back(ArgumentInfo::ArgType::POD, args.size(), size);
      }
    }

    kernels.emplace_back(name, args);
  });

  return BinaryProgram{resBinary, kernels};
}
} // namespace detail
} // namespace lcl
