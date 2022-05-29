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

namespace lcl {
namespace detail {
class VulkanSPVBackendImpl {
public:
  VulkanSPVBackendImpl();

  virtual std::vector<unsigned char>
  compile(std::unique_ptr<llvm::Module> module);

  virtual ~VulkanSPVBackendImpl() = default;

private:
  mlir::MLIRContext mContext;
  mlir::PassManager mPM;
};

} // namespace detail
} // namespace lcl
