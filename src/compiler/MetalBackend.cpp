#include "MetalBackend.hpp"
#include "VulkanSPVBackendImpl.hpp"

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

#include <memory>
#include <vector>

namespace lcl {
namespace detail {
class MetalBackendImpl : public VulkanSPVBackendImpl {
public:
  MetalBackendImpl() : VulkanSPVBackendImpl() {}

  std::vector<unsigned char>
  compile(std::unique_ptr<llvm::Module> module) override {
    return static_cast<VulkanSPVBackendImpl *>(this)->compile(
        std::move(module));
  }
};
} // namespace detail

MetalBackend::MetalBackend()
    : mImpl(std::make_shared<detail::MetalBackendImpl>()) {}

std::vector<unsigned char> MetalBackend::compile(FrontendResult &module) {
  return mImpl->compile(std::move(module.takeModule()));
}
} // namespace lcl
