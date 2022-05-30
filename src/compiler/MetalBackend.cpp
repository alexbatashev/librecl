#include "MetalBackend.hpp"
#include "VulkanSPVBackendImpl.hpp"
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

#include <memory>
#include <spirv_msl.hpp>
#include <vector>
#include <iostream>

namespace lcl {
namespace detail {
class LCL_COMP_EXPORT MetalBackendImpl : public VulkanSPVBackendImpl {
public:
  MetalBackendImpl() : VulkanSPVBackendImpl() {}

  std::vector<unsigned char>
  compile(std::unique_ptr<llvm::Module> module) override {
    std::vector<unsigned char> spv =
        VulkanSPVBackendImpl::compile(std::move(module));
    spirv_cross::CompilerMSL mslComp(reinterpret_cast<uint32_t *>(spv.data()),
                                     spv.size() / sizeof(uint32_t));
    spirv_cross::CompilerMSL::Options mslOpts;
    mslOpts.set_msl_version(2, 2);
    mslOpts.vertex_index_type = spirv_cross::CompilerMSL::Options::IndexType::UInt32;
    mslComp.set_msl_options(mslOpts);
    std::string source = mslComp.compile();

    return std::vector<unsigned char>{
        reinterpret_cast<unsigned char *>(source.data()),
        reinterpret_cast<unsigned char *>(source.data() + source.size())};
  }
};
} // namespace detail

MetalBackend::MetalBackend()
    : mImpl(std::make_shared<detail::MetalBackendImpl>()) {}

std::vector<unsigned char> MetalBackend::compile(FrontendResult &module) {
  return mImpl->compile(std::move(module.takeModule()));
}
} // namespace lcl
