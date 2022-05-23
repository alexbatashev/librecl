#include "MetalBackend.hpp"
#include "passes/mlir/passes.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "mlir/InitAllPasses.h"

#include <memory>
#include <vector>

namespace lcl {
namespace detail {
class MetalBackendImpl {
public:
  MetalBackendImpl() : mPM(&mContext) {
    mlir::DialectRegistry registry;
    // registry.insert<mlir::LLVM::LLVMDialect>();
    mlir::registerAllDialects(registry);
    mContext.appendDialectRegistry(registry);
    mlir::registerAllPasses();

    mPM.addPass(mlir::createCanonicalizerPass());
    mPM.addNestedPass<mlir::LLVM::LLVMFuncOp>(createAIRKernelABIPass());
  }
  std::vector<unsigned char> compile(std::unique_ptr<llvm::Module> module) {
    if (!module) {
      llvm::errs() << "INVALID MODULE!!!\n";
    }
    module->print(llvm::errs(), nullptr);

    auto clone = llvm::CloneModule(*module);
    auto mlirModule =
        mlir::translateLLVMIRToModule(std::move(clone), &mContext);

    // TODO check result
    mPM.run(mlirModule.get());

    mlirModule->dump();

    return {};
  }

private:
  mlir::MLIRContext mContext;
  mlir::PassManager mPM;
};
} // namespace detail

MetalBackend::MetalBackend()
    : mImpl(std::make_shared<detail::MetalBackendImpl>()) {}

std::vector<unsigned char>
MetalBackend::compile(std::unique_ptr<llvm::Module> module) {
  return mImpl->compile(std::move(module));
}
} // namespace lcl
