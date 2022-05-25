#include "MetalBackend.hpp"
#include "passes/mlir/passes.hpp"

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

#include <memory>
#include <vector>

namespace lcl {
namespace detail {
class MetalBackendImpl {
public:
  MetalBackendImpl() : mPM(&mContext) {
    mlir::registerAllDialects(mContext);
    mContext.loadAllAvailableDialects();
    mlir::registerAllPasses();

    mPM.addPass(mlir::createCanonicalizerPass());
    mPM.addPass(createSPIRToGPUPass());
    // mPM.addNestedPass<mlir::LLVM::LLVMFuncOp>(createAIRKernelABIPass());
  }
  std::vector<unsigned char> compile(std::unique_ptr<llvm::Module> module) {
    if (!module) {
      llvm::errs() << "INVALID MODULE!!!\n";
    }
    module->print(llvm::errs(), nullptr);

    {
      // TODO this is an unnecessary hack to avoid support of memrefs of memrefs
      // in MLIR.
      using namespace llvm;
      // Create the analysis managers.
      LoopAnalysisManager LAM;
      FunctionAnalysisManager FAM;
      CGSCCAnalysisManager CGAM;
      ModuleAnalysisManager MAM;

      // Create the new pass manager builder.
      // Take a look at the PassBuilder constructor parameters for more
      // customization, e.g. specifying a TargetMachine or various debugging
      // options.
      PassBuilder PB;

      // Register all the basic analyses with the managers.
      PB.registerModuleAnalyses(MAM);
      PB.registerCGSCCAnalyses(CGAM);
      PB.registerFunctionAnalyses(FAM);
      PB.registerLoopAnalyses(LAM);
      PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

      // Create the pass manager.
      // This one corresponds to a typical -O2 optimization pipeline.
      ModulePassManager MPM =
          PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2);

      // Optimize the IR!
      MPM.run(*module, MAM);
    }

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
