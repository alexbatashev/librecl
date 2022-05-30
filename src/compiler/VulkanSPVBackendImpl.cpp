#include "VulkanSPVBackendImpl.hpp"
#include "frontend.hpp"
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
VulkanSPVBackendImpl::VulkanSPVBackendImpl() : mPM(&mContext) {
  mlir::registerAllDialects(mContext);
  mlir::DialectRegistry registry;
  registry.insert<mlir::rawmem::RawMemoryDialect>();
  mContext.appendDialectRegistry(registry);
  mContext.loadAllAvailableDialects();
  mlir::registerAllPasses();

  mPM.addPass(mlir::createCanonicalizerPass());
  mPM.addPass(createSPIRToGPUPass());
  mPM.addPass(mlir::createCanonicalizerPass());
  mPM.addNestedPass<mlir::gpu::GPUModuleOp>(createExpandOpenCLFunctionsPass());
  mPM.addPass(mlir::createInlinerPass());
  // mPM.addPass(mlir::createConvertControlFlowToSPIRVPass());
  mPM.addPass(lcl::createGPUToSPIRVPass());
  mPM.addNestedPass<mlir::spirv::ModuleOp>(
      mlir::spirv::createLowerABIAttributesPass());
  mPM.addNestedPass<mlir::spirv::ModuleOp>(
      mlir::spirv::createUpdateVersionCapabilityExtensionPass());
}
std::vector<unsigned char>
VulkanSPVBackendImpl::compile(std::unique_ptr<llvm::Module> module) {
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
  auto mlirModule = mlir::translateLLVMIRToModule(std::move(clone), &mContext);

  // TODO check result
  mPM.run(mlirModule.get());
  mlirModule->dump();

  llvm::SmallVector<uint32_t, 10000> binary;

  /*
  auto spvModule =
      mlirModule->lookupSymbol<mlir::spirv::ModuleOp>("__spv__ocl_program");
  mlir::spirv::serialize(spvModule, binary);
*/

  llvm::SmallVector<mlir::spirv::ModuleOp, 1> spirvModules;
  mlirModule->walk(
      [&](mlir::spirv::ModuleOp op) { spirvModules.push_back(op); });

  if (spirvModules.size() == 0) {
    // todo return error
    return {};
  }

  if (spirvModules.size() > 1) {
    // todo return error
    return {};
  }

  mlir::spirv::serialize(spirvModules.front(), binary);

  std::vector<unsigned char> resBinary;
  resBinary.resize(sizeof(uint32_t) * binary.size());
  std::memcpy(resBinary.data(), binary.data(), resBinary.size());

  return resBinary;
}
} // namespace detail
} // namespace lcl
