#include "transformations.hpp"
#include "passes/passes.hpp"

#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"

namespace lcl {
IRCleanup::IRCleanup() {
  llvm::PassBuilder PB;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  MPM = PB.buildModuleSimplificationPipeline(llvm::OptimizationLevel::O2, llvm::ThinOrFullLTOPhase::None);
}

void IRCleanup::apply(llvm::Module &module) {
  MPM.run(module, MAM);
}

AIRLegalize::AIRLegalize() {
  llvm::PassBuilder PB;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  MPM = PB.buildO0DefaultPipeline(llvm::OptimizationLevel::O0);
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::InferAddressSpacesPass()));
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(AIRKernelABI()));
}

void AIRLegalize::apply(llvm::Module &module) {
  MPM.run(module, MAM);
}
}
