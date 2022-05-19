#include "transformations.hpp"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "passes/passes.hpp"
#include "devicelib_air_macosx_air.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Transforms/IPO/DeadArgumentElimination.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Linker/Linker.h"

#include <memory>

namespace lcl {
IRCleanup::IRCleanup() {
  llvm::PassBuilder PB;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  MPM = PB.buildModuleSimplificationPipeline(llvm::OptimizationLevel::O2,
                                             llvm::ThinOrFullLTOPhase::None);
}

std::unique_ptr<llvm::Module>
IRCleanup::apply(std::unique_ptr<llvm::Module> module) {
  MPM.run(*module, MAM);

  return module;
}

AIRLegalize::AIRLegalize(llvm::LLVMContext &ctx) {
  // Load device library
  // TODO technically this is UB
  auto osxBuffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef{reinterpret_cast<char *>(devicelib_air_macosx_air_data),
                      devicelib_air_macosx_air_size});
  auto res = llvm::parseBitcodeFile(*osxBuffer, ctx);
  if (!res) {
    llvm::errs() << res.takeError();
  }
  mOSXLib = std::move(res.get());

  {
    // Add necessary function arguments
    llvm::PassBuilder PB;

    PB.registerModuleAnalyses(MPreMAM);
    PB.registerCGSCCAnalyses(MPreCGAM);
    PB.registerFunctionAnalyses(MPreFAM);
    PB.registerLoopAnalyses(MPreLAM);
    PB.crossRegisterProxies(MPreLAM, MPreFAM, MPreCGAM, MPreMAM);

    MPreMPM = PB.buildO0DefaultPipeline(llvm::OptimizationLevel::O0);
    MPreMPM.addPass(AIRFunctionArguments());
  }

  {
    // Remove unused arguments
    llvm::PassBuilder PB;

    PB.registerModuleAnalyses(MPostMAM);
    PB.registerCGSCCAnalyses(MPostCGAM);
    PB.registerFunctionAnalyses(MPostFAM);
    PB.registerLoopAnalyses(MPostLAM);
    PB.crossRegisterProxies(MPostLAM, MPostFAM, MPostCGAM, MPostMAM);

    MPostMPM = PB.buildO0DefaultPipeline(llvm::OptimizationLevel::O0);
    MPostMPM.addPass(llvm::createModuleToFunctionPassAdaptor(
        llvm::InferAddressSpacesPass()));
    // MPostMPM.addPass(llvm::DeadArgumentEliminationPass(true));
    MPostMPM.addPass(AIRKernelABI());
  }
}

std::unique_ptr<llvm::Module>
AIRLegalize::apply(std::unique_ptr<llvm::Module> module) {
  MPreMPM.run(*module, MPreMAM);
  auto clone = llvm::CloneModule(*mOSXLib);
  if (llvm::Linker::linkModules(*module, std::move(clone),
                                llvm::Linker::Flags::OverrideFromSrc)) {
    llvm::errs() << "error linking\n";
  }
  MPostMPM.run(*module, MPostMAM);

  return module;
}
} // namespace lcl
