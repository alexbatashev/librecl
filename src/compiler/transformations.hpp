#pragma once

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"

namespace lcl {
class IRTransformation {
public:
  virtual std::unique_ptr<llvm::Module> apply(std::unique_ptr<llvm::Module> module) = 0;
};

class IRCleanup : public IRTransformation {
public:
  IRCleanup();
  std::unique_ptr<llvm::Module> apply(std::unique_ptr<llvm::Module> module) final;

private:
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  llvm::ModulePassManager MPM;
};

class AIRLegalize : public IRTransformation {
public:
  AIRLegalize(llvm::LLVMContext &ctx);
  std::unique_ptr<llvm::Module> apply(std::unique_ptr<llvm::Module> module) final;

private:
  llvm::LoopAnalysisManager MPreLAM;
  llvm::FunctionAnalysisManager MPreFAM;
  llvm::CGSCCAnalysisManager MPreCGAM;
  llvm::ModuleAnalysisManager MPreMAM;
  llvm::ModulePassManager MPreMPM;
  llvm::LoopAnalysisManager MPostLAM;
  llvm::FunctionAnalysisManager MPostFAM;
  llvm::CGSCCAnalysisManager MPostCGAM;
  llvm::ModuleAnalysisManager MPostMAM;
  llvm::ModulePassManager MPostMPM;
  std::unique_ptr<llvm::Module> mOSXLib = nullptr;
};
} // namespace lcl
