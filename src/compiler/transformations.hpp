#pragma once

#include "llvm/IR/Module.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/CGSCCPassManager.h"


namespace lcl {
class IRTransformation {
public:
  virtual void apply(llvm::Module &module) = 0;
};

class IRCleanup : public IRTransformation {
public:
  IRCleanup();
  void apply(llvm::Module &module) final;

private:
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  llvm::ModulePassManager MPM;
};

class AIRLegalize : public IRTransformation {
public:
  AIRLegalize();
  void apply(llvm::Module &module) final;

private:
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  llvm::ModulePassManager MPM;
};
}
