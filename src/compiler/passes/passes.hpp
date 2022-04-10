#pragma once

#include "llvm/IR/PassManager.h"

namespace lcl {
class AIRKernelABI : public llvm::PassInfoMixin<AIRKernelABI> {
public:
  llvm::PreservedAnalyses run(llvm::Function &func, llvm::FunctionAnalysisManager &AM);
};
}
