#pragma once

#include "llvm/IR/PassManager.h"

namespace lcl {
class AIRKernelABI : public llvm::PassInfoMixin<AIRKernelABI> {
public:
  llvm::PreservedAnalyses run(llvm::Module &module,
                              llvm::ModuleAnalysisManager &AM);
};
class AIRFunctionArguments : public llvm::PassInfoMixin<AIRFunctionArguments> {
public:
  llvm::PreservedAnalyses run(llvm::Module &module,
                              llvm::ModuleAnalysisManager &AM);
};
} // namespace lcl
