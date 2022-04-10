#include "passes.hpp"

#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace lcl {
llvm::PreservedAnalyses AIRKernelABI::run(llvm::Function &func,
                                          llvm::FunctionAnalysisManager &AM) {
  if (func.getCallingConv() == llvm::CallingConv::SPIR_FUNC) {
    func.setCallingConv(llvm::CallingConv::C);
  } else if (func.getCallingConv() != llvm::CallingConv::SPIR_KERNEL) {
    return llvm::PreservedAnalyses::all();
  }

  // Function is a kernel

  func.setCallingConv(llvm::CallingConv::C);

  return llvm::PreservedAnalyses::all();
}
}
