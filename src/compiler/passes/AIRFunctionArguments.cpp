#include "passes.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <iterator>
#include <unordered_map>
#include <utility>

namespace lcl {
llvm::PreservedAnalyses
AIRFunctionArguments::run(llvm::Module &module,
                          llvm::ModuleAnalysisManager &AM) {
  llvm::SmallVector<llvm::Function *, 16> oldFunctions;
  for (llvm::Function &f : module.functions()) {
    oldFunctions.push_back(&f);
  }

  std::unordered_map<llvm::Function*, llvm::Function*> replacements;
  for (llvm::Function *f : oldFunctions) {
    std::string oldName = f->getName().str();

    if (f->getName().startswith("llvm.")) {
      continue;
    }

    // Rename function before removing
    f->setName(oldName + "_alias");

    llvm::SmallVector<llvm::Type *, 6> newArgs(
        f->getFunctionType()->params().begin(),
        f->getFunctionType()->params().end());
    llvm::Type *i32 = llvm::Type::getInt32Ty(module.getContext());
    llvm::Type *vecType =
        llvm::VectorType::get(i32, llvm::ElementCount::getFixed(3));
    newArgs.push_back(vecType);
    newArgs.push_back(vecType);
    newArgs.push_back(vecType);
    newArgs.push_back(vecType);

    llvm::FunctionType *newFuncType =
        llvm::FunctionType::get(f->getReturnType(), newArgs, false);
    llvm::Function *newFunc =
        llvm::Function::Create(newFuncType, f->getLinkage(), oldName, module);

    llvm::ValueToValueMapTy vmap;
    for (auto arg : llvm::enumerate(f->args())) {
      vmap[&arg.value()] = newFunc->getArg(arg.index());
    }

    llvm::SmallVector<llvm::ReturnInst *, 8> Returns;
    llvm::CloneFunctionInto(newFunc, f, vmap,
                            llvm::CloneFunctionChangeType::LocalChangesOnly,
                            Returns);
    if (f->getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
      llvm::MDNode *md = llvm::MDNode::get(
          newFunc->getContext(),
          llvm::MDString::get(newFunc->getContext(), "opencl.kernel"));
      newFunc->addMetadata("opencl.kernel", *md);
    }

    auto specialArgBegin = std::prev(newFunc->arg_end(), 4);

    {
      llvm::AttrBuilder gridBuilder{newFunc->getContext()};
      auto threadInGrid = specialArgBegin;
      gridBuilder.addAttribute("opencl.arg_type", "thread_position_in_grid");
      threadInGrid->addAttrs(gridBuilder);

      auto threadInGroup = std::next(threadInGrid);
      llvm::AttrBuilder groupBuilder{newFunc->getContext()};
      groupBuilder.addAttribute("opencl.arg_type",
                                "thread_position_in_threadgroup");
      threadInGroup->addAttrs(groupBuilder);

      auto threadGroupInGrid = std::next(threadInGroup);
      llvm::AttrBuilder tgInGridBuilder{newFunc->getContext()};
      tgInGridBuilder.addAttribute("opencl.arg_type",
                                   "threadgroup_position_in_grid");
      threadGroupInGrid->addAttrs(tgInGridBuilder);

      auto threadsPerGroup = std::next(threadGroupInGrid);
      llvm::AttrBuilder tPerGroupBuilder{newFunc->getContext()};
      tPerGroupBuilder.addAttribute("opencl.arg_type",
                                    "threads_per_threadgroup");
      threadsPerGroup->addAttrs(tPerGroupBuilder);
    }
    
    replacements.insert({f, newFunc});
  }

  for (auto &p : replacements) {
    llvm::Function *f = p.first;
    llvm::Function *newFunc = p.second;

    auto *newFuncType = newFunc->getFunctionType();

    llvm::SmallVector<llvm::CallInst *, 8> calls;
    for (auto &replaced : replacements) {
      for (auto &bb : *replaced.second) {
        for (auto &inst : bb) {
          if (inst.getOpcode() == llvm::Instruction::Call) {
            auto &callInst = static_cast<llvm::CallInst &>(inst);
            if (callInst.getCalledFunction() == f) {
              calls.push_back(&callInst);
            }
          }
        }
      }
    }
    for (auto callInst : calls) {
      assert(callInst->getCalledFunction());
      llvm::SmallVector<llvm::Value *, 8> newCallArgs{callInst->args()};
      auto specialArgBegin = std::prev(callInst->getParent()->getParent()->arg_end(), 4);
      for (auto it = specialArgBegin; it != callInst->getParent()->getParent()->arg_end(); ++it) {
        newCallArgs.push_back(&*it);
      }
      llvm::CallInst *newCall = llvm::CallInst::Create(
          newFuncType, newFunc, newCallArgs, newFunc->getName(), callInst);
      if (!newFunc->getReturnType()->isVoidTy()) {
        callInst->replaceAllUsesWith(newCall);
      }
      callInst->removeFromParent();
    }

    f->replaceAllUsesWith(newFunc);
    f->removeFromParent();
  }

  return llvm::PreservedAnalyses::none();
}
} // namespace lcl
