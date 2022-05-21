#include "passes.hpp"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <utility>

namespace lcl {
llvm::PreservedAnalyses AIRKernelABI::run(llvm::Module &module,
                                          llvm::ModuleAnalysisManager &AM) {
  llvm::SmallVector<llvm::Function *, 16> kernels;
  for (llvm::Function &f : module.functions()) {
    if (f.getCallingConv() != llvm::CallingConv::SPIR_KERNEL) {
      continue;
    }

    kernels.push_back(&f);
  }

  llvm::NamedMDNode *kernelsMD = module.getOrInsertNamedMetadata("air.kernel");
  llvm::SmallVector<llvm::Metadata *, 16> mds;
  for (auto &k : kernels) {
    llvm::MDBuilder builder(module.getContext());
    llvm::ConstantAsMetadata *func = builder.createConstant(k);
    auto *empty = llvm::MDTuple::get(module.getContext(), {});

    llvm::SmallVector<llvm::Metadata *, 8> args;

    for (auto &arg : llvm::enumerate(k->args())) {
      llvm::SmallVector<llvm::Metadata *, 8> argMDs;
      auto attr = k->getAttributeAtIndex(arg.index() + 1, "opencl.arg_type");
      auto argNo = llvm::ConstantInt::get(module.getContext(),
                                          llvm::APInt(32, arg.index()));
      argMDs.push_back(builder.createConstant(argNo));

      if (attr.isValid()) {
        std::string type = llvm::Twine("air." + attr.getValueAsString()).str();
        argMDs.push_back(llvm::MDString::get(module.getContext(), type));
        argMDs.push_back(
            llvm::MDString::get(module.getContext(), "air.arg_type_name"));
        argMDs.push_back(llvm::MDString::get(module.getContext(), "uint3"));
      } else {
        argMDs.push_back(
            llvm::MDString::get(module.getContext(), "air.buffer"));
        argMDs.push_back(
            llvm::MDString::get(module.getContext(), "air.location_index"));
        argMDs.push_back(builder.createConstant(argNo));
        auto idx =
            llvm::ConstantInt::get(module.getContext(), llvm::APInt(32, 1));
        argMDs.push_back(builder.createConstant(idx));
        argMDs.push_back(
            llvm::MDString::get(module.getContext(), "air.read_write"));
        // TODO arg size and alignment
      }

      args.push_back(llvm::MDTuple::get(module.getContext(), argMDs));
    }

    llvm::MDTuple *argsMD = llvm::MDTuple::get(module.getContext(), args);
    mds.push_back(
        llvm::MDTuple::get(module.getContext(), {func, empty, argsMD}));
  }
  llvm::MDNode *node = llvm::MDNode::get(module.getContext(), mds);
  kernelsMD->addOperand(node);

  for (auto &f : module.functions()) {
    for (auto &arg : llvm::enumerate(f.args())) {
      f.removeParamAttr(arg.index(), "opencl.arg_type");
      f.removeParamAttr(arg.index(), llvm::Attribute::NoUndef);
    }
    // Remove unsupported attributes
    f.removeFnAttr(llvm::Attribute::AttrKind::ArgMemOnly);
    f.removeFnAttr(llvm::Attribute::AttrKind::NoCallback);
    f.removeFnAttr(llvm::Attribute::AttrKind::NoFree);
    f.removeFnAttr(llvm::Attribute::AttrKind::NoSync);
    f.removeFnAttr(llvm::Attribute::AttrKind::NoUnwind);
    f.removeFnAttr(llvm::Attribute::AttrKind::WillReturn);
    f.removeFnAttr(llvm::Attribute::AttrKind::Convergent);
    f.removeFnAttr((llvm::Attribute::AttrKind)71);
    f.removeFnAttr("frame-pointer");
    f.removeFnAttr("min-legal-vector-width");
    f.removeFnAttr("no-trapping-math");
    f.removeFnAttr("uniform-work-group-size");

    llvm::errs() << "Kind: " << (int)llvm::Attribute::InAlloca << "\n";
  }

  llvm::SmallVector<llvm::CallInst *, 30> lifetime;
  for (auto &f : module.functions()) {
    for (auto &bb : f) {
      for (auto &inst : bb) {
        if (inst.getOpcode() == llvm::Instruction::Call) {
          auto &callInst = static_cast<llvm::CallInst &>(inst);
          if (callInst.getCalledFunction() &&
              callInst.getCalledFunction()->getName().startswith(
                  "llvm.lifetime")) {
            lifetime.push_back(&callInst);
          }
        }
      }
    }
  }

  for (auto &callInst : lifetime) {
    callInst->removeFromParent();
  }

  return llvm::PreservedAnalyses::none();
}
} // namespace lcl
