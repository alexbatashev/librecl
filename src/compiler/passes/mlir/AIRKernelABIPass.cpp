#include "passes.hpp"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace {
struct AIRKernelABIPass
    : public mlir::PassWrapper<AIRKernelABIPass,
                               mlir::OperationPass<mlir::LLVM::LLVMFuncOp>> {
  void runOnOperation() override {
    mlir::LLVM::LLVMFuncOp func = getOperation();

    if (func.getCConv() != mlir::LLVM::CConv::SPIR_FUNC &&
        func.getCConv() != mlir::LLVM::CConv::SPIR_KERNEL) {
      return;
    }

    mlir::OpBuilder builder{func};

    auto oldType = func.getFunctionType();
    auto retType = func.getResultTypes();

    // TODO convert scalar types to pointers
    llvm::SmallVector<mlir::Type, 8> argTypes;

    for (auto &argType : llvm::enumerate(oldType.params())) {
      if (func.getCConv() == mlir::LLVM::CConv::SPIR_KERNEL &&
          !argType.value().isa<mlir::LLVM::LLVMPointerType>()) {
        mlir::OpBuilder::InsertionGuard _{builder};
        builder.setInsertionPointToStart(&func.getBody().front());

        mlir::Type ptrType =
            mlir::LLVM::LLVMPointerType::get(argType.value(), 1);
        argTypes.push_back(ptrType);

        if (func.isDeclaration() || func.getBody().empty()) {
          continue;
        }
        func.getBody().front().getArgument(argType.index()).setType(ptrType);

        auto load = builder.create<mlir::LLVM::LoadOp>(
            builder.getUnknownLoc(), func.getArgument(argType.index()));

        func.getArgument(argType.index()).replaceAllUsesExcept(load, {load});
      } else {
        argTypes.push_back(argType.value());
      }
    }

    mlir::Type indexType = mlir::VectorType::get({3}, builder.getI32Type(), 0);

    for (int i = 0; i < 4; i++) {
      argTypes.push_back(indexType);
    }

    // func.getBody().front().getArgument(0).replaceAllUsesExcept()

    auto newFuncType =
        mlir::LLVM::LLVMFunctionType::get(retType.front(), argTypes);

    func.setFunctionTypeAttr(mlir::TypeAttr::get(newFuncType));

    if (func.isDeclaration() || func.getBody().empty()) {
      return;
    }

    for (int i = 0; i < 4; i++)
      func.getBody().getBlocks().front().addArgument(indexType,
                                                     builder.getUnknownLoc());

    llvm::SmallVector<mlir::Value, 4> newVals;
    for (unsigned int i = newFuncType.getNumParams() - 4;
         i < newFuncType.getNumParams(); i++) {
      newVals.push_back(func.getArgument(i));
    }
    func.walk([&](mlir::LLVM::CallOp call) {
      if (!call.getCallee().hasValue())
        return;
      mlir::LLVM::LLVMFuncOp callee = llvm::cast<mlir::LLVM::LLVMFuncOp>(
          func->getParentOfType<mlir::ModuleOp>().lookupSymbol(
              call.getCallee().getValue()));
      if (callee.getCConv() != mlir::LLVM::CConv::SPIR_FUNC &&
          callee.getCConv() != mlir::LLVM::CConv::SPIR_KERNEL) {
        return;
      }
      call->insertOperands(call->getNumOperands(), newVals);
    });
  }
};
} // namespace

namespace lcl {
std::unique_ptr<mlir::Pass> createAIRKernelABIPass() {
  return std::make_unique<AIRKernelABIPass>();
}
} // namespace lcl
