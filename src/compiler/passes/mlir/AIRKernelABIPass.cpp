#include "passes.hpp"

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace {
struct AIRKernelABIPass : public mlir::PassWrapper<AIRKernelABIPass, mlir::OperationPass<mlir::LLVM::LLVMFuncOp>> {
  void runOnOperation() {
    mlir::LLVM::LLVMFuncOp func = getOperation();

    mlir::OpBuilder builder{func};

    auto oldType = func.getFunctionType();
    auto retType = func.getResultTypes();

    // TODO convert scalar types to pointers
    llvm::SmallVector<mlir::Type, 8> argTypes{oldType.params().begin(), oldType.params().end()};

    for (int i = 0; i < 4; i++) {
      argTypes.push_back(mlir::VectorType::get({3}, builder.getI32Type(), 0));
    }

    auto newFuncType = mlir::LLVM::LLVMFunctionType::get(retType.front(), argTypes);

    auto newFunc = builder.create<mlir::LLVM::LLVMFuncOp>(func.getLoc(), func.getName(), newFuncType, func.getLinkage(), func.getDsoLocal(), func.getCConv(), func->getAttrs());

    /*
    TypeConverter::SignatureConversion signatureConverter(newFuncType.getNumInputs());
    for (const auto &argType : enumerate(oldType.getInputs())) {
      auto convertedType = newFuncType.getInputs()[argType.index()];
      signatureConverter.addInputs
    }
    */
  }
};
}

namespace lcl {
std::unique_ptr<mlir::Pass> createAIRKernelABIPass() {
  return std::make_unique<AIRKernelABIPass>();
}
}
