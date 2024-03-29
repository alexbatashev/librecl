//===- AIRKernelABIPass.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RawMemory/RawMemoryOps.h"
#include "RawMemory/RawMemoryTypes.h"
#include "passes.hpp"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace {
template <typename FuncTy> static void processFunction(FuncTy func) {
  mlir::OpBuilder builder{func};

  auto oldType = func.getFunctionType();
  auto retType = func.getResultTypes();

  // TODO convert scalar types to pointers
  llvm::SmallVector<mlir::Type, 8> argTypes;

  for (auto &argType : llvm::enumerate(oldType.getInputs())) {
    if (std::is_same_v<FuncTy, mlir::gpu::GPUFuncOp> &&
        !argType.value().template isa<mlir::rawmem::PointerType>()) {

      mlir::OpBuilder::InsertionGuard _{builder};
      builder.setInsertionPointToStart(&func.getBody().front());

      mlir::Type ptrType = mlir::rawmem::PointerType::get(argType.value(), 1);
      argTypes.push_back(ptrType);

      if (func.isDeclaration() || func.getBody().empty()) {
        continue;
      }

      func.getBody().front().getArgument(argType.index()).setType(ptrType);

      auto load = builder.create<mlir::rawmem::LoadOp>(
          builder.getUnknownLoc(), func.getArgument(argType.index()));

      func.getArgument(argType.index()).replaceAllUsesExcept(load, {load});
    } else {
      argTypes.push_back(argType.value());
    }
  }

  mlir::Type indexType =
      mlir::VectorType::get({3}, builder.getIntegerType(32, false), 0);

  for (int i = 0; i < 4; i++) {
    argTypes.push_back(indexType);
  }

  auto newFuncType =
      mlir::FunctionType::get(func.getContext(), argTypes, retType);

  if constexpr (std::is_same_v<mlir::gpu::GPUFuncOp, FuncTy>) {
    func.setFunctionTypeAttr(mlir::TypeAttr::get(newFuncType));
  } else if constexpr (std::is_same_v<mlir::func::FuncOp, FuncTy>) {
    func.setFunctionTypeAttr(mlir::TypeAttr::get(newFuncType));
  }

  if (func.isDeclaration() || func.getBody().empty()) {
    return;
  }

  for (int i = 0; i < 4; i++) {
    func.getBody().getBlocks().front().addArgument(indexType,
                                                   builder.getUnknownLoc());
    if constexpr (std::is_same_v<mlir::gpu::GPUFuncOp, FuncTy>) {
      switch (i) {
      case 0:
        func.setArgAttrs(
            newFuncType.getNumInputs() - 4 + i,
            {mlir::NamedAttribute{
                builder.getStringAttr("emitc.thread_position_in_grid"),
                builder.getUnitAttr()}});
        break;
      case 1:
        func.setArgAttrs(
            newFuncType.getNumInputs() - 4 + i,
            {mlir::NamedAttribute{
                builder.getStringAttr("emitc.thread_position_in_threadgroup"),
                builder.getUnitAttr()}});
        break;
      case 2:
        func.setArgAttrs(
            newFuncType.getNumInputs() - 4 + i,
            {mlir::NamedAttribute{
                builder.getStringAttr("emitc.threads_per_threadgroup"),
                builder.getUnitAttr()}});
        break;
      case 3:
        func.setArgAttrs(
            newFuncType.getNumInputs() - 4 + i,
            {mlir::NamedAttribute{
                builder.getStringAttr("emitc.threadgroups_per_grid"),
                builder.getUnitAttr()}});
        break;
      }
    }
  }

  llvm::SmallVector<mlir::Value, 4> newVals;
  for (unsigned int i = newFuncType.getNumInputs() - 4;
       i < newFuncType.getNumInputs(); i++) {
    newVals.push_back(func.getArgument(i));
  }
  func.walk([&](mlir::func::CallOp call) {
    call->insertOperands(call->getNumOperands(), newVals);
  });
}

struct AIRKernelABIPass
    : public mlir::PassWrapper<AIRKernelABIPass,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AIRKernelABIPass);

  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("air-kernel-abi");
  }
  ::llvm::StringRef getArgument() const override { return "air-kernel-abi"; }

  ::llvm::StringRef getDescription() const override {
    return "Rewrite functions honoring Metal Shading Language ABI";
  }

  void runOnOperation() override {
    mlir::gpu::GPUModuleOp module = getOperation();

    module.walk([](mlir::gpu::GPUFuncOp func) { processFunction(func); });
    module.walk([](mlir::func::FuncOp func) { processFunction(func); });
  }
};
} // namespace

namespace lcl {
std::unique_ptr<mlir::Pass> createAIRKernelABIPass() {
  return std::make_unique<AIRKernelABIPass>();
}
} // namespace lcl
