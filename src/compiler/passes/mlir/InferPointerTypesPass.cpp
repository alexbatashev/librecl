//===- InferPointerTypesPass.cpp --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Struct/StructTypes.h"
#include "passes.hpp"

#include "RawMemory/RawMemoryDialect.h"
#include "RawMemory/RawMemoryOps.h"
#include "RawMemory/RawMemoryTypes.h"
#include "Struct/StructOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;

namespace {
static Type getCommonType(Type type, mlir::Value::user_range users) {
  Type commonType = type;

  bool isSupportedUser = llvm::all_of(users, [](auto user) {
    return llvm::isa<rawmem::ReinterpretCastOp>(user);
  });

  if (!isSupportedUser)
    return commonType;

  SmallVector<Type, 8> allTypes;

  for (auto user : users) {
    auto reinterpret = llvm::cast<rawmem::ReinterpretCastOp>(user);
    allTypes.push_back(reinterpret.getResult().getType());
  }

  // No active users, just skip
  if (allTypes.size() == 0)
    return type;

  bool sameType =
      llvm::all_of(allTypes, [&](Type t) { return t == allTypes.front(); });

  if (sameType) {
    type = allTypes.front();
  }

  return type;
}

struct UnrealizedPattern
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using Base = OpConversionPattern<UnrealizedConversionCastOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    rewriter.eraseOp(op);
    return success();
  }
};

/*
struct AddrOfPattern
    : public OpConversionPattern<structure::AddressOfOp> {
  using Base = OpConversionPattern<structure::AddressOfOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(structure::AddressOfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto oldPtrType = op.getResult().cast<rawmem::PointerType>();
    auto structType = op.addr().getType().cast<structure::StructType>();
    rewriter.replaceOpWithNewOp<>(Operation *op, Args &&args...)
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    rewriter.eraseOp(op);
    return success();
  }
};
*/

template <typename FuncT>
struct FunctionPattern : public OpConversionPattern<FuncT> {
  using Base = OpConversionPattern<FuncT>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncT func, typename FuncT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcType = func.getFunctionType();

    bool hasPointers = llvm::any_of(funcType.getInputs(), [](Type type) {
      return type.isa<rawmem::PointerType>();
    });

    if (!hasPointers) {
      return failure();
    }

    TypeConverter::SignatureConversion sigConversion{funcType.getNumInputs()};

    Region *oldRegion;
    if constexpr (std::is_same_v<func::FuncOp, FuncT>) {
      oldRegion = &func.getBody();
    } else {
      oldRegion = &func.getBody();
    }

    llvm::SmallSet<unsigned, 10> replacedArgs;

    for (auto inp : llvm::enumerate(funcType.getInputs())) {
      if (!inp.value().template isa<rawmem::PointerType>()) {
        sigConversion.addInputs(inp.index(), {inp.value()});
        continue;
      }

      auto ptrType = inp.value().template cast<rawmem::PointerType>();

      if (!ptrType.isOpaque()) {
        sigConversion.addInputs(inp.index(), {inp.value()});
        continue;
      }

      auto arg = oldRegion->getArgument(inp.index());

      Type argType = arg.getUsers().empty()
                         ? rawmem::PointerType::get(rewriter.getI8Type(), 1)
                         : getCommonType(inp.value(), arg.getUsers());

      if (argType != inp.value())
        replacedArgs.insert(inp.index());

      sigConversion.addInputs(inp.index(), argType);
    }

    auto newFunc = rewriter.create<FuncT>(
        func.getLoc(), func.getName(),
        rewriter.getFunctionType(sigConversion.getConvertedTypes(),
                                 funcType.getResults()));
    for (const auto &namedAttr : func->getAttrs()) {
      if (namedAttr.getName() == FunctionOpInterface::getTypeAttrName() ||
          namedAttr.getName() == SymbolTable::getSymbolAttrName())
        continue;
      newFunc->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    Region *newRegion;

    if constexpr (std::is_same_v<func::FuncOp, FuncT>) {
      newRegion = &newFunc.getBody();
    } else {
      newRegion = &newFunc.getBody();
    }

    BlockAndValueMapping mapping;
    {
      OpBuilder::InsertionGuard _{rewriter};
      rewriter.setInsertionPointToStart(&newRegion->front());
      for (unsigned idx = 0; idx < newRegion->getNumArguments(); idx++) {
        if (replacedArgs.count(idx)) {
          auto arg = newRegion->getArgument(idx);
          Value cast = rewriter.create<rawmem::ReinterpretCastOp>(
              func.getLoc(),
              rawmem::PointerType::get(
                  arg.getContext(),
                  arg.getType().cast<rawmem::PointerType>().getAddressSpace()),
              arg);
          mapping.map(oldRegion->getArgument(idx), cast);
        } else {
          mapping.map(oldRegion->getArgument(idx), newRegion->getArgument(idx));
        }
      }
    }

    rewriter.cloneRegionBefore(*oldRegion, *newRegion, newRegion->end(),
                               mapping);
    {
      OpBuilder::InsertionGuard _{rewriter};
      rewriter.setInsertionPointToEnd(&newRegion->front());
      rewriter.create<cf::BranchOp>(rewriter.getUnknownLoc(),
                                    &*std::next(newRegion->begin()));
    }

    rewriter.eraseOp(func);

    return success();
  }
};

static bool isArgLegal(Type type, mlir::Value::user_range users) {
  if (!type.isa<rawmem::PointerType>())
    return true;

  auto ptrType = type.cast<rawmem::PointerType>();
  if (!ptrType.isOpaque())
    return true;

  bool isReinterpret = llvm::all_of(users, [](auto user) {
    return llvm::isa<rawmem::ReinterpretCastOp>(user);
  });

  if (!isReinterpret)
    return true;

  SmallVector<rawmem::PointerType, 8> allTypes;

  for (auto user : users) {
    auto reinterpret = llvm::cast<rawmem::ReinterpretCastOp>(user);
    allTypes.push_back(
        reinterpret.getResult().getType().cast<rawmem::PointerType>());
  }

  bool sameType =
      llvm::all_of(allTypes, [&](Type t) { return t == allTypes.front(); });

  if (sameType) {
    return false;
  }

  return true;
}

Optional<bool> isAddrOfLegal(structure::AddressOfOp addrOf) {
  auto ptrType = addrOf.getResult().getType().cast<rawmem::PointerType>();

  return !ptrType.isOpaque();
}

template <typename FuncT> Optional<bool> isFunctionLegal(FuncT func) {
  auto funcType = func.getFunctionType();

  Region *body;
  if constexpr (std::is_same_v<func::FuncOp, FuncT>) {
    if (func.isDeclaration()) {
      return Optional(true);
    }
    body = &func.getBody();
  } else {
    body = &func.getBody();
  }

  SmallVector<bool, 10> isValid;
  for (auto inp : llvm::enumerate(funcType.getInputs())) {
    auto arg = body->getArgument(inp.index());

    isValid.push_back(isArgLegal(inp.value(), arg.getUsers()));
  }

  if (!llvm::all_of(isValid, [](bool v) { return v == true; })) {
    return false;
  }

  return true;
}

struct InferPointerTypesPass
    : public PassWrapper<InferPointerTypesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InferPointerTypesPass);

  void runOnOperation() override {
    ModuleOp module = getOperation();

    ConversionTarget target{*module.getContext()};

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });

    target.addLegalDialect<gpu::GPUDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<rawmem::RawMemoryDialect>();
    // target.addDynamicallyLegalOp<structure::AddressOfOp>(isAddrOfLegal);
    target.addDynamicallyLegalOp<func::FuncOp>(isFunctionLegal<func::FuncOp>);
    target.addDynamicallyLegalOp<gpu::GPUFuncOp>(
        isFunctionLegal<gpu::GPUFuncOp>);
    target.markOpRecursivelyLegal<gpu::GPUFuncOp>(
        isFunctionLegal<gpu::GPUFuncOp>);
    // target.addIllegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns{&getContext()};
    patterns
        .add<FunctionPattern<func::FuncOp>, FunctionPattern<gpu::GPUFuncOp>>(
            typeConverter, patterns.getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
    // TODO process calls here
  }
};
} // namespace

namespace lcl {
std::unique_ptr<Pass> createInferPointerTypesPass() {
  return std::make_unique<InferPointerTypesPass>();
}
} // namespace lcl
