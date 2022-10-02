//===- ExpandGPUBuiltins.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibreCL/IR/LibreCLDialect.h"
#include "LibreCL/IR/LibreCLOps.h"
#include "passes.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct GloaglIDPattern : public OpConversionPattern<gpu::GlobalIdOp> {
  using Base = OpConversionPattern<gpu::GlobalIdOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::GlobalIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value gidArg;
    auto parent = op->getParentOfType<func::FuncOp>();
    if (parent) {
      // TODO replace 4 with a constant
      gidArg = parent.getArgument(parent.getNumArguments() - 4);
    }

    if (!gidArg)
      return failure();

    // TODO support gpu functions (in case of inlining)

    Value id;

    switch (op.getDimension()) {
    case gpu::Dimension::x:
      id = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                              rewriter.getIndexAttr(0));
      break;
    case gpu::Dimension::y:
      id = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                              rewriter.getIndexAttr(1));
      break;
    case gpu::Dimension::z:
      id = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                              rewriter.getIndexAttr(2));
      break;
    }

    Value extracted =
        rewriter.create<vector::ExtractElementOp>(op.getLoc(), gidArg, id);
    rewriter.replaceOpWithNewOp<lcl::AnyCastOp>(op, rewriter.getIndexType(),
                                                extracted);

    return success();
  }
};

struct ExpandGPUBuiltinsPass
    : public mlir::PassWrapper<ExpandGPUBuiltinsPass,
                               OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandGPUBuiltinsPass);

  static constexpr llvm::StringLiteral getArgumentName() {
    return llvm::StringLiteral("expand-gpu-builtins");
  }
  llvm::StringRef getArgument() const override { return "expand-gpu-builtins"; }

  llvm::StringRef getDescription() const override {
    return "Expand GPU dialect operations for Metal Shading Language";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    TypeConverter typeConverter{};
    typeConverter.addConversion([](Type type) { return type; });

    ConversionTarget target{getContext()};
    target.addIllegalOp<mlir::gpu::GlobalIdOp>();
    target.addLegalDialect<vector::VectorDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<lcl::LibreCLDialect>();

    RewritePatternSet patterns{&getContext()};
    patterns.add<GloaglIDPattern>(typeConverter, patterns.getContext());

    ModuleOp module = getOperation();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace lcl {
std::unique_ptr<mlir::Pass> createExpandGPUBuiltinsPass() {
  return std::make_unique<ExpandGPUBuiltinsPass>();
}
} // namespace lcl
