//===- GPUToCppPass.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "passes.hpp"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct GPUReturnPattern : public OpConversionPattern<gpu::ReturnOp> {
  using Base = OpConversionPattern<gpu::ReturnOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);

    return success();
  }
};
struct GPUFuncPattern : public OpConversionPattern<gpu::GPUFuncOp> {
  using Base = OpConversionPattern<gpu::GPUFuncOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::GPUFuncOp func, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fnType = func.getFunctionType();

    TypeConverter::SignatureConversion signatureConverter(
        fnType.getNumInputs());
    {
      for (const auto &argType : enumerate(fnType.getInputs())) {
        signatureConverter.addInputs(argType.index(), argType.value());
      }
    }

    auto newFuncOp =
        rewriter.create<func::FuncOp>(func.getLoc(), func.getName(), fnType);
    newFuncOp.getBody().getBlocks().splice(newFuncOp.end(),
                                           func.getBody().getBlocks());

    newFuncOp->setAttr("msl_kernel", rewriter.getUnitAttr());

    for (size_t i = 0; i < fnType.getNumInputs(); i++) {
      newFuncOp.setArgAttrs(i, func.getArgAttrs(i));
    }

    rewriter.inlineRegionBefore(func.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(
            &newFuncOp.getBody(), *getTypeConverter(), &signatureConverter)))
      return failure();

    rewriter.eraseOp(func);

    return success();
  }
};

struct GPUToCppPass
    : public PassWrapper<GPUToCppPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUToCppPass);

  static constexpr llvm::StringLiteral getArgumentName() {
    return llvm::StringLiteral("convert-gpu-to-cpp");
  }
  llvm::StringRef getArgument() const override { return "convert-gpu-to-cpp"; }

  llvm::StringRef getDescription() const override {
    return "Convert MLIR GPU dialect to EmitC";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<emitc::EmitCDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    OpBuilder builder(context);

    builder.setInsertionPointToStart(module.getBody());

    builder.create<emitc::IncludeOp>(builder.getUnknownLoc(), "metal_stdlib",
                                     true);
    builder.create<emitc::IncludeOp>(builder.getUnknownLoc(), "simd/simd.h",
                                     true);

    module.walk([&builder](gpu::GPUModuleOp moduleOp) {
      moduleOp.walk([&builder, &moduleOp](func::FuncOp func) {
        builder.setInsertionPoint(moduleOp.getOperation());
        builder.clone(*func.getOperation());
      });
      moduleOp.walk([&builder, &moduleOp](gpu::GPUFuncOp func) {
        builder.setInsertionPoint(moduleOp.getOperation());
        builder.clone(*func.getOperation());
      });
      moduleOp.erase();
    });

    TypeConverter typeConverter{};
    typeConverter.addConversion([](Type type) { return type; });

    ConversionTarget target{getContext()};
    target.addIllegalOp<mlir::gpu::GPUFuncOp>();
    target.addLegalDialect<func::FuncDialect>();

    RewritePatternSet patterns{&getContext()};
    patterns.add<GPUFuncPattern, GPUReturnPattern>(typeConverter,
                                                   patterns.getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> lcl::createGPUToCppPass() {
  return std::make_unique<GPUToCppPass>();
}
