//===- GPUToSPIRV.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "passes.hpp"

#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"

#include "../../dialects/RawMemory/RawMemoryDialect.h"
#include "../../dialects/RawMemory/RawMemoryOps.h"
#include "../../dialects/RawMemory/RawMemoryTypes.h"
#include "mlir/Conversion/ArithmeticToSPIRV/ArithmeticToSPIRV.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct GPUToSPIRVPass
    : public PassWrapper<GPUToSPIRVPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUToSPIRVPass);
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    SmallVector<Operation *, 1> kernelModules;
    OpBuilder builder(context);
    module.walk([&builder, &kernelModules](gpu::GPUModuleOp moduleOp) {
      // For each kernel module (should be only 1 for now, but that is not a
      // requirement here), clone the module for conversion because the
      // gpu.launch function still needs the kernel module.
      builder.setInsertionPoint(moduleOp.getOperation());
      kernelModules.push_back(builder.clone(*moduleOp.getOperation()));
    });

    auto targetAttr = spirv::lookupTargetEnvOrDefault(module);
    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);
    target->addIllegalDialect<rawmem::RawMemoryDialect>();

    SPIRVTypeConverter typeConverter(targetAttr);
    RewritePatternSet patterns(context);
    populateGPUToSPIRVPatterns(typeConverter, patterns);

    // TODO: Change SPIR-V conversion to be progressive and remove the following
    // patterns.
    mlir::arith::populateArithmeticToSPIRVPatterns(typeConverter, patterns);
    populateMemRefToSPIRVPatterns(typeConverter, patterns);
    populateFuncToSPIRVPatterns(typeConverter, patterns);
    cf::populateControlFlowToSPIRVPatterns(typeConverter, patterns);

    lcl::populateRawMemoryToSPIRVTypeConversions(typeConverter, targetAttr);
    lcl::populateRawMemoryToSPIRVConversionPatterns(typeConverter, patterns);

    if (failed(
            applyFullConversion(kernelModules, *target, std::move(patterns))))
      return signalPassFailure();
  }
};

struct LoadPattern : public OpConversionPattern<rawmem::LoadOp> {
  using Base = OpConversionPattern<rawmem::LoadOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(rawmem::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // TODO access chain
    rewriter.replaceOpWithNewOp<spirv::LoadOp>(op, adaptor.addr());

    return success();
  }
};

struct StorePattern : public OpConversionPattern<rawmem::StoreOp> {
  using Base = OpConversionPattern<rawmem::StoreOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(rawmem::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<spirv::StoreOp>(op, adaptor.addr(),
                                                adaptor.value());

    return success();
  }
};

struct OffsetPattern : public OpConversionPattern<rawmem::OffsetOp> {
  using Base = OpConversionPattern<rawmem::OffsetOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(rawmem::OffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<spirv::PtrAccessChainOp>(
        op, adaptor.addr(), adaptor.offset(), ValueRange{});

    return success();
  }
};

struct ReinterpretCastPattern
    : public OpConversionPattern<rawmem::ReinterpretCastOp> {
  using Base = OpConversionPattern<rawmem::ReinterpretCastOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(rawmem::ReinterpretCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type type = getTypeConverter()->convertType(
        op.out().getType().cast<rawmem::PointerType>());
    rewriter.replaceOpWithNewOp<spirv::BitcastOp>(op, type, adaptor.addr());

    return success();
  }
};

struct FuncPattern : public OpConversionPattern<func::FuncOp> {
  using Base = OpConversionPattern<func::FuncOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fnType = funcOp.getFunctionType();

    // Update the signature to valid SPIR-V types and add the ABI
    // attributes. These will be "materialized" by using the
    // LowerABIAttributesPass.
    TypeConverter::SignatureConversion signatureConverter(
        fnType.getNumInputs());
    {
      for (const auto &argType :
           enumerate(funcOp.getFunctionType().getInputs())) {
        auto convertedType = getTypeConverter()->convertType(argType.value());
        signatureConverter.addInputs(argType.index(), convertedType);
      }
    }
    SmallVector<Type, 1> retTypes;
    for (auto t : fnType.getResults()) {
      retTypes.push_back(getTypeConverter()->convertType(t));
    }
    auto newFuncOp = rewriter.create<spirv::FuncOp>(
        funcOp.getLoc(), funcOp.getName(),
        rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                                 retTypes));
    for (const auto &namedAttr : funcOp->getAttrs()) {
      if (namedAttr.getName() == FunctionOpInterface::getTypeAttrName() ||
          namedAttr.getName() == SymbolTable::getSymbolAttrName())
        continue;
      newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(
            &newFuncOp.getBody(), *getTypeConverter(), &signatureConverter)))
      return failure();
    rewriter.eraseOp(funcOp);

    return success();
  }
};
} // namespace

static mlir::spirv::StorageClass
addrSpaceToVulkanStorageClass(unsigned addrSpace) {
  switch (addrSpace) {
  case 1:
    return spirv::StorageClass::CrossWorkgroup;
  default:
    llvm_unreachable("Unknown address space");
  }
}

static mlir::spirv::StorageClass
addrSpaceToStorageClass(unsigned addrSpace, mlir::spirv::TargetEnvAttr) {
  // TODO OpenCL?
  return addrSpaceToVulkanStorageClass(addrSpace);
}

void lcl::populateRawMemoryToSPIRVTypeConversions(
    mlir::TypeConverter &converter, mlir::spirv::TargetEnvAttr env) {
  converter.addConversion([=, &converter](rawmem::PointerType ptr) {
    auto sc = addrSpaceToStorageClass(ptr.getAddressSpace(), env);
    Type type;
    if (ptr.isOpaque()) {
      type = IntegerType::get(ptr.getContext(), 8);
    } else {
      type = converter.convertType(ptr.getElementType());
    }

    return spirv::PointerType::get(type, sc);
  });
}
void lcl::populateRawMemoryToSPIRVConversionPatterns(
    mlir::TypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<LoadPattern, StorePattern, OffsetPattern, ReinterpretCastPattern,
               FuncPattern>(converter, patterns.getContext());
}

std::unique_ptr<mlir::Pass> lcl::createGPUToSPIRVPass() {
  return std::make_unique<GPUToSPIRVPass>();
}
