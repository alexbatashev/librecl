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
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
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

static mlir::spirv::StorageClass
addrSpaceToVulkanStorageClass(unsigned addrSpace);

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
    target->addLegalOp<UnrealizedConversionCastOp>();

    SPIRVTypeConverter typeConverter(targetAttr);
    RewritePatternSet patterns(context);
    populateGPUToSPIRVPatterns(typeConverter, patterns);

    // TODO: Change SPIR-V conversion to be progressive and remove the following
    // patterns.
    mlir::arith::populateArithmeticToSPIRVPatterns(typeConverter, patterns);
    populateMemRefToSPIRVPatterns(typeConverter, patterns);
    populateFuncToSPIRVPatterns(typeConverter, patterns);
    cf::populateControlFlowToSPIRVPatterns(typeConverter, patterns);
    ScfToSPIRVContext scfContext;
    populateSCFToSPIRVPatterns(typeConverter, scfContext, patterns);

    lcl::populateRawMemoryToSPIRVTypeConversions(typeConverter, targetAttr);
    lcl::populateRawMemoryToSPIRVConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(kernelModules, *target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

static bool isExRawPointer(Type type) {
  if (type.isa<spirv::PointerType>()) {
    auto outerPtr = type.cast<spirv::PointerType>();
    if (outerPtr.getPointeeType().isa<spirv::StructType>()) {
      auto structType = outerPtr.getPointeeType().cast<spirv::StructType>();
      if (structType.getNumElements() == 1 &&
          structType.getElementType(0).isa<spirv::RuntimeArrayType>()) {
        return true;
      }
    }
  }
  return false;
}

static Value generateAccessChain(Value base, ValueRange indices,
                                 ConversionPatternRewriter &rewriter) {
  SmallVector<Value, 2> realIndices;

  Value zero = rewriter.create<spirv::ConstantOp>(
      base.getLoc(), rewriter.getI64Type(),
      rewriter.getIntegerAttr(rewriter.getI64Type(), 0));

  if (isExRawPointer(base.getType())) {
    realIndices.push_back(zero);
  }
  if (indices.begin() == indices.end()) {
    realIndices.push_back(zero);
  }

  for (auto idx : indices)
    realIndices.push_back(idx);

  // TODO not sure if this loc is correct
  return rewriter.create<spirv::AccessChainOp>(base.getLoc(), base,
                                               realIndices);
}

struct LoadPattern : public OpConversionPattern<rawmem::LoadOp> {
  using Base = OpConversionPattern<rawmem::LoadOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(rawmem::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value baseAddr =
        generateAccessChain(adaptor.addr(), adaptor.indices(), rewriter);

    rewriter.replaceOpWithNewOp<spirv::LoadOp>(op, baseAddr);

    return success();
  }
};

struct StorePattern : public OpConversionPattern<rawmem::StoreOp> {
  using Base = OpConversionPattern<rawmem::StoreOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(rawmem::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value baseAddr =
        generateAccessChain(adaptor.addr(), adaptor.indices(), rewriter);
    rewriter.replaceOpWithNewOp<spirv::StoreOp>(op, baseAddr, adaptor.value());

    return success();
  }
};

struct OffsetPattern : public OpConversionPattern<rawmem::OffsetOp> {
  using Base = OpConversionPattern<rawmem::OffsetOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(rawmem::OffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value base = rewriter.create<spirv::ConstantOp>(
        op.getLoc(), rewriter.getI64Type(),
        rewriter.getIntegerAttr(rewriter.getI64Type(), 0));
    rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(
        op, adaptor.addr(), ValueRange{base, adaptor.offset()});
    // rewriter.replaceOpWithNewOp<spirv::PtrAccessChainOp>(
    //     op, adaptor.addr(), adaptor.offset(), ValueRange{});

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
    rawmem::PointerType ptrType =
        op.out().getType().cast<rawmem::PointerType>();
    Type resType;
    // TODO come up with a more elegant way
    /*
    if (llvm::isa<rawmem::OffsetOp>(op.out().getDefiningOp())) {
      Type elementType = ptrType.getElementType();
      resType =
    spirv::PointerType::get(getTypeConverter()->convertType(elementType),
    addrSpaceToVulkanStorageClass(ptrType.getAddressSpace())); } else {
    */
    resType = getTypeConverter()->convertType(ptrType);
    /*
  }
  */
    rewriter.replaceOpWithNewOp<spirv::BitcastOp>(op, resType, adaptor.addr());
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
    return spirv::StorageClass::StorageBuffer;
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

    // TODO add Block decoration
    spirv::StructType::MemberDecorationInfo dec(0, 0, spirv::Decoration::BufferBlock, 0);
    // TODO correct stride
    auto ptrElement =
        spirv::StructType::get({spirv::RuntimeArrayType::get(type, 4)}, {0}, {});

    return spirv::PointerType::get(ptrElement, sc);
    // return spirv::PointerType::get(type, sc);
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
