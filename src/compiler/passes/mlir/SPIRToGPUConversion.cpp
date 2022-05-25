#include "passes.hpp"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct GPUTarget : public ConversionTarget {
  GPUTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<gpu::GPUDialect>();
    addLegalDialect<cf::ControlFlowDialect>();
    addLegalDialect<arith::ArithmeticDialect>();
    addLegalDialect<memref::MemRefDialect>();
    addLegalDialect<func::FuncDialect>();

    addIllegalDialect<LLVM::LLVMDialect>();
  }
};

// Copied from Builders.h
template <typename OpT>
RegisteredOperationName getCheckRegisteredInfo(MLIRContext *ctx) {
  Optional<RegisteredOperationName> opName =
      RegisteredOperationName::lookup(OpT::getOperationName(), ctx);
  if (LLVM_UNLIKELY(!opName)) {
    llvm::report_fatal_error(
        "Building op `" + OpT::getOperationName() +
        "` but it isn't registered in this MLIRContext: the dialect may not "
        "be loaded or this operation isn't registered by the dialect. See "
        "also https://mlir.llvm.org/getting_started/Faq/"
        "#registered-loaded-dependent-whats-up-with-dialects-management");
  }
  return *opName;
}

class ConvertSPIRToGPUPass
    : public PassWrapper<ConvertSPIRToGPUPass, mlir::OperationPass<ModuleOp>> {
  void runOnOperation() override {
    TypeConverter typeConverter{};

    GPUTarget target{getContext()};

    RewritePatternSet patterns{&getContext()};

    lcl::populateSPIRToGPUTypeConversions(typeConverter);
    lcl::populateSPIRToGPUConversionPatterns(typeConverter, patterns);

    ModuleOp module = getOperation();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct FuncConversionPattern : public OpConversionPattern<LLVM::LLVMFuncOp> {
  using Base = OpConversionPattern<LLVM::LLVMFuncOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::LLVMFuncOp func, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (func.getName().startswith("llvm.lifetime")) {
      rewriter.eraseOp(func);
      return success();
    }

    ModuleOp module = func->getParentOfType<ModuleOp>();

    gpu::GPUModuleOp gpuModule =
        module.lookupSymbol<gpu::GPUModuleOp>("ocl_program");

    if (!gpuModule) {
      mlir::OpBuilder::InsertionGuard _{rewriter};
      rewriter.setInsertionPointToStart(&module.getBodyRegion().front());
      gpuModule =
          rewriter.create<gpu::GPUModuleOp>(module.getLoc(), "ocl_program");
    }

    auto funcType = func.getFunctionType();

    TypeConverter &typeConverter = *getTypeConverter();
    TypeConverter::SignatureConversion signatureConverter(
        funcType.getNumParams());

    for (auto &t : llvm::enumerate(funcType.getParams())) {
      Type newType = typeConverter.convertType(t.value());
      signatureConverter.addInputs(t.index(), newType);
    }

    SmallVector<Type, 1> retTypes;
    if (!funcType.getReturnType().isa<LLVM::LLVMVoidType>())
      retTypes.push_back(
          getTypeConverter()->convertType(funcType.getReturnType()));

    auto gpuType = FunctionType::get(
        func.getContext(), signatureConverter.getConvertedTypes(), retTypes);

    StringRef name = func.getName();

    rewriter.setInsertionPoint(&gpuModule.body().front().back());

    if (!func.getBody().empty()) {
      auto gpuFunc =
          rewriter.create<gpu::GPUFuncOp>(func.getLoc(), name, gpuType);
      rewriter.inlineRegionBefore(func.getBody(), gpuFunc.getBody(),
                                  gpuFunc.getBody().end());
      // gpuFunc.body().back().erase();
      if (failed(rewriter.convertRegionTypes(&gpuFunc.getBody(), typeConverter,
                                             &signatureConverter))) {

        return failure();
      }
    } else {
      rewriter.create<func::FuncOp>(func.getLoc(), name, gpuType,
                                    rewriter.getStringAttr("private"));
    }

    rewriter.eraseOp(func);

    return success();
  }
};

template <typename LLVMTy, typename NewTy>
struct DirectConversion : public OpConversionPattern<LLVMTy> {
  using Base = OpConversionPattern<LLVMTy>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVMTy op, typename LLVMTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = this->getTypeConverter()->convertType(op.getType());
    if (!dstType)
      return failure();
    rewriter.template replaceOpWithNewOp<NewTy>(
        op, dstType, adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

struct LoadPattern : public OpConversionPattern<LLVM::LoadOp> {
  using Base = OpConversionPattern<LLVM::LoadOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO deal with pointers of pointers
    auto resType = getTypeConverter()->convertType(op.getRes().getType());
    MemRefType::Builder builder({-1}, resType);
    builder.setMemorySpace(
        op.getAddr().getType().cast<LLVM::LLVMPointerType>().getAddressSpace());
    MemRefType newType = builder;

    Value stride = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    Value size = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), -1);
    SmallVector<OpFoldResult, 1> sizes, strides;
    sizes.push_back(size);
    strides.push_back(stride);

    SmallVector<Value, 2> indices;

    // auto base = adaptor.getAddr().getDefiningOp();
    mlir::Value addr = adaptor.getAddr();

    /*
    if (llvm::isa<LLVM::GEPOp>(base)) {
      auto gep = llvm::cast<LLVM::GEPOp>(base);
      addr = gep.getBase();
      for (auto idx : gep.getIndices()) {
        indices.push_back(idx);
      }
    }
    */

    auto newAddr = rewriter.create<memref::ReinterpretCastOp>(
        op.getLoc(), newType, addr, rewriter.getIndexAttr(0), sizes, strides);

    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, newAddr, indices).dump();

    return success();
  }
};

struct StorePattern : public OpConversionPattern<LLVM::StoreOp> {
  using Base = OpConversionPattern<LLVM::StoreOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType = getTypeConverter()->convertType(op.getValue().getType());
    MemRefType::Builder builder({-1}, resType);
    builder.setMemorySpace(
        op.getAddr().getType().cast<LLVM::LLVMPointerType>().getAddressSpace());
    MemRefType newType = builder;

    Value stride = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    Value size = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), -1);
    SmallVector<OpFoldResult, 1> sizes, strides;
    sizes.push_back(size);
    strides.push_back(stride);

    SmallVector<Value, 2> indices;

    // auto base = adaptor.getAddr().getDefiningOp();
    mlir::Value addr = adaptor.getAddr();

    /*
    if (llvm::isa<LLVM::GEPOp>(base)) {
      auto gep = llvm::cast<LLVM::GEPOp>(base);
      addr = gep.getBase();
      for (auto idx : gep.getIndices()) {
        indices.push_back(idx);
      }
    }
    */

    auto newAddr = rewriter.create<memref::ReinterpretCastOp>(
        op.getLoc(), newType, addr, rewriter.getIndexAttr(0), sizes, strides);

    rewriter
        .replaceOpWithNewOp<memref::StoreOp>(op, adaptor.getValue(), newAddr,
                                             indices)
        .dump();

    return success();
  }
};

struct CallPattern : public OpConversionPattern<LLVM::CallOp> {
  using Base = OpConversionPattern<LLVM::CallOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Indirect calls are not supported yet
    if (!op.getCallee())
      return failure();

    auto calleeName = op.getCallee().getValue();

    if (calleeName.startswith("llvm.lifetime")) {
      rewriter.eraseOp(op);
      return success();
    }

    SmallVector<Type, 1> resTypes;
    for (auto t : op.getResultTypes()) {
      resTypes.push_back(getTypeConverter()->convertType(t));
    }
    rewriter.replaceOpWithNewOp<func::CallOp>(op, calleeName, resTypes,
                                              adaptor.getOperands());

    return success();
  }
};

struct ICmpPattern : public OpConversionPattern<LLVM::ICmpOp> {
  using Base = OpConversionPattern<LLVM::ICmpOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::ICmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    arith::CmpIPredicate pred;
    switch (op.getPredicate()) {
    case LLVM::ICmpPredicate::eq:
      pred = arith::CmpIPredicate::eq;
      break;
    case LLVM::ICmpPredicate::ne:
      pred = arith::CmpIPredicate::ne;
      break;
    case LLVM::ICmpPredicate::slt:
      pred = arith::CmpIPredicate::slt;
      break;
    case LLVM::ICmpPredicate::sle:
      pred = arith::CmpIPredicate::sle;
      break;
    case LLVM::ICmpPredicate::sgt:
      pred = arith::CmpIPredicate::sgt;
      break;
    case LLVM::ICmpPredicate::sge:
      pred = arith::CmpIPredicate::sge;
      break;
    case LLVM::ICmpPredicate::ult:
      pred = arith::CmpIPredicate::ult;
      break;
    case LLVM::ICmpPredicate::ule:
      pred = arith::CmpIPredicate::ule;
      break;
    case LLVM::ICmpPredicate::ugt:
      pred = arith::CmpIPredicate::ugt;
      break;
    case LLVM::ICmpPredicate::uge:
      pred = arith::CmpIPredicate::uge;
      break;
    }

    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, pred, adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

struct CondBrPattern : public OpConversionPattern<LLVM::CondBrOp> {
  using Base = OpConversionPattern<LLVM::CondBrOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::CondBrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, op.getCondition(), op.getTrueDest(), adaptor.getTrueDestOperands(),
        op.getFalseDest(), adaptor.getFalseDestOperands());
    return success();
  }
};

struct BrPattern : public OpConversionPattern<LLVM::BrOp> {
  using Base = OpConversionPattern<LLVM::BrOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::BrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(),
                                              adaptor.getDestOperands());
    return success();
  }
};

struct ReturnPattern : public OpConversionPattern<LLVM::ReturnOp> {
  using Base = OpConversionPattern<LLVM::ReturnOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<gpu::ReturnOp>(
        op, ValueRange{adaptor.getOperands()});
    return success();
  }
};

struct GEPPattern : public OpConversionPattern<LLVM::GEPOp> {
  using Base = OpConversionPattern<LLVM::GEPOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::GEPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO do something!!!
    rewriter.replaceOp(op, op.getBase());
    // rewriter.eraseOp(op);
    /*
    // TODO structures access is not supported yet
    if (!op.getBase().getType().isa<LLVM::LLVMPointerType>())
      return failure();

    SmallVector<Value, 4> indices;
    for (auto val : adaptor.getIndices()) {
      auto index = rewriter.create<arith::IndexCastOp>(op.getLoc(),
    rewriter.getIndexType(), val); indices.push_back(index);
    }
    auto resType =
    getTypeConverter()->convertType(op.getBase().getType()).cast<MemRefType>();

    Value stride = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    Value size = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), -1);

    rewriter.replaceOpWithNewOp<memref::SubViewOp>(op, resType,
    adaptor.getBase(), indices, ValueRange{size}, ValueRange{stride});
    */

    return success();
  }
};

struct AllocaPattern : public OpConversionPattern<LLVM::AllocaOp> {
  using Base = OpConversionPattern<LLVM::AllocaOp>;
  using Base::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::AllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType = getTypeConverter()->convertType(op.getRes().getType());
    auto index = rewriter.create<arith::IndexCastOp>(
        op.getLoc(), rewriter.getIndexType(), adaptor.getArraySize());
    // TODO fix alignment
    rewriter.replaceOpWithNewOp<memref::AllocaOp>(
        op, resType.cast<MemRefType>(), index.getOut());
    return success();
  }
};
}

namespace lcl {
void populateSPIRToGPUTypeConversions(TypeConverter &converter) {
  converter.addConversion([](LLVM::LLVMPointerType type) {
    Type elementType;
    // TODO are opaque pointers this big of a deal?
    if (type.isOpaque()) {
      elementType = IntegerType::get(type.getContext(), 8);
    } else {
      elementType = type.getElementType();
    }
    MemRefType::Builder builder({-1}, elementType);
    builder.setMemorySpace(type.getAddressSpace());

    return static_cast<MemRefType>(builder);
  });
  converter.addConversion([](IntegerType t) { return t; });
  converter.addConversion([](FloatType t) { return t; });
  converter.addConversion([](VectorType t) { return t; });
}

void populateSPIRToGPUConversionPatterns(TypeConverter &converter,
                                         RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
    FuncConversionPattern,
    DirectConversion<LLVM::ConstantOp, arith::ConstantOp>,
    DirectConversion<LLVM::SExtOp, arith::ExtSIOp>,
    DirectConversion<LLVM::TruncOp, arith::TruncIOp>,
    DirectConversion<LLVM::FAddOp, arith::AddFOp>,
    AllocaPattern,
    LoadPattern,
    StorePattern,
    CallPattern,
    ICmpPattern,
    CondBrPattern,
    BrPattern,
    ReturnPattern,
    GEPPattern
      // clang-format on
      >(converter, patterns.getContext());
}

std::unique_ptr<mlir::Pass> createSPIRToGPUPass() {
  return std::make_unique<ConvertSPIRToGPUPass>();
}
}
